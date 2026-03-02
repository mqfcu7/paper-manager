import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSTULayer(nn.Module):
    """
    HSTU (Hierarchical Sequential Transduction Unit) 层
    基于论文 "Actions Speak Louder than Words" 的设计
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        # 线性投影层
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)  # Q, K, V
        self.gate_proj = nn.Linear(model_dim, model_dim)     # U (门控)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        # 相对位置和时间偏差
        self.rel_pos_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()
        
    def forward(self, x, attention_mask=None):
        """
        前向传播
        x: [batch_size, seq_len, model_dim]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        """
        batch_size, seq_len, model_dim = x.shape
        
        # 残差连接
        residual = x
        
        # 层归一化
        x_norm = self.layer_norm(x)
        
        # 线性投影得到 Q, K, V, U
        qkv = self.qkv_proj(x_norm).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        u = self.silu(self.gate_proj(x_norm))  # [batch_size, seq_len, model_dim]
        u = u.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 缩放查询
        q = q * (self.head_dim ** -0.5)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        
        # 添加相对位置偏差
        attn_weights = attn_weights + self.rel_pos_bias
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # 使用点积聚合注意力而非softmax
        # 这是HSTU的一个关键创新，用点积聚合替代softmax以更好地捕获用户偏好强度
        attn_weights = torch.sigmoid(attn_weights)  # 使用sigmoid替代softmax
        
        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重新排列
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, model_dim)
        
        # 应用门控机制 (类似SwiGLU的变体)
        # 这是HSTU的另一个关键设计：通过Norm(A(X)V(X)) ⊙ U(X)进行门控
        gated_output = attn_output * u.transpose(1, 2).contiguous().reshape(batch_size, seq_len, model_dim)
        
        # 输出投影
        output = self.out_proj(gated_output)
        output = self.dropout(output)
        
        # 残差连接
        output = residual + output
        
        return output


class HSTUEncoder(nn.Module):
    """
    HSTU编码器，包含多个HSTU层
    """
    def __init__(self, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        
        # HSTU层堆叠
        self.layers = nn.ModuleList([
            HSTULayer(model_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, model_dim))  # 最大序列长度512
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        """
        前向传播
        x: [batch_size, seq_len, model_dim]
        attention_mask: [batch_size, seq_len, seq_len] or None
        """
        batch_size, seq_len, model_dim = x.shape
        
        # 添加位置编码
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        x = x + pos_encoding
        
        x = self.dropout(x)
        
        # 调整注意力掩码形状以适应多头注意力
        if attention_mask is not None:
            # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1)
        
        # 通过HSTU层
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x


class SequentialFeatureProcessor(nn.Module):
    """
    序列化特征处理器
    将异构特征空间统一到单一时间序列中
    """
    def __init__(self, embed_dim, max_categorical_cardinality=1000000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 为不同类型的特征创建嵌入层
        self.categorical_embedding = nn.Embedding(max_categorical_cardinality, embed_dim)
        self.numeric_to_embedding = nn.Linear(1, embed_dim)  # 用于数值特征
        
        # 用于处理不同特征时间序列的融合
        self.feature_fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, categorical_features=None, numeric_features=None):
        """
        处理异构特征
        categorical_features: [batch_size, seq_len] 或类似张量 (整数ID)
        numeric_features: [batch_size, seq_len] 或类似张量 (浮点数)
        """
        batch_size = None
        seq_len = None
        
        # 处理分类特征
        cat_embeds = None
        if categorical_features is not None:
            batch_size = categorical_features.size(0)
            seq_len = categorical_features.size(1) if len(categorical_features.shape) > 1 else categorical_features.size(-1)
            cat_embeds = self.categorical_embedding(categorical_features)
        
        # 处理数值特征
        num_embeds = None
        if numeric_features is not None:
            if batch_size is None:
                batch_size = numeric_features.size(0)
                seq_len = numeric_features.size(1) if len(numeric_features.shape) > 1 else numeric_features.size(-1)
            # 将数值特征扩展到embedding维度
            numeric_features_expanded = numeric_features.view(batch_size, -1, 1)  # 确保最后一个维度是1
            num_embeds = self.numeric_to_embedding(numeric_features_expanded)
        
        # 如果有两个特征类型，融合它们
        if cat_embeds is not None and num_embeds is not None:
            # 确保两个嵌入张量形状相同
            if cat_embeds.shape != num_embeds.shape:
                # 如果形状不匹配，进行扩展
                max_seq_len = max(cat_embeds.size(1), num_embeds.size(1))
                if cat_embeds.size(1) < max_seq_len:
                    cat_embeds = F.pad(cat_embeds, (0, 0, 0, max_seq_len - cat_embeds.size(1)))
                if num_embeds.size(1) < max_seq_len:
                    num_embeds = F.pad(num_embeds, (0, 0, 0, max_seq_len - num_embeds.size(1)))
            
            # 融合特征
            combined = torch.cat([cat_embeds, num_embeds], dim=-1)
            fused = self.feature_fusion(combined)
            return fused
        elif cat_embeds is not None:
            return cat_embeds
        elif num_embeds is not None:
            return num_embeds
        else:
            raise ValueError("At least one of categorical_features or numeric_features must be provided")


class HSTUForSequenceModeling(nn.Module):
    """
    HSTU模型用于序列建模
    适用于推荐系统的生成式建模
    """
    def __init__(self, 
                 vocab_size, 
                 model_dim=512, 
                 num_heads=8, 
                 num_layers=6, 
                 max_seq_len=512,
                 dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, model_dim)
        
        # 序列化特征处理器
        self.feature_processor = SequentialFeatureProcessor(model_dim)
        
        # HSTU编码器
        self.encoder = HSTUEncoder(model_dim, num_heads, num_layers, dropout)
        
        # 输出层
        self.output_proj = nn.Linear(model_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        with torch.no_grad():
            # 初始化嵌入层
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, 
                input_ids=None, 
                categorical_features=None, 
                numeric_features=None, 
                attention_mask=None):
        """
        前向传播
        input_ids: [batch_size, seq_len] - 词汇表ID
        categorical_features: [batch_size, seq_len] - 分类特征
        numeric_features: [batch_size, seq_len] - 数值特征
        attention_mask: [batch_size, seq_len] - 注意力掩码
        """
        # 处理输入特征
        if input_ids is not None:
            # 使用词汇表嵌入
            x = self.embedding(input_ids)
        else:
            # 使用序列化特征处理器
            x = self.feature_processor(categorical_features, numeric_features)
        
        # 创建注意力掩码 (如果是因果掩码)
        if attention_mask is None:
            batch_size, seq_len = x.shape[:2]
            # 创建因果掩码 (下三角矩阵)
            attention_mask = torch.tril(torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=x.device))
        else:
            # 将注意力掩码转换为布尔值
            attention_mask = attention_mask.bool()
        
        # 应用HSTU编码器
        encoded = self.encoder(x, attention_mask)
        
        # 应用输出投影
        logits = self.output_proj(encoded)
        
        return logits


class GenerativeRecommender(nn.Module):
    """
    生成式推荐系统
    基于HSTU架构实现
    """
    def __init__(self, 
                 vocab_size, 
                 model_dim=512, 
                 num_heads=8, 
                 num_layers=6, 
                 max_seq_len=512,
                 dropout=0.1):
        super().__init__()
        
        # HSTU模型
        self.hstu_model = HSTUForSequenceModeling(
            vocab_size, model_dim, num_heads, num_layers, max_seq_len, dropout
        )
        
        # 任务特定层 (多任务学习)
        self.task_layers = nn.ModuleDict({
            'ranking': nn.Linear(model_dim, 1),  # 排名任务
            'retrieval': nn.Linear(model_dim, 1)  # 检索任务
        })
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, 
                input_ids=None, 
                categorical_features=None, 
                numeric_features=None, 
                attention_mask=None,
                task='ranking'):
        """
        前向传播
        task: 'ranking' 或 'retrieval'
        """
        # 获取HSTU编码表示
        logits = self.hstu_model(input_ids, categorical_features, numeric_features, attention_mask)
        
        # 通过任务特定层
        if task in self.task_layers:
            output = self.task_layers[task](logits)
            output = self.sigmoid(output)  # 生成概率
        else:
            # 如果任务不在预定义列表中，返回原始logits
            output = logits
        
        return output


# 测试函数
def test_hstu():
    """
    测试HSTU模型
    """
    # 参数设置
    vocab_size = 10000  # 词汇表大小
    model_dim = 512     # 模型维度
    num_heads = 8       # 注意力头数
    num_layers = 6      # 层数
    batch_size = 4      # 批次大小
    seq_len = 32        # 序列长度
    
    print("创建HSTU模型...")
    
    # 创建模型
    model = GenerativeRecommender(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    print(f"输入形状: {input_ids.shape}")
    
    # 排名任务前向传播
    print("测试排名任务...")
    ranking_output = model(input_ids, attention_mask=attention_mask, task='ranking')
    print(f"排名输出形状: {ranking_output.shape}")
    
    # 检索任务前向传播
    print("测试检索任务...")
    retrieval_output = model(input_ids, attention_mask=attention_mask, task='retrieval')
    print(f"检索输出形状: {retrieval_output.shape}")
    
    # 测试序列化特征处理器
    print("测试序列化特征处理器...")
    categorical_features = torch.randint(0, 100, (batch_size, seq_len))
    numeric_features = torch.randn(batch_size, seq_len)
    
    feature_output = model.hstu_model.feature_processor(
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )
    print(f"特征处理输出形状: {feature_output.shape}")
    
    return model


if __name__ == "__main__":
    test_hstu()