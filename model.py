import torch
import torch.nn as nn
from attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257, # 词汇表大小，被 BPE 分词器处理后的词汇数量
    "context_length": 1024, # 上下文长度，模型通过位置嵌入能够处理的最大输入词元数量
    "emb_dim": 768, # 词元嵌入的维度，可将每个词元转化为 768 维的向量
    "num_heads": 12, # 多头注意力中的注意力头的数量
    "num_layers": 12, # 模型中的 Transformer 块的数量
    "dropout": 0.1, # 用于 dropout 的丢弃率，防止过拟合
    "qkv_bias": False # 是否在 MultiHeadAttention 中使用可学习的偏置项
}

