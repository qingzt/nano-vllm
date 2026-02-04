import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str # 模型目录或名称
    max_num_batched_tokens: int = 16384 # 单批次最大token数
    max_num_seqs: int = 512 # 单批次最大序列数
    max_model_len: int = 4096 # 单序列最大长度
    gpu_memory_utilization: float = 0.8 # GPU显存利用率上限
    tensor_parallel_size: int = 1 # 张量并行进程数
    enforce_eager: bool = False # 是否强制eager模式
    hf_config: AutoConfig | None = None # transformers的模型配置
    eos: int = -1 # 终止token id
    kvcache_block_size: int = 256 # KV缓存块大小
    num_kvcache_blocks: int = -1 # KV缓存块数量

    def __post_init__(self):
        assert os.path.isdir(self.model) # 检查模型目录是否存在
        assert self.kvcache_block_size % 256 == 0 # KV缓存块大小必须为256的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 并行进程数必须在1~8之间
        self.hf_config = AutoConfig.from_pretrained(self.model) # 加载transformers模型配置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 限制最大长度不超过模型支持
        assert self.max_num_batched_tokens >= self.max_model_len # 批次token数不能小于单序列最大长度
