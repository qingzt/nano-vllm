import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        # 初始化config
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 存储分布式进程和事件
        self.ps = []
        self.events = []
        # 使用spawn方式创建多进程上下文
        ctx = mp.get_context("spawn")
        # tp的多进程管理
        for i in range(1, config.tensor_parallel_size):
            event = (
                ctx.Event()
            )  # 创建一个进程间同步事件，用于主进程和worker进程之间的通信与同步
            process = ctx.Process(
                target=ModelRunner, args=(config, i, event)
            )  # 创建一个新的worker进程，目标函数是ModelRunner，参数包括配置、进程编号i、同步事件
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 主进程的ModelRunner实例
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id # 设置终止token id
        # 初始化调度器
        self.scheduler = Scheduler(config)
        # 注册退出函数，确保进程正确关闭
        atexit.register(self.exit)

    def exit(self):
        # 通知所有worker进程退出
        self.model_runner.call("exit")
        del self.model_runner
        # 等待所有worker进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果输入的prompt是字符串，则使用分词器将其编码为token id列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 创建一个新的生成序列对象，包含输入的prompt和采样参数
        seq = Sequence(prompt, sampling_params)
        # 将生成序列添加到调度器中，准备进行生成
        self.scheduler.add(seq)

    def step(self):
        # 调用调度器的schedule方法，获取当前可以生成的序列列表和是否是prefill阶段的标志
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 调用model_runner的run方法，传入当前可以生成的序列列表和prefill阶段的标志，获取生成的token id列表
        self.scheduler.postprocess(seqs, token_ids) # 调用调度器的postprocess方法，传入当前生成的序列列表和生成的token id列表，进行生成结果的后处理
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 生成接口，接受多个prompt和对应的采样参数，返回生成结果列表
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            # 如果传入的采样参数不是列表，则将其复制为与prompts长度相同的列表
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 将每个prompt和对应的采样参数添加到调度器中，准备生成
            self.add_request(prompt, sp)
        # outputs字典用于存储生成结果，prefill_throughput和decode_throughput用于计算生成速度
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        # 循环直到所有生成序列都完成
        while not self.is_finished():
            t = perf_counter()
            # 调用step方法执行生成步骤，获取生成结果和生成的token数量
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
