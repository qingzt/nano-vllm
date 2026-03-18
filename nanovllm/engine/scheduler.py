from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 将新的生成序列添加到等待队列中，准备进行调度
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # 获取等待队列中的第一个生成序列，检查是否可以将其加入当前的生成批次
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 如果可以加入生成批次，则分配kv缓存块给该序列，并更新当前批次的token数量
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 将该序列的状态更新为RUNNING，并从等待队列中移除，加入到正在生成的队列中，同时将其添加到当前调度的序列列表中
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 如果当前调度的序列列表不为空，则返回这些序列和一个标志表示这是prefill阶段
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 如果没有新的序列可以加入生成批次，则检查正在生成的队列中的序列，看看是否有序列已经完成或者可以继续生成
        while self.running and num_seqs < self.max_num_seqs:
            # 从正在生成的队列中获取第一个序列，检查是否可以继续生成
            seq = self.running.popleft()
            # 如果该序列已经完成或者不能继续生成，则将其从正在生成的队列中移除，并继续检查下一个序列
            while not self.block_manager.can_append(seq):
                # 如果该序列不能继续生成，则将其从正在生成的队列中移除，并继续检查下一个序列
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 如果正在生成的队列中没有其他序列可以继续生成，则将当前序列抢占出来，更新其状态为WAITING，并将其从正在生成的队列中移除，加入到等待队列的前面，同时释放其占用的kv缓存块，然后退出循环
                    self.preempt(seq)
                    break
            # 如果该序列可以继续生成，则将其加入当前调度的序列列表中，并更新当前批次的token数量
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 如果当前调度的序列列表不为空，则返回这些序列和一个标志表示这是decode阶段
        assert scheduled_seqs
        # 将当前调度的序列列表中的序列重新加入到正在生成的队列的前面，保持它们的生成顺序不变
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 将正在生成的序列抢占出来，更新其状态为WAITING，并将其从正在生成的队列中移除，加入到等待队列的前面，同时释放其占用的kv缓存块
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # postprocess方法用于在生成步骤完成后对生成序列进行后处理，更新它们的状态，并释放已经完成的序列占用的kv缓存块
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 将生成的token_id追加到序列的completion_token_ids中，并更新序列的状态
            # 如果生成的token_id是结束符，或者序列的completion_token_ids数量达到了max_tokens，则将序列的状态更新为FINISHED，并释放该序列占用的kv缓存块，同时将其从正在生成的队列中移除
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
