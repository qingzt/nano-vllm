from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    # BlockManager的基本单位，包含一个block_id、一个ref_count、一个hash和一个token_ids列表
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    # 参数num_blocks表示BlockManager中Block的数量，block_size表示每个Block中token_ids的数量
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash_to_block_id用于根据hash值快速找到对应的block_id
        self.hash_to_block_id: dict[int, int] = dict()
        # free_block_ids用于存储空闲的block_id
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used_block_ids用于存储已使用的block_id
        self.used_block_ids: set[int] = set()

    @classmethod
    # compute_hash方法用于计算一个token_ids列表的hash值，prefix参数用于在计算hash时加入前一个block的hash值，以增加hash的唯一性
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # _allocate_block方法用于分配一个block_id，并将对应的Block重置为初始状态
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # _deallocate_block方法用于释放一个block_id，并将其加入free_block_ids中
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # can_allocate方法用于判断是否有足够的空闲block_id来分配给一个Sequence，seq.num_blocks表示Sequence需要的block数量
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate方法用于为一个Sequence分配block_id，并更新hash_to_block_id和Sequence的block_table
    def allocate(self, seq: Sequence):
        # 这里假设seq.block_table在调用allocate时是空的，即Sequence还没有分配过block_id
        assert not seq.block_table
        h = -1 # 用于存储当前block的hash值，初始值为-1表示没有hash
        cache_miss = False # 用于标记当前block是否命中缓存，初始值为False表示没有发生缓存未命中
        # 遍历Sequence的每个block
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 获取当前block的token_ids
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 计算当前block的hash值，如果当前block的token_ids数量不等于block_size，则不计算hash，直接设置为-1
            block_id = self.hash_to_block_id.get(h, -1) # 根据hash值查找对应的block_id，如果没有找到则返回-1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # 如果没有找到对应的block_id，或者找到了但对应的Block中的token_ids与当前block的token_ids不匹配，则说明发生了缓存未命中
                cache_miss = True
            if cache_miss: # 如果发生了缓存未命中，则需要分配一个新的block_id
                block_id = self.free_block_ids[0] # 从free_block_ids中获取一个空闲的block_id
                block = self._allocate_block(block_id) # 分配这个block_id，并获取对应的Block对象
            else: # 如果没有发生缓存未命中，则说明找到了一个可以复用的Block
                seq.num_cached_tokens += self.block_size # 更新Sequence的num_cached_tokens，增加一个block_size的数量
                if block_id in self.used_block_ids: # 如果这个block_id已经在used_block_ids中，说明这个Block正在被使用，可以直接复用
                    block = self.blocks[block_id] # 获取这个Block对象
                    block.ref_count += 1 # 增加这个Block的引用计数
                else:
                    block = self._allocate_block(block_id) # 否则，这个block_id虽然在hash_to_block_id中，但并没有被分配出去，需要先分配这个block_id
            if h != -1:
                block.update(h, token_ids) # 更新这个Block的hash值和token_ids
                self.hash_to_block_id[h] = block_id # 更新hash_to_block_id字典，将这个hash值映射到这个block_id
            seq.block_table.append(block_id) # 将这个block_id添加到Sequence的block_table中
    # deallocate方法用于释放一个Sequence占用的block_id，并更新Sequence的num_cached_tokens和block_table
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id] # 获取这个block_id对应的Block对象
            block.ref_count -= 1 # 减少这个Block的引用计数
            if block.ref_count == 0: # 如果这个Block的引用计数变为0，说明没有Sequence在使用这个Block了，可以将这个block_id加入free_block_ids中
                self._deallocate_block(block_id) # 释放这个block_id
        seq.num_cached_tokens = 0 # 将Sequence的num_cached_tokens重置为0
        seq.block_table.clear() # 清空Sequence的block_table

    # can_append方法用于判断是否可以在当前BlockManager的状态下将一个新的token_id追加到一个Sequence中
    def can_append(self, seq: Sequence) -> bool:
        # 条件是当前BlockManager中至少有一个空闲的block_id，并且这个Sequence的长度加1后需要的block数量不超过当前BlockManager中空闲的block数量
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # may_append方法用于在当前BlockManager的状态下将一个新的token_id追加到一个Sequence中
    def may_append(self, seq: Sequence):
        block_table = seq.block_table # 获取这个Sequence的block_table
        last_block = self.blocks[block_table[-1]] # 获取这个Sequence的最后一个block_id对应的Block对象
        if len(seq) % self.block_size == 1: # 如果这个Sequence的长度加1后需要的block数量增加了1，说明需要分配一个新的block_id来存储这个新的token_id
            assert last_block.hash != -1 # 这个时候最后一个Block应该已经有了hash值，因为它已经满了
            block_id = self.free_block_ids[0] # 从free_block_ids中获取一个空闲的block_id
            self._allocate_block(block_id) # 分配这个block_id
            block_table.append(block_id) # 将这个block_id添加到Sequence的block_table中
        elif len(seq) % self.block_size == 0: # 如果这个Sequence的长度加1后需要的block数量没有增加，说明可以直接在当前的最后一个Block中追加这个新的token_id
            assert last_block.hash == -1 # 这个时候最后一个Block应该没有hash值，因为它还没有满
            token_ids = seq.block(seq.num_blocks-1) # 获取这个Sequence的最后一个block的token_ids
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # 获取这个Sequence的倒数第二个block的hash值作为prefix，如果没有倒数第二个block，则prefix为-1
            h = self.compute_hash(token_ids, prefix) # 计算这个Sequence的最后一个block的hash值，使用prefix来增加hash的唯一性
            last_block.update(h, token_ids) # 更新这个Block的hash值和token_ids
            self.hash_to_block_id[h] = last_block.block_id # 更新hash_to_block_id字典，将这个hash值映射到这个block_id
        else:
            assert last_block.hash == -1 # 这个时候最后一个Block应该没有hash值，因为它还没有满
