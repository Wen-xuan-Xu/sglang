import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import maybe_strip_thinking_tokens
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class _MockReqToTokenPool:
    def __init__(self, max_reqs: int = 8, max_seq_len: int = 256):
        self.req_to_token = torch.zeros((max_reqs, max_seq_len), dtype=torch.int64)
        self.device = torch.device("cpu")

    def write(self, key, value):
        req_idx, sl = key
        self.req_to_token[req_idx, sl] = value


class _MockAllocator:
    def __init__(self):
        self.freed: list[torch.Tensor] = []
        self.device = torch.device("cpu")

    def alloc(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.int64)

    def free(self, indices: torch.Tensor):
        if isinstance(indices, torch.Tensor) and indices.numel() > 0:
            self.freed.append(indices.clone())

    def available_size(self) -> int:
        return 1_000_000


class _PageTrackingAllocator(_MockAllocator):
    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size
        self.page_counts: dict[int, int] = {}

    def free(self, indices: torch.Tensor):
        if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
            return
        pages = torch.unique(indices // self.page_size)
        self.freed.append(pages.clone())
        for page in pages.tolist():
            self.page_counts[page] = self.page_counts.get(page, 0) + 1


class _MockReq:
    def __init__(self):
        self._kv_committed_len = 0
        self.kv_committed_freed = False
        self.origin_input_ids: list[int] = []
        self.output_ids: list[int] = []
        self.fill_ids: list[int] = []
        self.req_pool_idx = 0
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len = 0
        self.priority = 0
        self.reasoning_tokens = 0
        self.strip_thinking_from_cache = True
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.swa_uuid_for_lock = None
        self.swa_evicted_seqlen = 0

    def pop_committed_kv_cache(self):
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self._kv_committed_len


PROMPT = [10, 20, 30]
THINKING = [50, 51, 52]
ANSWER = [200, 201]
OUTPUT_WITH_THINKING = THINKING + ANSWER


def _build_radix_cache(page_size: int = 1, allocator=None):
    allocator = allocator or _MockAllocator()
    pool = _MockReqToTokenPool()
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
        )
    )
    return cache, pool, allocator


def _prepare_req(
    cache,
    pool,
    prompt,
    output,
    *,
    fill_ids=None,
    reasoning_tokens=0,
    strip_thinking_from_cache=True,
    kv_base=100,
):
    req = _MockReq()
    req.origin_input_ids = list(prompt)
    req.output_ids = list(output)
    req.fill_ids = list(fill_ids if fill_ids is not None else prompt + output)
    req.reasoning_tokens = reasoning_tokens
    req.strip_thinking_from_cache = strip_thinking_from_cache
    req._kv_committed_len = len(prompt) + len(output)
    req.last_node = cache.root_node

    total_len = len(req.fill_ids)
    pool.req_to_token[req.req_pool_idx, :total_len] = torch.arange(
        kv_base, kv_base + total_len, dtype=torch.int64
    )
    return req


def _flatten_freed(allocator):
    result = set()
    for tensor in allocator.freed:
        result.update(tensor.tolist())
    return result


class TestMaybeStripThinkingTokens(unittest.TestCase):
    def test_returns_prompt_len_for_separated_reasoning(self):
        req = _MockReq()
        req.origin_input_ids = [1, 2, 3, 4]
        req.reasoning_tokens = 6
        self.assertEqual(maybe_strip_thinking_tokens(req), 4)

    def test_caps_to_current_token_length(self):
        req = _MockReq()
        req.origin_input_ids = [1, 2, 3, 4]
        req.reasoning_tokens = 2
        self.assertEqual(maybe_strip_thinking_tokens(req, 2), 2)

    def test_disabled_gating_keeps_full_cache(self):
        req = _MockReq()
        req.origin_input_ids = [1, 2]
        req.reasoning_tokens = 2
        req.strip_thinking_from_cache = False
        self.assertIsNone(maybe_strip_thinking_tokens(req))

    def test_zero_reasoning_tokens_does_not_strip(self):
        req = _MockReq()
        req.origin_input_ids = [1, 2]
        self.assertIsNone(maybe_strip_thinking_tokens(req))


class TestRadixCacheThinking(unittest.TestCase):
    def test_cache_finished_strips_to_prompt_prefix(self):
        cache, pool, allocator = _build_radix_cache()
        req = _prepare_req(
            cache,
            pool,
            PROMPT,
            OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING),
            kv_base=100,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(PROMPT))
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT + OUTPUT_WITH_THINKING))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT))
        self.assertEqual(_flatten_freed(allocator), set(range(103, 108)))

    def test_cache_finished_preserves_non_reasoning_behavior(self):
        cache, pool, allocator = _build_radix_cache()
        all_tokens = PROMPT + OUTPUT_WITH_THINKING
        req = _prepare_req(cache, pool, PROMPT, OUTPUT_WITH_THINKING, kv_base=10)

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(all_tokens))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        self.assertEqual(len(match.device_indices), len(all_tokens))
        self.assertEqual(_flatten_freed(allocator), set())

    def test_cache_unfinished_does_not_insert_output_tokens(self):
        cache, pool, _allocator = _build_radix_cache()
        req = _prepare_req(
            cache,
            pool,
            PROMPT,
            OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING),
            kv_base=0,
        )

        cache.cache_unfinished_req(req)

        self.assertEqual(cache.total_size(), len(PROMPT))
        self.assertEqual(len(req.prefix_indices), len(PROMPT) + len(OUTPUT_WITH_THINKING))
        torch.testing.assert_close(
            req.prefix_indices[len(PROMPT) :],
            torch.arange(len(PROMPT), len(PROMPT) + len(OUTPUT_WITH_THINKING)),
        )
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT + OUTPUT_WITH_THINKING))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT))

    def test_paged_boundary_page_freed_once(self):
        allocator = _PageTrackingAllocator(page_size=4)
        cache, pool, _ = _build_radix_cache(page_size=4, allocator=allocator)
        prompt = [1, 2, 3, 4, 5, 6]
        output = [7, 8, 9, 10]
        req = _prepare_req(
            cache,
            pool,
            prompt,
            output,
            reasoning_tokens=2,
            kv_base=0,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), 4)
        self.assertEqual(allocator.page_counts.get(1, 0), 1)


if __name__ == "__main__":
    unittest.main()
