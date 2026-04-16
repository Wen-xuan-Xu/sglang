import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    MiniMaxAppendThinkDetector,
    ReasoningParser,
)
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

    def free(self, indices: torch.Tensor):
        if isinstance(indices, torch.Tensor) and indices.numel() > 0:
            self.freed.append(indices.clone())

    def available_size(self) -> int:
        return 1_000_000


class _MockReq:
    def __init__(self):
        self._kv_committed_len = 0
        self.kv_committed_freed = False
        self.origin_input_ids = []
        self.output_ids = []
        self.fill_ids = []
        self.req_pool_idx = 0
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len = 0
        self.priority = 0
        self.reasoning_tokens = 0
        self.strip_thinking_from_cache = True
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)

    def pop_committed_kv_cache(self):
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self._kv_committed_len


PROMPT = [10, 20, 30]
THINKING = [50, 51, 52]
ANSWER = [200, 201]
OUTPUT_WITH_THINKING = THINKING + ANSWER


def _build_cache():
    allocator = _MockAllocator()
    pool = _MockReqToTokenPool()
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return cache, pool, allocator


def _prepare_req(
    cache,
    pool,
    prompt,
    output,
    *,
    reasoning_tokens=0,
    strip_thinking_from_cache=True,
    kv_base=100,
):
    req = _MockReq()
    req.origin_input_ids = list(prompt)
    req.output_ids = list(output)
    req.fill_ids = list(prompt + output)
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


class TestReasoningParserGating(unittest.TestCase):
    def test_base_detector_strips_by_default(self):
        self.assertTrue(BaseReasoningFormatDetector.strip_thinking_from_cache)

    def test_minimax_append_think_keeps_cache(self):
        self.assertFalse(MiniMaxAppendThinkDetector.strip_thinking_from_cache)

    def test_qwen3_parser_strips(self):
        parser = ReasoningParser(model_type="qwen3", stream_reasoning=False)
        self.assertTrue(parser.detector.strip_thinking_from_cache)

    def test_minimax_append_think_parser_keeps_output(self):
        parser = ReasoningParser(
            model_type="minimax-append-think", stream_reasoning=False
        )
        self.assertFalse(parser.detector.strip_thinking_from_cache)


class TestGatedCachingBehavior(unittest.TestCase):
    def test_strip_false_keeps_full_output_cached(self):
        cache, pool, allocator = _build_cache()
        req = _prepare_req(
            cache,
            pool,
            PROMPT,
            OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING),
            strip_thinking_from_cache=False,
            kv_base=10,
        )

        cache.cache_finished_req(req, is_insert=True)

        all_tokens = PROMPT + OUTPUT_WITH_THINKING
        self.assertEqual(cache.total_size(), len(all_tokens))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        self.assertEqual(len(match.device_indices), len(all_tokens))
        self.assertEqual(_flatten_freed(allocator), set())

    def test_strip_false_supports_minimax_multiturn_prefix_match(self):
        cache, pool, _allocator = _build_cache()
        req = _prepare_req(
            cache,
            pool,
            PROMPT,
            OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING),
            strip_thinking_from_cache=False,
            kv_base=0,
        )

        cache.cache_finished_req(req, is_insert=True)

        turn1 = PROMPT + OUTPUT_WITH_THINKING
        turn2 = turn1 + [300, 301]
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(turn2)))
        self.assertEqual(len(match.device_indices), len(turn1))

    def test_per_request_flag_is_respected(self):
        cache, pool, _allocator = _build_cache()
        req1 = _prepare_req(
            cache,
            pool,
            [1, 2],
            [3, 4, 5],
            reasoning_tokens=2,
            strip_thinking_from_cache=True,
            kv_base=100,
        )
        cache.cache_finished_req(req1, is_insert=True)

        req2 = _prepare_req(
            cache,
            pool,
            [6, 7],
            [8, 9, 10],
            reasoning_tokens=2,
            strip_thinking_from_cache=False,
            kv_base=200,
        )
        req2.req_pool_idx = 1
        req2.last_node = cache.root_node
        pool.req_to_token[1, :5] = torch.arange(200, 205, dtype=torch.int64)

        cache.cache_finished_req(req2, is_insert=True)

        self.assertEqual(cache.total_size(), 7)

    def test_reasoning_tokens_zero_keeps_old_behavior(self):
        cache, pool, _allocator = _build_cache()
        req = _prepare_req(
            cache,
            pool,
            PROMPT,
            ANSWER,
            reasoning_tokens=0,
            strip_thinking_from_cache=False,
        )

        cache.cache_finished_req(req, is_insert=True)
        self.assertEqual(cache.total_size(), len(PROMPT) + len(ANSWER))


if __name__ == "__main__":
    unittest.main()
