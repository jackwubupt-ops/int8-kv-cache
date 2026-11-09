import json
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "stage4" / "l1_model_check.py"
_spec = importlib.util.spec_from_file_location("l1_model_check", _PATH)
L1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L1)


def test_compare_identical_is_perfect():
    ref = [[(-0.1, 5), (-2.0, 9)], [(-0.3, 1)]]
    out = L1.compare_token_logprobs(ref, ref)
    assert out == {"top1_agreement": 1.0, "top1_per_prompt_mean": 1.0, "mean_abs_logprob_diff": 0.0, "n_tokens": 3}


def test_compare_counts_mismatch():
    ref = [[(-0.1, 5), (-2.0, 9), (-0.5, 3), (-1.0, 4)]]
    other = [[(-0.2, 5), (-2.0, 8), (-0.5, 3), (-1.0, 4)]]
    out = L1.compare_token_logprobs(ref, other)
    assert out["top1_agreement"] == 0.75 and out["n_tokens"] == 4
    assert out["mean_abs_logprob_diff"] == pytest.approx(0.1 / 4)


def test_compare_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        L1.compare_token_logprobs([[(-0.1, 5)]], [[(-0.1, 5), (-0.2, 6)]])


def test_compare_reports_per_prompt_mean_like_stage3():
    ref = [[(-0.1, 1), (-0.1, 2)], [(-0.1, 3), (-0.1, 4), (-0.1, 5), (-0.1, 6)]]
    other = [[(-0.1, 1), (-0.1, 9)], [(-0.1, 3), (-0.1, 4), (-0.1, 5), (-0.1, 6)]]
    out = L1.compare_token_logprobs(ref, other)
    assert out["top1_agreement"] == pytest.approx(5 / 6)
    assert out["top1_per_prompt_mean"] == pytest.approx((0.5 + 1.0) / 2)


def test_step_score_pairs_reference_logprob_with_top1():
    meta = {"output_top_logprobs": [[(-0.2, 42, None)]], "output_token_ids_logprobs": [[(-0.9, 7, None)]]}
    assert L1.step_score(meta, ref_token=7) == (-0.9, 42)


def test_step_score_rejects_wrong_reference_token():
    meta = {"output_top_logprobs": [[(-0.2, 42, None)]], "output_token_ids_logprobs": [[(-0.9, 8, None)]]}
    with pytest.raises(ValueError, match="reference token"):
        L1.step_score(meta, ref_token=7)


class _StepEngine:
    """Records requests; top-1 is the reference token and its logprob is minus the input length."""

    def __init__(self):
        self.calls = []

    def generate(self, *, input_ids, sampling_params, return_logprob, logprob_start_len, top_logprobs_num,
                 token_ids_logprob):
        self.calls.append((input_ids, logprob_start_len, token_ids_logprob))
        return [{"meta_info": {"output_top_logprobs": [[(-1.0, tid[0], None)]],
                               "output_token_ids_logprobs": [[(-float(len(ids)), tid[0], None)]]}}
                for ids, tid in zip(input_ids, token_ids_logprob)]


def test_forced_decode_feeds_one_token_per_step_over_a_cached_prefix():
    engine = _StepEngine()
    out = L1.forced_decode_scores(engine, [("short", [1, 2]), ("short", [5])], [[7, 8], [9]])
    assert out == [[(-2.0, 7), (-3.0, 8)], [(-1.0, 9)]]
    first, second = engine.calls
    # logprob_start_len = len - 1 lets SGLang reuse every earlier token from the KV cache.
    assert first == ([[1, 2], [5]], [1, 0], [[7], [9]])
    assert second == ([[1, 2, 7]], [2], [[8]])


def test_generation_agreement_counts_exact_matches_and_first_divergence():
    out = L1.generation_agreement([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 9, 6]])
    assert out == {"exact_rate": 0.5, "mean_first_divergence": 1.0}


def test_int8_verdict_compares_with_stage3_on_the_same_prompts():
    assert L1.verdict(0.985, expected=0.9879, tol=0.005, noise_floor=1.0) == "PASS"
    assert L1.verdict(0.9684, expected=0.9879, tol=0.005, noise_floor=1.0) == "FAIL"
    assert L1.verdict(0.995, expected=0.9879, tol=0.005, noise_floor=0.99) == "INVALID"


def _stage3_metrics(tmp_path, rows):
    path = tmp_path / "model_metrics.json"
    path.write_text(json.dumps({"results": {"per_token_head:dequant": {"per_prompt": rows}}}))
    return path


def test_stage3_expected_averages_the_matching_prompts(tmp_path):
    rows = [{"kind": "short", "top1": 1.0}, {"kind": "short", "top1": 0.9}, {"kind": "short", "top1": 0.0},
            {"kind": "needle", "top1": 0.6}, {"kind": "needle", "top1": 0.0}]
    expected = L1.stage3_expected(_stage3_metrics(tmp_path, rows), n_short=2, n_needle=1)
    assert expected == pytest.approx({"short": 0.95, "needle": 0.6, "all": (1.0 + 0.9 + 0.6) / 3})


def test_stage3_expected_rejects_prompts_stage3_never_ran(tmp_path):
    rows = [{"kind": "short", "top1": 1.0}, {"kind": "needle", "top1": 1.0}]
    with pytest.raises(ValueError, match="needle"):
        L1.stage3_expected(_stage3_metrics(tmp_path, rows), n_short=1, n_needle=2)


def test_engine_kwargs_disable_chunked_prefill_by_default():
    kwargs = L1.engine_kwargs("m", "int8_per_token_head", cuda_graph=False)
    assert kwargs["chunked_prefill_size"] == -1 and kwargs["disable_cuda_graph"] is True
    assert L1.engine_kwargs("m", "auto", cuda_graph=True, chunked_prefill_size=2048)["chunked_prefill_size"] == 2048
