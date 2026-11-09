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


def _meta(steps):
    """steps: list of (chosen_id, [(lp, id), ...] top-2)."""
    return {"output_token_logprobs": [(top[0][0] if top[0][1] == tok else top[1][0], tok, None) for tok, top in steps],
            "output_top_logprobs": [[(lp, i, None) for lp, i in top] for _, top in steps]}


def test_trace_from_meta_keeps_tokens_logprobs_and_top2_margin():
    t = L1.trace_from_meta(_meta([(5, [(-0.1, 5), (-2.6, 7)]), (9, [(-0.5, 9), (-0.55, 4)])]))
    assert t["tokens"] == [5, 9]
    assert t["lps"] == pytest.approx([-0.1, -0.5])
    assert t["margins"] == pytest.approx([2.5, 0.05])


def _trace(tokens, lps, margins):
    return {"tokens": tokens, "lps": lps, "margins": margins}


def test_decode_path_compare_measures_the_shared_prefix_and_divergence_margin():
    a = [_trace([1, 2, 3], [-0.1, -0.2, -0.3], [1.0, 0.02, 1.0]), _trace([4, 5], [-0.1, -0.1], [1.0, 1.0])]
    b = [_trace([1, 8, 3], [-0.12, -0.25, -0.3], [1.0, 0.01, 1.0]), _trace([4, 5], [-0.1, -0.1], [1.0, 1.0])]
    r = L1.decode_path_compare(a, b)
    assert r["n_diverged"] == 1 and r["n"] == 2
    assert r["prefix_mean_abs_dlogp"] == pytest.approx(0.02 / 3)  # token 1 of prompt 0, both tokens of prompt 1
    assert r["divergence_margins"] == pytest.approx([0.02])  # the larger margin at the flip: still a near-tie


def test_graph_verdict_accepts_near_tie_flips_with_bf16_sized_drift():
    int8 = {"n_diverged": 3, "prefix_mean_abs_dlogp": 0.004, "divergence_margins": [0.01, 0.05, 0.08]}
    bf16 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.003, "divergence_margins": []}
    assert L1.graph_verdict(int8, bf16)["pass"]


def test_graph_verdict_rejects_a_flip_that_is_not_a_tie():
    int8 = {"n_diverged": 1, "prefix_mean_abs_dlogp": 0.004, "divergence_margins": [0.9]}
    bf16 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.003, "divergence_margins": []}
    assert not L1.graph_verdict(int8, bf16)["pass"]


def test_graph_verdict_rejects_drift_well_beyond_bf16():
    int8 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.05, "divergence_margins": []}
    bf16 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.003, "divergence_margins": []}
    assert not L1.graph_verdict(int8, bf16)["pass"]


def test_noise_floor_only_counts_graph_on_bf16():
    results = {"auto:plain:on": 1.0, "auto:plain:off": 0.97, "int8_per_token_head:plain:on": 0.98}
    assert L1.noise_floor(results) == 1.0


def test_graph_verdict_treats_one_bf16_logit_step_as_a_tie():
    # Server run 20261109_202409: every flip had margin exactly 0.125 = one bf16 step for logits in [16, 32).
    int8 = {"n_diverged": 3, "prefix_mean_abs_dlogp": 0.0033, "divergence_margins": [0.125, 0.125, 0.125]}
    bf16 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.0032, "divergence_margins": []}
    assert L1.graph_verdict(int8, bf16)["pass"]


def test_graph_verdict_rejects_a_two_bf16_step_gap():
    int8 = {"n_diverged": 1, "prefix_mean_abs_dlogp": 0.0033, "divergence_margins": [0.25]}
    bf16 = {"n_diverged": 0, "prefix_mean_abs_dlogp": 0.0032, "divergence_margins": []}
    assert not L1.graph_verdict(int8, bf16)["pass"]


def test_impl_gap_passes_when_fused_tracks_plain():
    top1 = {"int8_per_token_head:plain:on": 0.9888, "int8_per_token_head:fused:on": 0.9861,
            "int8_per_token_head:plain:off": 0.9895, "int8_per_token_head:fused:off": 0.9895}
    gaps = L1.impl_gaps(top1, tol=0.005)
    assert gaps == {"on": {"gap": pytest.approx(0.0027), "pass": True},
                    "off": {"gap": pytest.approx(0.0), "pass": True}}


def test_impl_gap_fails_when_fused_drifts_from_plain():
    top1 = {"int8_per_token_head:plain:on": 0.9888, "int8_per_token_head:fused:on": 0.9800}
    assert not L1.impl_gaps(top1, tol=0.005)["on"]["pass"]


def test_impl_gap_skips_graph_modes_without_both_impls():
    assert L1.impl_gaps({"int8_per_token_head:plain:on": 0.99}, tol=0.005) == {}
