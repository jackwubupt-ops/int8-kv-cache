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
    assert out == {"top1_agreement": 1.0, "mean_abs_logprob_diff": 0.0, "n_tokens": 3}


def test_compare_counts_mismatch():
    ref = [[(-0.1, 5), (-2.0, 9), (-0.5, 3), (-1.0, 4)]]
    other = [[(-0.2, 5), (-2.0, 8), (-0.5, 3), (-1.0, 4)]]
    out = L1.compare_token_logprobs(ref, other)
    assert out["top1_agreement"] == 0.75 and out["n_tokens"] == 4
    assert out["mean_abs_logprob_diff"] == pytest.approx(0.1 / 4)


def test_compare_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        L1.compare_token_logprobs([[(-0.1, 5)]], [[(-0.1, 5), (-0.2, 6)]])


def test_parse_meta_info_pairs_chosen_logprob_with_top1():
    meta = {
        "input_token_logprobs": [(-0.5, 11, None), (-1.5, 12, None)],
        "input_top_logprobs": [[(-0.4, 99, None)], [(-1.5, 12, None)]],
    }
    assert L1.token_logprobs_from_meta(meta) == [(-0.5, 99), (-1.5, 12)]


def test_verdict_marks_pass_and_fail():
    assert L1.verdict({"top1_agreement": 0.995}, min_top1=0.99) == "PASS"
    assert L1.verdict({"top1_agreement": 0.98}, min_top1=0.99) == "FAIL"
