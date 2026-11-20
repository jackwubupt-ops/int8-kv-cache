"""Pure-PyTorch pieces of the Task 8 wiring that also run on the Mac."""
import pytest

from tests_stage4.conftest import load_sglang_file

M = load_sglang_file("srt/layers/quantization/int8_kv.py")


def test_fused_reads_need_the_standard_extend_kernel():
    with pytest.raises(NotImplementedError, match="deterministic"):
        M.check_int8_backend_mode(impl="fused", enable_deterministic=True, kv_group_num=7)


def test_plain_reads_allow_deterministic_mode():
    M.check_int8_backend_mode(impl="plain", enable_deterministic=True, kv_group_num=7)


def test_fused_reads_in_normal_mode_are_fine():
    M.check_int8_backend_mode(impl="fused", enable_deterministic=False, kv_group_num=7)


def test_fused_reads_reject_mha():
    """The fused decode kernel only has the grouped (GQA/MQA) path; MHA must fail at startup, not on first decode."""
    with pytest.raises(NotImplementedError, match="GQA"):
        M.check_int8_backend_mode(impl="fused", enable_deterministic=False, kv_group_num=1)


def test_plain_reads_allow_mha():
    M.check_int8_backend_mode(impl="plain", enable_deterministic=False, kv_group_num=1)


def test_fused_reads_allow_gqa():
    M.check_int8_backend_mode(impl="fused", enable_deterministic=False, kv_group_num=7)
