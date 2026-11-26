"""Run the stage 5 scripts end to end on a tiny random model (calibrate -> operator A/B -> eval)."""
import json

from tests.test_experiments_smoke import COMMON, load_script, tiny_model  # noqa: F401


def test_calibrate_operator_and_eval_chain(tiny_model, tmp_path):  # noqa: F811
    calib = tmp_path / "calib.jsonl"
    calib.write_text("\n".join(json.dumps({"question": "the code the"}) for _ in range(3)))
    load_script("stage5_calibrate").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--samples", "3", "--max-new-tokens", "4",
         "--data-file", str(calib), "--quantile", "0.9", *COMMON]
    )
    summary = json.loads((tmp_path / "calibration.json").read_text())
    assert set(summary["channel_spread"]) == {"k", "v"}
    scales = tmp_path / "scales.pt"
    assert scales.exists()

    configs = "per_channel:dequant:static,per_channel:int8:static,per_head:dequant:dynamic,per_head:int8:dynamic"
    load_script("stage5_operator").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--max-new-tokens", "3", "--max-prompts", "1",
         "--needle-facts", "2", "--configs", configs, "--scales", str(scales), *COMMON]
    )
    data = json.loads((tmp_path / "operator_ab.json").read_text())
    assert set(data["a_vs_b"]) == {"per_channel:static", "per_head:dynamic"}

    he = tmp_path / "he.jsonl"
    he.write_text(json.dumps({
        "task_id": "t/0", "prompt": "def add(a, b):\n", "entry_point": "add",
        "test": "def check(f):\n    assert f(1, 2) == 3\n",
    }))
    args = ["--task", "humaneval", "--model", tiny_model, "--out", str(tmp_path), "--max-new-tokens", "3",
            "--data-file", str(he), "--configs", "per_channel:int8:static", "--scales", str(scales), *COMMON]
    out = load_script("stage5_eval").main(args)
    assert set(out["results"]) == {"baseline", "per_channel:int8:static"}
    assert "mcnemar_p" in out["results"]["per_channel:int8:static"]
    again = load_script("stage5_eval").main(args)  # resumes: nothing re-run, same results
    assert again["results"] == out["results"]
