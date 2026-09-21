"""Config, device and dtype resolution. No model needed."""

from __future__ import annotations

import pytest
import torch

from januscribe.config import GenerationConfig, Settings, resolve_device, resolve_dtype


def test_default_geometry_is_576_tokens() -> None:
    cfg = GenerationConfig()
    cfg.validate_geometry()
    assert cfg.grid == 24
    assert cfg.image_token_num_per_image == 24 * 24 == 576


def test_geometry_mismatch_is_rejected() -> None:
    cfg = GenerationConfig(img_size=384, patch_size=16, image_token_num_per_image=512)
    with pytest.raises(ValueError, match="does not match"):
        cfg.validate_geometry()


def test_model_id_is_configurable_not_hardcoded(tmp_path) -> None:
    yaml_file = tmp_path / "s.yaml"
    yaml_file.write_text("model_id: deepseek-ai/Janus-Pro-7B\ndevice: cpu\n", encoding="utf-8")
    assert Settings.from_yaml(yaml_file).model_id == "deepseek-ai/Janus-Pro-7B"
    assert Settings.from_yaml(yaml_file, model_id="other/model").model_id == "other/model"


def test_dtype_auto_is_fp32_on_cpu() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_dtype("auto", torch.device("cuda")) is torch.bfloat16
    assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16


def test_cpu_device_always_resolves() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present")
def test_requesting_absent_cuda_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda")


def test_repl_lock_keeps_settings_across_commands() -> None:
    """Inside the repl, re-entering the callback must not reset Settings.

    If it did, every typed command would revert to the default model id and
    reload the model -- the exact thing the repl exists to avoid.
    """
    from januscribe import cli

    cli._STATE.clear()
    try:
        cli.main(config=None, model="deepseek-ai/Janus-Pro-7B", device="cpu", dtype=None)
        assert cli._settings().model_id == "deepseek-ai/Janus-Pro-7B"

        cli._STATE["locked"] = True
        cli.main(config=None, model=None, device=None, dtype=None)  # as repl re-dispatches
        assert cli._settings().model_id == "deepseek-ai/Janus-Pro-7B"
    finally:
        cli._STATE.clear()


def test_cli_validates_device_against_config_types() -> None:
    """The CLI's accepted values must come from config.py, not a parallel list."""
    import typer

    from januscribe import cli

    assert set(cli._DEVICES) == {"auto", "cuda", "mps", "cpu"}
    assert "bfloat16" in cli._DTYPES
    cli._STATE.clear()
    try:
        with pytest.raises(typer.BadParameter, match="unknown device"):
            cli.main(config=None, model=None, device="tpu", dtype=None)
    finally:
        cli._STATE.clear()
