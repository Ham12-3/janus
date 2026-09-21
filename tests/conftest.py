"""Shared fixtures.

The model is never mocked. Tests that need it are marked ``slow`` and skip with
an explicit reason when the weights are not on disk, so a green run on a laptop
with no weights cannot be mistaken for a green run that actually exercised the
model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOGE = REPO_ROOT / "assets" / "doge.png"


def _weights_present(model_id: str) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(model_id, local_files_only=True, allow_patterns=["config.json"])
        return True
    except (LocalEntryNotFoundError, OSError, ValueError):
        return False


@pytest.fixture(scope="session")
def settings():
    from januscribe.config import Settings

    return Settings()


@pytest.fixture(scope="session")
def bundle(settings):
    """The real model, loaded once for the whole session."""
    if not _weights_present(settings.model_id):
        pytest.skip(
            f"weights for {settings.model_id} are not in the local HF cache; "
            "run `januscribe info` once to download them"
        )
    from januscribe.model import get_bundle

    return get_bundle(settings)


@pytest.fixture(scope="session")
def doge_path() -> Path:
    if not DOGE.exists():
        pytest.skip(f"test asset missing: {DOGE}")
    return DOGE
