"""Web app routes and job bookkeeping. No model needed.

The model is never loaded here: every test exercises the HTTP surface and the
job state machine, which is where the server's own bugs live. Actually building
a document is the pipeline's job and is covered by its own tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="web extra not installed")

from fastapi.testclient import TestClient  # noqa: E402

from januscribe.server import BuildRequest, BuildService, Job, create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(out_root=tmp_path))


def _request(**overrides) -> BuildRequest:
    base = dict(topic="a fox naturalist recording the weather", subjects=["fox"], sections=3)
    base.update(overrides)
    return BuildRequest(**base)


def test_index_serves_the_ui(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "JanusScribe" in response.text
    assert "text/html" in response.headers["content-type"]


def test_status_reports_before_the_model_is_loaded(client) -> None:
    """The server must answer before the 30 s model load, not block on it."""
    body = client.get("/api/status").json()
    assert body["model_loaded"] is False
    assert body["busy"] is False
    assert body["model_id"]


def test_unknown_job_is_404_not_a_crash(client) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/document").status_code == 404


def test_empty_subject_list_is_rejected(client) -> None:
    response = client.post("/api/build", json={"topic": "a topic here", "subjects": []})
    assert response.status_code == 422


def test_section_count_is_bounded(client) -> None:
    """An unbounded section count would queue hours of CPU from one click."""
    too_many = client.post(
        "/api/build",
        json={"topic": "a topic here", "subjects": ["fox"], "sections": 500},
    )
    assert too_many.status_code == 422


def test_request_maps_onto_a_pipeline_config() -> None:
    config = _request(sections=4, max_attempts=3, strategy="tier2").to_config("doc")
    assert config["subjects"] == ["fox"]
    assert config["sections"] == 4
    assert config["strategy"] == "tier2"
    assert config["retry"] == {"max_attempts": 3}
    assert config["stem"] == "doc"


def test_progress_counts_what_the_pipeline_actually_wrote(tmp_path) -> None:
    """Progress is derived from artefacts on disk, so it cannot drift from reality."""
    job = Job(id="abc", request=_request(sections=4), out_dir=tmp_path / "abc")
    assert job.completed_sections() == 0
    assert job.as_dict()["percent"] == 0

    outcomes = job.out_dir / "work" / "outcomes"
    outcomes.mkdir(parents=True)
    for index in range(3):
        (outcomes / f"section_{index:02d}.json").write_text("{}", encoding="utf-8")

    assert job.completed_sections() == 3
    assert job.as_dict()["percent"] == 75
    assert job.as_dict()["document_ready"] is False


def test_job_reports_ready_only_when_the_document_exists(tmp_path) -> None:
    job = Job(id="abc", request=_request(sections=1), out_dir=tmp_path / "abc")
    job.status = "done"
    assert job.as_dict()["document_ready"] is False, "done but no file is not ready"

    job.out_dir.mkdir(parents=True, exist_ok=True)
    job.document_path().write_text("<html></html>", encoding="utf-8")
    assert job.as_dict()["document_ready"] is True


def test_service_serialises_builds(tmp_path) -> None:
    """One model, one build at a time: the lock is the whole safety story."""
    service = BuildService.__new__(BuildService)
    BuildService.__init__(service, settings=__import__(
        "januscribe.config", fromlist=["Settings"]
    ).Settings(), out_root=tmp_path)

    assert service.busy is False
    with service._lock:
        assert service.busy is True
    assert service.busy is False
