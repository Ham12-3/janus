"""A small local web app over the document pipeline.

This is the "persistent mode" the brief asks for: the model is loaded once when
the server starts and held for the process's life, so a build no longer pays the
load cost per command.

Scope, stated plainly:

* **Local only.** It binds to 127.0.0.1 by default. There is no authentication,
  no multi-user isolation, and it must not be exposed to a network.
* **One build at a time.** A single worker thread holds a lock around the model.
  Janus generation is not safe to run concurrently against one set of weights,
  and on a CPU box two builds would each be twice as slow anyway, so requests
  queue rather than overlap.
* **The UI does not make anything faster.** A page is still ~6 minutes on CPU.
  What the UI buys is not waiting blind: progress is reported per section as
  each image lands.

Progress is derived by counting the artefacts the pipeline writes as it goes,
rather than by threading a callback through it. That keeps the pipeline unaware
of the server, and it stays accurate even if a job is resumed after a restart.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from januscribe.config import Settings
from januscribe.logging import get_logger

log = get_logger(__name__)

JobStatus = Literal["queued", "running", "done", "error"]

DEFAULT_OUT = Path("dist") / "web"


class BuildRequest(BaseModel):
    """What the browser sends to start a build."""

    topic: str = Field(min_length=3, max_length=400)
    subjects: list[str] = Field(min_length=1)
    sections: int = Field(3, ge=1, le=12)
    title: str | None = None
    planner: Literal["template", "janus"] = "template"
    strategy: Literal["tier0", "tier2"] = "tier0"
    max_attempts: int = Field(1, ge=1, le=4)
    debug: bool = True

    def to_config(self, stem: str) -> dict:
        return {
            "topic": self.topic,
            "subjects": self.subjects,
            "sections": self.sections,
            "title": self.title,
            "planner": self.planner,
            "strategy": self.strategy,
            "stem": stem,
            "debug": self.debug,
            "retry": {"max_attempts": self.max_attempts},
        }


@dataclass
class Job:
    """One build, its state, and where its artefacts live."""

    id: str
    request: BuildRequest
    out_dir: Path
    status: JobStatus = "queued"
    error: str | None = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started: str | None = None
    finished: str | None = None
    n_low_confidence: int = 0
    plan_summary: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return "document"

    def completed_sections(self) -> int:
        """Count what the pipeline has actually written, so progress is real."""
        outcomes = self.out_dir / "work" / "outcomes"
        if not outcomes.is_dir():
            return 0
        return len(list(outcomes.glob("section_*.json")))

    def document_path(self) -> Path:
        return self.out_dir / f"{self.stem}.html"

    def as_dict(self) -> dict[str, Any]:
        done = self.completed_sections()
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "topic": self.request.topic,
            "subjects": self.request.subjects,
            "sections_total": self.request.sections,
            "sections_done": done,
            "percent": round(100 * done / max(self.request.sections, 1)),
            "n_low_confidence": self.n_low_confidence,
            "plan": self.plan_summary,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "document_ready": self.status == "done" and self.document_path().exists(),
        }


class BuildService:
    """Holds the model and runs one build at a time."""

    def __init__(self, settings: Settings, out_root: Path = DEFAULT_OUT) -> None:
        self.settings = settings
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()   # guards the model: one build at a time
        self._jobs_lock = threading.Lock()
        self._bundle = None
        self._registry = None
        self._scenes: list[str] = []

    # -- lazy, one-time model load ----------------------------------------- #

    def warm(self) -> dict:
        """Load the model and registries. Safe to call repeatedly."""
        from januscribe.model import get_bundle
        from januscribe.subjects import SubjectRegistry, load_scenes

        if self._bundle is None:
            log.info("server_loading_model", model_id=self.settings.model_id)
            self._bundle = get_bundle(self.settings)
            self._registry = SubjectRegistry.from_yaml("configs/subjects.yaml")
            self._scenes = load_scenes("configs/scenes.yaml")
            log.info("server_ready", subjects=[s.id for s in self._registry])
        return self._bundle.describe()

    @property
    def registry(self):
        self.warm()
        return self._registry

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # -- jobs --------------------------------------------------------------- #

    def preview(self, request: BuildRequest) -> list[str]:
        """Plan without generating. Costs nothing: the template planner has no model."""
        from januscribe.planner import TemplatePlanner, plan_summary, resolve_subjects

        subjects = resolve_subjects(self.registry, request.subjects)
        plan = TemplatePlanner(self._scenes).plan(
            request.topic, subjects, request.sections, title=request.title
        )
        return plan_summary(plan).split("\n")

    def submit(self, request: BuildRequest) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, request=request, out_dir=self.out_root / job_id)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            job.plan_summary = self.preview(request)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with self._jobs_lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        log.info("job_submitted", job=job_id, topic=request.topic, sections=request.sections)
        return job

    def _run(self, job: Job) -> None:
        # Blocks until any in-flight build finishes: the model cannot be shared.
        with self._lock:
            job.status = "running"
            job.started = datetime.now(timezone.utc).isoformat()
            try:
                from januscribe.pipeline import build_from_config

                self.warm()
                document, _ = build_from_config(
                    self._bundle,
                    job.request.to_config(job.stem),
                    self._registry,
                    self._scenes,
                    job.out_dir,
                    self.settings,
                )
                job.n_low_confidence = document.n_low_confidence
                job.status = "done"
                log.info(
                    "job_done", job=job.id, low_confidence=job.n_low_confidence,
                    sections=len(document.sections),
                )
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                log.error("job_failed", job=job.id, error=job.error)
                traceback.print_exc()
            finally:
                job.finished = datetime.now(timezone.utc).isoformat()

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return job

    def recent(self, limit: int = 20) -> list[Job]:
        with self._jobs_lock:
            return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]


def create_app(settings: Settings | None = None, out_root: Path = DEFAULT_OUT) -> FastAPI:
    service = BuildService(settings or Settings(), out_root=out_root)
    app = FastAPI(title="JanusScribe", version="0.1.0")
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/status")
    def status() -> dict:
        return {
            "model_loaded": service._bundle is not None,
            "busy": service.busy,
            "model_id": service.settings.model_id,
            "device": service.settings.device,
        }

    @app.post("/api/warm")
    def warm() -> dict:
        return service.warm()

    @app.get("/api/subjects")
    def subjects() -> list[dict]:
        return [
            {
                "id": s.id,
                "noun": s.noun,
                "description": s.canonical_description,
                "attributes": s.attributes,
                "n_references": len(s.reference_paths(service.registry.root)),
            }
            for s in service.registry
        ]

    @app.post("/api/preview")
    def preview(request: BuildRequest) -> dict:
        return {"plan": service.preview(request)}

    @app.post("/api/build")
    def build(request: BuildRequest) -> dict:
        return service.submit(request).as_dict()

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [j.as_dict() for j in service.recent()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return service.get(job_id).as_dict()

    @app.get("/api/jobs/{job_id}/document", response_class=HTMLResponse)
    def document(job_id: str) -> str:
        job = service.get(job_id)
        path = job.document_path()
        if not path.exists():
            raise HTTPException(status_code=404, detail="document not ready")
        return path.read_text(encoding="utf-8")

    @app.get("/api/jobs/{job_id}/scores")
    def scores(job_id: str) -> JSONResponse:
        import json

        job = service.get(job_id)
        path = job.out_dir / f"{job.stem}.scores.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="scores not ready")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JanusScribe</title>
<style>
:root { --bg:#fff; --fg:#161616; --muted:#5f5f5f; --rule:#e3e3e3; --accent:#1f6feb;
        --warn-bg:#fff4e5; --warn-fg:#8a4b00; --ok:#1d7a3e; --bad:#b3261e; --card:#fafafa; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#13151a; --fg:#ececec; --muted:#a2a6ad; --rule:#2b2f37; --accent:#6ea8ff;
  --warn-bg:#3a2a12; --warn-fg:#ffcc80; --ok:#6bd08c; --bad:#ff8a80; --card:#191c22; } }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:var(--muted);font-size:.85rem;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;
      padding:18px;margin-bottom:18px}
label{display:block;font-size:.8rem;color:var(--muted);margin:12px 0 4px;
      text-transform:uppercase;letter-spacing:.04em}
input,select,textarea{width:100%;padding:9px 11px;border:1px solid var(--rule);
      border-radius:7px;background:var(--bg);color:var(--fg);font:inherit}
textarea{min-height:64px;resize:vertical}
.row{display:flex;gap:12px;flex-wrap:wrap}
.row>div{flex:1;min-width:130px}
button{margin-top:16px;padding:10px 18px;border:0;border-radius:7px;background:var(--accent);
       color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
button:disabled{opacity:.5;cursor:not-allowed}
pre{background:var(--bg);border:1px solid var(--rule);border-radius:7px;padding:12px;
    overflow-x:auto;font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap}
.bar{height:8px;background:var(--rule);border-radius:99px;overflow:hidden;margin:10px 0}
.bar>div{height:100%;background:var(--accent);width:0;transition:width .4s}
.jobs a{color:var(--accent)}
.badge{display:inline-block;font-size:.72rem;padding:2px 8px;border-radius:99px;
       border:1px solid var(--rule);color:var(--muted)}
.badge.running{border-color:var(--accent);color:var(--accent)}
.badge.done{border-color:var(--ok);color:var(--ok)}
.badge.error{border-color:var(--bad);color:var(--bad)}
.warn{background:var(--warn-bg);color:var(--warn-fg);border-radius:7px;padding:10px 12px;
      font-size:.85rem;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:.85rem}
td{padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
@media(max-width:520px){.row>div{min-width:100%}}
</style></head><body><div class="wrap">
<h1>JanusScribe</h1>
<div class="sub" id="status">checking server...</div>

<div class="card">
  <label>Topic</label>
  <textarea id="topic">a fox naturalist recording the weather across one year</textarea>
  <div class="row">
    <div><label>Subject</label><select id="subject"></select></div>
    <div><label>Sections</label><input id="sections" type="number" value="3" min="1" max="12"></div>
    <div><label>Attempts per image</label>
      <input id="attempts" type="number" value="1" min="1" max="4"></div>
  </div>
  <div class="row">
    <div><label>Planner</label><select id="planner">
      <option value="template">template (instant, deterministic)</option>
      <option value="janus">janus (slower, weaker prose)</option></select></div>
    <div><label>Strategy</label><select id="strategy">
      <option value="tier0">tier0 (canonical description)</option>
      <option value="tier2">tier2 (learned soft token)</option></select></div>
  </div>
  <button class="ghost" id="previewBtn">Preview plan</button>
  <button id="buildBtn">Generate document</button>
  <div class="warn" id="timeNote"></div>
  <pre id="plan" style="display:none"></pre>
</div>

<div class="card" id="jobCard" style="display:none">
  <div><strong id="jobTitle">Build</strong> <span class="badge" id="jobBadge">queued</span></div>
  <div class="bar"><div id="jobBar"></div></div>
  <div class="sub" id="jobDetail"></div>
  <div id="jobLinks"></div>
</div>

<div class="card jobs">
  <strong>Recent builds</strong>
  <table id="jobs"></table>
</div>
</div>
<script>
const $ = id => document.getElementById(id);
let poll = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function payload() {
  return {
    topic: $('topic').value,
    subjects: [$('subject').value],
    sections: +$('sections').value,
    planner: $('planner').value,
    strategy: $('strategy').value,
    max_attempts: +$('attempts').value,
    debug: true
  };
}
function estimate() {
  const mins = Math.round(+$('sections').value * +$('attempts').value * 6);
  $('timeNote').textContent =
    `Roughly ${mins} minutes on this machine (~6 min per image per attempt). ` +
    `Generation is CPU-bound; the page will update as each section lands.`;
}
['sections','attempts'].forEach(id => $(id).addEventListener('input', estimate));

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    $('status').textContent =
      `${s.model_id} on ${s.device} — ` +
      (s.model_loaded ? (s.busy ? 'building' : 'model loaded, idle') : 'model not loaded yet');
  } catch (e) { $('status').textContent = 'server unreachable'; }
}
async function loadSubjects() {
  const subs = await api('/api/subjects');
  $('subject').innerHTML = subs.map(s =>
    `<option value="${s.id}">${s.id} — ${s.n_references} reference images</option>`).join('');
}
async function refreshJobs() {
  const jobs = await api('/api/jobs');
  $('jobs').innerHTML = jobs.length ? jobs.map(j => `<tr>
    <td><span class="badge ${j.status}">${j.status}</span></td>
    <td>${j.topic.slice(0, 48)}</td>
    <td>${j.sections_done}/${j.sections_total}</td>
    <td>${j.document_ready ? `<a href="/api/jobs/${j.id}/document" target="_blank">open</a>` : ''}</td>
  </tr>`).join('') : '<tr><td class="sub">none yet</td></tr>';
}
function renderJob(j) {
  $('jobCard').style.display = 'block';
  $('jobTitle').textContent = j.topic.slice(0, 60);
  $('jobBadge').textContent = j.status;
  $('jobBadge').className = 'badge ' + j.status;
  $('jobBar').style.width = j.percent + '%';
  $('jobDetail').textContent = j.status === 'error'
    ? j.error
    : `${j.sections_done} of ${j.sections_total} sections`
      + (j.n_low_confidence ? ` — ${j.n_low_confidence} low confidence` : '');
  $('jobLinks').innerHTML = j.document_ready
    ? `<button class="ghost" onclick="window.open('/api/jobs/${j.id}/document','_blank')">Open document</button>`
    : '';
  if (j.status === 'done' || j.status === 'error') {
    clearInterval(poll); poll = null; $('buildBtn').disabled = false; refreshJobs(); refreshStatus();
  }
}
$('previewBtn').onclick = async () => {
  try {
    const r = await api('/api/preview', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload())
    });
    $('plan').style.display = 'block';
    $('plan').textContent = r.plan.join('\\n');
  } catch (e) { alert(e.message); }
};
$('buildBtn').onclick = async () => {
  $('buildBtn').disabled = true;
  try {
    const job = await api('/api/build', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload())
    });
    renderJob(job);
    poll = setInterval(async () => renderJob(await api('/api/jobs/' + job.id)), 3000);
  } catch (e) { alert(e.message); $('buildBtn').disabled = false; }
};
estimate(); refreshStatus(); loadSubjects().catch(() => {}); refreshJobs().catch(() => {});
setInterval(refreshStatus, 10000);
</script></body></html>
"""


def serve(host: str = "127.0.0.1", port: int = 8000, settings: Settings | None = None) -> None:
    """Run the server. Binds to localhost: this app has no authentication."""
    import uvicorn

    app = create_app(settings)
    log.info("server_start", url=f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
