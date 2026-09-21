"""Render a generated document to self-contained HTML, and to PDF.

The HTML has every image inlined as a base64 data URI, so one file is the whole
deliverable -- no asset directory to lose. Debug mode renders each image's
consistency scores beside it, including the low-confidence flag from the retry
loop, so a weak page is visible in the artefact itself rather than only in a log.

PDF goes through reportlab. If reportlab is missing the HTML is still written
and the PDF is skipped with a warning, because losing the whole build over an
optional output format would be the wrong trade.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from januscribe.logging import get_logger
from januscribe.planner import DocumentPlan, Section
from januscribe.retry import RetryOutcome

log = get_logger(__name__)


@dataclass
class RenderedSection:
    """A planned section plus whatever was actually generated for it."""

    section: Section
    image: Image.Image | None = None
    outcome: RetryOutcome | None = None

    @property
    def low_confidence(self) -> bool:
        return bool(self.outcome and self.outcome.low_confidence)


@dataclass
class Document:
    """A plan with its images, ready to render."""

    plan: DocumentPlan
    sections: list[RenderedSection] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_low_confidence(self) -> int:
        return sum(1 for s in self.sections if s.low_confidence)


def image_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5c5c5c; --rule: #e2e2e2;
  --warn-bg: #fff4e5; --warn-fg: #8a4b00; --warn-rule: #ffb866;
  --ok: #1d7a3e; --bad: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14161a; --fg: #ececec; --muted: #a0a4ab; --rule: #2c3038;
    --warn-bg: #3a2a12; --warn-fg: #ffcc80; --warn-rule: #8a5a1a;
    --ok: #6bd08c; --bad: #ff8a80;
  }
}
:root[data-theme="dark"] {
  --bg: #14161a; --fg: #ececec; --muted: #a0a4ab; --rule: #2c3038;
  --warn-bg: #3a2a12; --warn-fg: #ffcc80; --warn-rule: #8a5a1a;
  --ok: #6bd08c; --bad: #ff8a80;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font: 16px/1.65 ui-serif, Georgia, "Times New Roman", serif;
}
.wrap { max-width: 780px; margin: 0 auto; padding: 48px 16px 96px; }
header { border-bottom: 2px solid var(--rule); padding-bottom: 20px; margin-bottom: 40px; }
h1 { font-size: 2rem; margin: 0 0 8px; line-height: 1.2; }
.sub { color: var(--muted); font-size: 0.9rem;
       font-family: ui-sans-serif, system-ui, sans-serif; }
section { margin: 0 0 56px; }
h2 { font-size: 1.25rem; margin: 0 0 12px; }
figure { margin: 0 0 16px; }
figure img { width: 100%; height: auto; display: block; border-radius: 8px; }
p { margin: 0 0 12px; }
.scores {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem; color: var(--muted);
  border: 1px solid var(--rule); border-radius: 8px;
  padding: 10px 12px; margin-top: 12px; overflow-x: auto;
}
.scores table { border-collapse: collapse; width: 100%; }
.scores td { padding: 2px 10px 2px 0; vertical-align: top; }
.scores .yes { color: var(--ok); }
.scores .no  { color: var(--bad); }
.flag {
  background: var(--warn-bg); color: var(--warn-fg);
  border: 1px solid var(--warn-rule); border-left-width: 4px;
  border-radius: 6px; padding: 8px 12px; margin-bottom: 12px;
  font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.85rem;
}
footer { border-top: 1px solid var(--rule); padding-top: 16px; color: var(--muted);
         font-size: 0.8rem; font-family: ui-sans-serif, system-ui, sans-serif; }
@media (max-width: 480px) { .wrap { padding: 28px 16px 64px; } h1 { font-size: 1.6rem; } }
"""


def _scores_block(rendered: RenderedSection) -> str:
    outcome = rendered.outcome
    if outcome is None:
        return ""
    report = outcome.report
    rows = [
        f"<tr><td>rubric</td><td>{report.rubric.n_yes}/{report.rubric.n_total}"
        f" = {report.rubric.score:.3f}</td></tr>",
        f"<tr><td>embedding</td><td>mean {report.embedding.mean:.4f}"
        f" / worst {report.embedding.worst:.4f}</td></tr>",
        f"<tr><td>attempts</td><td>{outcome.n_attempts}</td></tr>",
        f"<tr><td>seed</td><td>{outcome.generated.seed}</td></tr>",
    ]
    for verdict in report.rubric.verdicts:
        css = "yes" if verdict.verdict == "yes" else "no"
        rows.append(
            f"<tr><td>&nbsp;</td><td class='{css}'>{verdict.verdict:>7} "
            f"{html.escape(verdict.attribute)}</td></tr>"
        )
    return f"<div class='scores'><table>{''.join(rows)}</table></div>"


def render_html(document: Document, debug: bool = False, title: str | None = None) -> str:
    """Build the self-contained HTML document."""
    plan = document.plan
    doc_title = title or plan.title
    parts: list[str] = []

    for rendered in document.sections:
        section = rendered.section
        body = "".join(
            f"<p>{html.escape(p.strip())}</p>" for p in section.body.split("\n") if p.strip()
        )
        flag = ""
        if rendered.low_confidence:
            outcome = rendered.outcome
            flag = (
                "<div class='flag'><strong>Low confidence.</strong> "
                f"{outcome.n_attempts} attempts, none met the threshold; the best-scoring "
                f"image is shown (rubric {outcome.report.rubric.score:.2f}).</div>"
            )
        figure = ""
        if rendered.image is not None:
            alt = html.escape(section.image.scene_text() if section.image else section.heading)
            figure = (
                f"<figure><img alt='{alt}' src='{image_to_data_uri(rendered.image)}'></figure>"
            )
        scores = _scores_block(rendered) if debug else ""
        parts.append(
            f"<section><h2>{html.escape(section.heading)}</h2>"
            f"{flag}{figure}{body}{scores}</section>"
        )

    meta_bits = [
        f"{len(document.sections)} sections",
        f"{sum(1 for s in document.sections if s.image is not None)} images",
        f"planner: {html.escape(plan.planner)}",
    ]
    if document.n_low_confidence:
        meta_bits.append(f"<strong>{document.n_low_confidence} low-confidence</strong>")
    model = document.meta.get("model", {})
    if model.get("model_id"):
        meta_bits.append(html.escape(str(model["model_id"])))

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(doc_title)}</title><style>{_CSS}</style></head><body>"
        f"<div class='wrap'><header><h1>{html.escape(doc_title)}</h1>"
        f"<div class='sub'>{' &middot; '.join(meta_bits)}</div></header>"
        f"{''.join(parts)}"
        f"<footer>Generated by JanusScribe. Subjects: "
        f"{html.escape(', '.join(plan.subject_ids))}."
        f"{' Debug mode: consistency scores shown per image.' if debug else ''}</footer>"
        "</div></body></html>"
    )


def write_html(document: Document, path: str | Path, debug: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(document, debug=debug), encoding="utf-8")
    log.info("html_written", path=str(out), sections=len(document.sections), debug=debug)
    return out


def write_pdf(document: Document, path: str | Path, debug: bool = False) -> Path | None:
    """Render to PDF. Returns None (with a warning) if reportlab is unavailable."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image as RLImage,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError:
        log.warning("pdf_skipped", reason="reportlab not installed", path=str(path))
        return None

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    mono = ParagraphStyle(
        "mono", parent=styles["BodyText"], fontName="Courier", fontSize=7.5, leading=10
    )
    warn = ParagraphStyle(
        "warn", parent=styles["BodyText"], fontSize=9, textColor="#8a4b00", leading=12
    )

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=document.plan.title,
    )
    flow: list = [Paragraph(html.escape(document.plan.title), styles["Title"]), Spacer(1, 8)]

    for index, rendered in enumerate(document.sections):
        flow.append(Paragraph(html.escape(rendered.section.heading), styles["Heading2"]))
        if rendered.low_confidence and rendered.outcome:
            flow.append(
                Paragraph(
                    f"Low confidence: {rendered.outcome.n_attempts} attempts, none met the "
                    f"threshold (best rubric {rendered.outcome.report.rubric.score:.2f}).",
                    warn,
                )
            )
        if rendered.image is not None:
            buffer = io.BytesIO()
            rendered.image.save(buffer, format="PNG")
            buffer.seek(0)
            # reportlab's Image flowable takes a path or a file-like object, not
            # an ImageReader. Width is capped to the frame; height follows the
            # image's own aspect ratio so non-square art is not distorted.
            width = min(doc.width, 110 * mm)
            src_w, src_h = rendered.image.size
            height = width * (src_h / src_w) if src_w else width
            flow.append(RLImage(buffer, width=width, height=height))
            flow.append(Spacer(1, 6))
        flow.append(Paragraph(html.escape(rendered.section.body), styles["BodyText"]))
        if debug and rendered.outcome:
            report = rendered.outcome.report
            lines = [
                f"rubric {report.rubric.n_yes}/{report.rubric.n_total} = "
                f"{report.rubric.score:.3f} | embedding {report.embedding.mean:.4f} | "
                f"attempts {rendered.outcome.n_attempts} | seed {rendered.outcome.generated.seed}"
            ]
            lines += [f"{v.verdict:>7}  {v.attribute}" for v in report.rubric.verdicts]
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("<br/>".join(html.escape(x) for x in lines), mono))
        if index < len(document.sections) - 1:
            flow.append(PageBreak())

    doc.build(flow)
    log.info("pdf_written", path=str(out), sections=len(document.sections))
    return out


def write_all(
    document: Document, out_dir: str | Path, stem: str = "document", debug: bool = False
) -> dict[str, Path | None]:
    """Write HTML and PDF, plus the plan and a machine-readable score sidecar."""
    import json

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path | None] = {
        "html": write_html(document, directory / f"{stem}.html", debug=debug),
        "pdf": write_pdf(document, directory / f"{stem}.pdf", debug=debug),
        "plan": document.plan.save(directory / f"{stem}.plan.json"),
    }

    scores = directory / f"{stem}.scores.json"
    scores.write_text(
        json.dumps(
            {
                "title": document.plan.title,
                "planner": document.plan.planner,
                "n_sections": len(document.sections),
                "n_low_confidence": document.n_low_confidence,
                "meta": document.meta,
                "sections": [
                    {
                        "heading": s.section.heading,
                        "outcome": s.outcome.as_dict() if s.outcome else None,
                    }
                    for s in document.sections
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["scores"] = scores
    return paths
