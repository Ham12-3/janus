"""JanusScribe command line interface.

Every command shares one ``Settings`` object built from (in increasing
precedence) defaults, an optional YAML config, environment variables, and
flags. The model is loaded through ``model.get_bundle``, so ``repl`` can run
many commands against a single load -- which is the point, given how long a 7B
load takes.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated, Optional, get_args

import typer

from januscribe.config import DeviceName, DTypeName, Settings

# config.py owns the valid values; read them back rather than restating them here.
_DEVICES: tuple[str, ...] = get_args(DeviceName)
_DTYPES: tuple[str, ...] = get_args(DTypeName)
from januscribe.logging import configure_logging, get_logger

app = typer.Typer(
    name="januscribe",
    help="Consistent illustrated document generation on DeepSeek Janus-Pro.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)

_STATE: dict[str, object] = {}


def _settings() -> Settings:
    settings = _STATE.get("settings")
    if settings is None:  # command invoked without the group callback (tests)
        settings = Settings()
        _STATE["settings"] = settings
    return settings  # type: ignore[return-value]


def _bundle():
    from januscribe.model import get_bundle

    return get_bundle(_settings())


@app.callback()
def main(
    config: Annotated[
        Optional[Path], typer.Option("--config", help="YAML settings file.", exists=True)
    ] = None,
    model: Annotated[
        Optional[str], typer.Option("--model", help="HF model id, e.g. deepseek-ai/Janus-Pro-7B.")
    ] = None,
    device: Annotated[
        Optional[str], typer.Option("--device", help=f"{' | '.join(_DEVICES)}.")
    ] = None,
    dtype: Annotated[
        Optional[str], typer.Option("--dtype", help=f"{' | '.join(_DTYPES)}.")
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
    json_logs: Annotated[bool, typer.Option("--json-logs/--no-json-logs")] = False,
) -> None:
    """Build the shared Settings for whichever subcommand runs next.

    Inside ``repl`` this callback runs again for every typed command. It must not
    rebuild Settings there, or each command would silently revert to defaults and
    reload the model -- defeating the entire purpose of holding it in memory.
    """
    configure_logging(level=log_level, json_logs=json_logs)
    if _STATE.get("locked"):
        return
    overrides: dict[str, object] = {"log_level": log_level}
    if model:
        overrides["model_id"] = model
    if device:
        if device not in _DEVICES:
            raise typer.BadParameter(f"unknown device {device!r}; use {'|'.join(_DEVICES)}")
        overrides["device"] = device
    if dtype:
        if dtype not in _DTYPES:
            raise typer.BadParameter(f"unknown dtype {dtype!r}; use {'|'.join(_DTYPES)}")
        overrides["dtype"] = dtype

    settings = Settings.from_yaml(config, **overrides) if config else Settings(**overrides)  # type: ignore[arg-type]
    _STATE["settings"] = settings


@app.command()
def info() -> None:
    """Load the model and print what actually got loaded."""
    bundle = _bundle()
    typer.echo(json.dumps(bundle.describe(), indent=2))


@app.command()
def gen(
    prompt: Annotated[str, typer.Argument(help="Text prompt.")],
    seed: Annotated[int, typer.Option("--seed", help="Base seed; image i uses seed+i.")] = 42,
    n: Annotated[int, typer.Option("--n", "-n", help="Images to sample.")] = 1,
    cfg_weight: Annotated[float, typer.Option("--cfg", help="Classifier-free guidance.")] = 5.0,
    temperature: Annotated[float, typer.Option("--temp")] = 1.0,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("outputs/gen"),
    stem: Annotated[str, typer.Option("--stem", help="Filename prefix.")] = "img",
    progress_every: Annotated[int, typer.Option("--progress-every")] = 64,
) -> None:
    """Text to image. `januscribe gen "a red fox" --seed 42`"""
    from januscribe.config import GenerationConfig
    from januscribe.generate import generate_images, save_all

    bundle = _bundle()
    cfg = GenerationConfig(parallel_size=n, cfg_weight=cfg_weight, temperature=temperature)
    images = generate_images(bundle, prompt, seed=seed, cfg=cfg, progress_every=progress_every)
    paths = save_all(images, out, stem=stem)

    sidecar = Path(out) / f"{stem}_seed{seed}.json"
    sidecar.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "base_seed": seed,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "images": [
                    {"path": str(p), "seed": g.seed, "tokens": g.tokens.tolist()}
                    for p, g in zip(paths, images)
                ],
                "model": bundle.describe(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for p in paths:
        typer.echo(str(p))
    typer.echo(f"metadata: {sidecar}")


@app.command()
def ask(
    image: Annotated[Path, typer.Argument(exists=True, help="Image to look at.")],
    question: Annotated[str, typer.Argument(help="Question about the image.")],
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 256,
    temperature: Annotated[float, typer.Option("--temp", help="0 = greedy.")] = 0.0,
) -> None:
    """Image plus question to text. `januscribe ask image.png "what colour is the fox?"`"""
    from januscribe.config import UnderstandConfig
    from januscribe.understand import ask as ask_model

    bundle = _bundle()
    cfg = UnderstandConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    answer = ask_model(bundle, image, question, cfg=cfg)
    typer.echo(answer.text)


@app.command("vq-roundtrip")
def vq_roundtrip(
    image: Annotated[Path, typer.Argument(exists=True)],
    out: Annotated[Path, typer.Option("--out")] = Path("outputs/vq/roundtrip.png"),
    save_tokens: Annotated[bool, typer.Option("--save-tokens/--no-save-tokens")] = True,
) -> None:
    """Encode an image to 576 VQ tokens, decode it back, save the pair side by side."""
    from januscribe.vq import roundtrip, save_side_by_side

    bundle = _bundle()
    result = roundtrip(bundle, image)
    path = save_side_by_side(result, out)
    if save_tokens:
        token_path = Path(out).with_suffix(".tokens.json")
        token_path.write_text(
            json.dumps(
                {
                    "source": str(image),
                    "n_tokens": result.n_tokens,
                    "grid": result.grid,
                    "psnr_db": result.psnr,
                    "tokens": result.tokens.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"tokens: {token_path}")
    typer.echo(f"{path}  psnr={result.psnr:.2f} dB  tokens={result.n_tokens}")


@app.command("subjects")
def subjects_cmd(
    registry: Annotated[Path, typer.Option("--registry", exists=True)] = Path(
        "configs/subjects.yaml"
    ),
) -> None:
    """List the subject registry: prompts, seeds, attributes, reference sheets."""
    from januscribe.subjects import SubjectRegistry

    reg = SubjectRegistry.from_yaml(registry)
    for subject in reg:
        refs = subject.reference_paths(reg.root)
        typer.echo(f"{subject.id}  (noun={subject.noun}, base_seed={subject.base_seed})")
        typer.echo(f"  canonical: {subject.canonical_description}")
        typer.echo(f"  refs: {len(refs)} in {subject.reference_dir(reg.root)}")
        for attribute in subject.attributes:
            typer.echo(f"    - {attribute}")


@app.command("build-refs")
def build_refs(
    subject_ids: Annotated[
        Optional[str], typer.Option("--subject", help="Comma-separated ids; default all.")
    ] = None,
    registry: Annotated[Path, typer.Option("--registry", exists=True)] = Path(
        "configs/subjects.yaml"
    ),
    n: Annotated[int, typer.Option("--n", help="Reference images per subject.")] = 4,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
) -> None:
    """Generate each subject's reference sheet from its canonical description."""
    from januscribe.baseline import build_reference_sheet
    from januscribe.subjects import SubjectRegistry

    bundle = _bundle()
    reg = SubjectRegistry.from_yaml(registry)
    chosen = reg.select(subject_ids.split(",") if subject_ids else None)
    for subject in chosen:
        paths = build_reference_sheet(
            bundle, subject, reg.root, n, cfg=_settings().generation, force=force
        )
        for path in paths:
            typer.echo(str(path))


@app.command()
def baseline(
    scenes: Annotated[
        Optional[int], typer.Option("--scenes", help="Use the first N scenes; default all.")
    ] = None,
    subject_ids: Annotated[
        Optional[str], typer.Option("--subject", help="Comma-separated ids; default all.")
    ] = None,
    registry: Annotated[Path, typer.Option("--registry", exists=True)] = Path(
        "configs/subjects.yaml"
    ),
    scene_file: Annotated[Path, typer.Option("--scene-file", exists=True)] = Path(
        "configs/scenes.yaml"
    ),
    n_refs: Annotated[int, typer.Option("--refs")] = 4,
    out: Annotated[Path, typer.Option("--out")] = Path("outputs/baseline/tier0"),
    force: Annotated[bool, typer.Option("--force/--no-force", help="Ignore cached work.")] = False,
) -> None:
    """Run the Tier 0 consistency baseline over subjects x scenes.

    Resumable: images and scores already on disk are reused unless --force.
    """
    from januscribe.baseline import Tier0Strategy, run_baseline, write_report
    from januscribe.subjects import SubjectRegistry, load_scenes

    bundle = _bundle()
    settings = _settings()
    reg = SubjectRegistry.from_yaml(registry)
    chosen = reg.select(subject_ids.split(",") if subject_ids else None)
    all_scenes = load_scenes(scene_file)
    scene_list = load_scenes(scene_file, limit=scenes)

    result = run_baseline(
        bundle, reg, chosen, scene_list, out,
        strategy=Tier0Strategy(), n_refs=n_refs,
        gen_cfg=settings.generation, und_cfg=settings.understand, force=force,
        n_scenes_available=len(all_scenes),
    )
    json_path, md_path = write_report(result, chosen, out)
    typer.echo(str(json_path))
    typer.echo(str(md_path))


@app.command()
def learn(
    subject_id: Annotated[str, typer.Option("--subject", help="Subject id from the registry.")],
    images: Annotated[
        Optional[str],
        typer.Option("--images", help="Glob of reference images; defaults to the subject sheet."),
    ] = None,
    registry: Annotated[Path, typer.Option("--registry", exists=True)] = Path(
        "configs/subjects.yaml"
    ),
    steps: Annotated[int, typer.Option("--steps")] = 500,
    lr: Annotated[float, typer.Option("--lr")] = 1e-3,
    batch_size: Annotated[int, typer.Option("--batch", help="Effective batch via grad accum.")] = 4,
    out: Annotated[Path, typer.Option("--out")] = Path("tokens"),
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Learn a soft token for a subject. `januscribe learn --subject fox`

    With no --images, uses the subject's reference sheet. The learned token is
    written as a small .safetensors file and is usable in any prompt afterwards.
    """
    import glob as globmod

    from januscribe.inversion import InversionConfig, train_soft_token
    from januscribe.subjects import SubjectRegistry

    bundle = _bundle()
    reg = SubjectRegistry.from_yaml(registry)
    subject = reg.get(subject_id)

    paths = (
        [Path(p) for p in sorted(globmod.glob(images))]
        if images
        else subject.reference_paths(reg.root)
    )
    if not paths:
        raise typer.BadParameter(
            f"no reference images for {subject_id!r}; run `januscribe build-refs` first"
        )

    cfg = InversionConfig(steps=steps, lr=lr, batch_size=batch_size, seed=seed)
    result = train_soft_token(bundle, subject, paths, cfg=cfg)
    path = result.soft_token.save(Path(out) / f"{subject_id}.safetensors")

    typer.echo(str(path))
    typer.echo(
        f"token {result.soft_token.token}  "
        f"loss {result.first_loss:.4f} -> {result.final_loss:.4f}  "
        f"{result.seconds:.0f}s"
    )


@app.command()
def repl() -> None:
    """Load the model once, then run many commands against it.

    The poor man's daemon. A 7B load costs minutes, so interactive work and
    ablation sweeps should not pay it per command. Type subcommands exactly as
    you would on the shell, without the leading `januscribe`.
    """
    bundle = _bundle()
    typer.echo(json.dumps(bundle.describe(), indent=2))
    typer.echo(
        "model held in memory. commands: gen | ask | vq-roundtrip | subjects | "
        "build-refs | baseline | info | quit"
    )
    _STATE["locked"] = True  # keep the settings (and therefore the loaded model) fixed
    while True:
        try:
            line = input("januscribe> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            _STATE.pop("locked", None)
            return
        if not line:
            continue
        if line in {"quit", "exit", ":q"}:
            _STATE.pop("locked", None)
            return
        try:
            app(shlex.split(line), standalone_mode=False)
        except SystemExit:
            pass
        except Exception as exc:  # keep the model loaded across mistakes
            log.error("repl_command_failed", command=line, error=str(exc))


if __name__ == "__main__":
    app()
