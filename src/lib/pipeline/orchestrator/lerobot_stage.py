"""Optional ``lerobot`` stage: convert the built WebDataset into a LeRobot v3.0 dataset.

The stage is a thin adapter around ``scripts/build/wds_to_lerobot.py`` (see
``docs/dataset_format.md``). It never runs unless ``lerobot`` is in ``--stages``; the
default stage strings are untouched. Everything in the config's ``lerobot:`` block that is not
resolved here is forwarded verbatim as CLI flags, so the converter's options need no mirroring.

```yaml
lerobot:                      # all keys optional
  output_dir: /data/lerobot/my_dataset   # default: <final_dataset_root>/../lerobot
  fps: 30                     # default: int(build.target_fps)
  hand_frame: camera
  state_layout: egosteer74
  task_source: language
  no_video: false             # true = labels-only export
  validate: true              # structural checks + no-depth guard after conversion
  workers: 0                  # 0 = auto
```
"""

from __future__ import annotations

from pathlib import Path

from .helpers import cli_args_from_mapping

LEROBOT_RESERVED_KEYS = {"wds_dir", "output_dir", "descriptor_manifest", "source_map", "validate_only", "overwrite"}
_LEROBOT_DEFAULTS = {"validate": True}


def resolve_lerobot_output_dir(*, lerobot_cfg: dict, paths_cfg: dict, final_dataset_root: Path) -> Path:
    """``lerobot.output_dir`` > ``paths.lerobot_root`` > sibling of the WebDataset root named ``lerobot``."""
    if lerobot_cfg.get("output_dir"):
        return Path(str(lerobot_cfg["output_dir"]))
    if paths_cfg.get("lerobot_root"):
        return Path(str(paths_cfg["lerobot_root"]))
    return Path(final_dataset_root).parent / "lerobot"


def build_lerobot_stage_command(
    *,
    python: str,
    project_root: Path,
    final_dataset_root: Path,
    active_manifest_path: Path,
    paths_cfg: dict,
    build_cfg: dict,
    lerobot_cfg: dict,
    resume: bool,
) -> tuple[list[str], Path]:
    """Return (argv, output_dir) for the converter subprocess."""
    lerobot_cfg = dict(lerobot_cfg or {})
    output_dir = resolve_lerobot_output_dir(lerobot_cfg=lerobot_cfg, paths_cfg=paths_cfg, final_dataset_root=final_dataset_root)
    passthrough = {k: v for k, v in lerobot_cfg.items() if k not in LEROBOT_RESERVED_KEYS}
    for key, default in _LEROBOT_DEFAULTS.items():
        passthrough.setdefault(key, default)
    if "fps" not in passthrough:
        target_fps = build_cfg.get("target_fps")
        if target_fps is None:
            raise ValueError("lerobot stage needs `lerobot.fps` or `build.target_fps`")
        fps = float(target_fps)
        if abs(fps - round(fps)) > 1e-6:
            raise ValueError(f"lerobot stage needs an integer fps, build.target_fps is {fps}")
        passthrough["fps"] = int(round(fps))
    cmd = [
        python,
        str(Path(project_root) / "scripts" / "build" / "wds_to_lerobot.py"),
        "--wds_dir",
        str(final_dataset_root),
        "--output_dir",
        str(output_dir),
        "--descriptor_manifest",
        str(active_manifest_path),
        *cli_args_from_mapping(passthrough),
    ]
    if lerobot_cfg.get("source_map"):
        cmd += ["--source_map", str(lerobot_cfg["source_map"])]
    if resume:
        cmd.append("--resume")
    return cmd, output_dir


__all__ = ["LEROBOT_RESERVED_KEYS", "build_lerobot_stage_command", "resolve_lerobot_output_dir"]
