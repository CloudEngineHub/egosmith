"""Source map: where every exported episode sits in the *original* dataset's media.

A source map is a parquet table with one row per WebDataset episode key. The LeRobot converter
joins it in to write the per-frame ``source_frame_index`` / ``source_frame_observed`` features and
the per-episode ``source_media*`` / ``calibration/source_camera`` columns that a labels-only release
needs for rehydration. The map itself is produced by a dataset-specific locator (for BuildAI see
``buildai_source_map.py``); this module only defines the table and the column names.

Columns
-------
key                   WebDataset episode key (``record.key`` in the converter)
source_media          media path relative to the source dataset root; ``tar/member`` when packed
source_media_fps      native frame rate of that media
source_frame_start    frame number (at ``source_media_fps``) of the episode's first exported frame
frame_stride          source frames per exported frame (1 = same rate)
observed_stride       every k-th exported frame carries a measured label (1 = all frames observed)
undistort             "" or the name of the mapping applied to source frames ("opencv_fisheye_knew_k")
source_camera         8 floats [fx, fy, cx, cy, k1, k2, k3, k4] of the source camera, NaN when unknown
match_score           locator confidence (1.0 = exact)
runner_up             best competing score away from the chosen offset
status                "ok" | "ambiguous" | "overrun" | "inconsistent" | "no_media" | "no_key"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_FRAME_INDEX_KEY = "source_frame_index"
SOURCE_FRAME_OBSERVED_KEY = "source_frame_observed"
SOURCE_MEDIA_KEY = "source_media"
SOURCE_MEDIA_FPS_KEY = "source_media_fps"
SOURCE_MEDIA_FRAME_START_KEY = "source_media_frame_start"
SOURCE_MEDIA_FRAME_END_KEY = "source_media_frame_end"
SOURCE_MEDIA_UNDISTORT_KEY = "source_media_undistort"
SOURCE_CAMERA_KEY = "calibration/source_camera"
IMAGE_SIZE_KEY = "calibration/head_image_size"
SOURCE_MAP_PATH = "meta/source_map.parquet"

SOURCE_CAMERA_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
UNDISTORT_FISHEYE_KNEW_K = "opencv_fisheye_knew_k"
OK_STATUSES = ("ok",)
NAN_CAMERA = [float("nan")] * len(SOURCE_CAMERA_NAMES)


@dataclass
class SourceMapRow:
    key: str
    source_media: str = ""
    source_media_fps: float = 0.0
    source_frame_start: int = -1
    frame_stride: int = 1
    observed_stride: int = 1
    undistort: str = ""
    source_camera: list[float] = field(default_factory=lambda: list(NAN_CAMERA))
    match_score: float = 0.0
    runner_up: float = 0.0
    status: str = "no_media"

    @property
    def usable(self) -> bool:
        return self.status in OK_STATUSES and self.source_frame_start >= 0 and bool(self.source_media)

    def camera_dict(self) -> dict | None:
        if not self.source_camera or any(math.isnan(v) for v in self.source_camera):
            return None
        return dict(zip(SOURCE_CAMERA_NAMES, [float(v) for v in self.source_camera]))


SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("source_media", pa.string()),
        ("source_media_fps", pa.float64()),
        ("source_frame_start", pa.int64()),
        ("frame_stride", pa.int64()),
        ("observed_stride", pa.int64()),
        ("undistort", pa.string()),
        ("source_camera", pa.list_(pa.float64())),
        ("match_score", pa.float64()),
        ("runner_up", pa.float64()),
        ("status", pa.string()),
    ]
)


def write_source_map(rows: list[SourceMapRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [] for name in SCHEMA.names}
    for row in rows:
        d = asdict(row)
        for name in SCHEMA.names:
            columns[name].append(d[name])
    table = pa.Table.from_pydict(columns, schema=SCHEMA)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, str(tmp), compression="snappy")
    tmp.replace(path)


class SourceMap:
    """In-memory episode-key -> SourceMapRow lookup."""

    def __init__(self, rows: dict[str, SourceMapRow], path: str | None = None):
        self.rows = rows
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "SourceMap":
        table = pq.read_table(str(path))
        rows: dict[str, SourceMapRow] = {}
        for rec in table.to_pylist():
            row = SourceMapRow(
                key=str(rec["key"]),
                source_media=str(rec.get("source_media") or ""),
                source_media_fps=float(rec.get("source_media_fps") or 0.0),
                source_frame_start=int(rec.get("source_frame_start") if rec.get("source_frame_start") is not None else -1),
                frame_stride=int(rec.get("frame_stride") or 1),
                observed_stride=int(rec.get("observed_stride") or 1),
                undistort=str(rec.get("undistort") or ""),
                source_camera=[float(v) for v in (rec.get("source_camera") or NAN_CAMERA)],
                match_score=float(rec.get("match_score") or 0.0),
                runner_up=float(rec.get("runner_up") or 0.0),
                status=str(rec.get("status") or "no_media"),
            )
            rows[row.key] = row
        return cls(rows, str(path))

    def get(self, key: str) -> SourceMapRow | None:
        return self.rows.get(key)

    def __len__(self) -> int:
        return len(self.rows)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows.values():
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts


def merge_source_maps(paths: list[str | Path]) -> dict[str, SourceMapRow]:
    """Later maps override earlier ones for the same key (lets a rerun patch a few episodes)."""
    rows: dict[str, SourceMapRow] = {}
    for p in paths:
        rows.update(SourceMap.load(p).rows)
    return rows
