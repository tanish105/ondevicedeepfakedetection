from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from src.common.constants import MANIPULATION_TYPES, QUANTIZATION_IDS, SPLIT_NAMES, TASK_IDS

PathLike = Union[str, Path]
T = TypeVar("T")


def _normalize_path(value: PathLike) -> str:
    return value.as_posix() if isinstance(value, Path) else str(value)


def _is_optional(field_type: Any) -> bool:
    origin = get_origin(field_type)
    if origin is Union:
        return type(None) in get_args(field_type)
    return False


def _coerce_value(field_type: Any, value: Any) -> Any:
    if value in ("", None):
        if _is_optional(field_type):
            return None
        return value

    origin = get_origin(field_type)
    if origin is Union:
        non_none = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(non_none[0], value)

    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is str:
        return str(value)
    return value


class CsvRowMixin:
    @classmethod
    def field_names(cls) -> List[str]:
        return list(cls.__dataclass_fields__.keys())

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        for key, value in row.items():
            if isinstance(value, Path):
                row[key] = _normalize_path(value)
        return row

    @classmethod
    def from_dict(cls: Type[T], row: Dict[str, Any]) -> T:
        values: Dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for field_name, field_info in cls.__dataclass_fields__.items():
            if field_name not in row:
                raise KeyError(f"Missing required column '{field_name}' for {cls.__name__}")
            values[field_name] = _coerce_value(type_hints[field_name], row[field_name])
        return cls(**values)

    @classmethod
    def validate_rows(cls, rows: Sequence["CsvRowMixin"]) -> None:
        for row in rows:
            if not isinstance(row, cls):
                raise TypeError(f"Expected {cls.__name__}, got {type(row).__name__}")


@dataclass(frozen=True)
class VideoIndexRow(CsvRowMixin):
    video_id: str
    source_video_id: str
    manipulation_type: str
    video_path: str

    def __post_init__(self) -> None:
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")


@dataclass(frozen=True)
class SplitRow(CsvRowMixin):
    video_id: str
    source_video_id: str
    manipulation_type: str
    binary_label: int
    split: str
    video_path: str

    def __post_init__(self) -> None:
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")
        if self.binary_label not in (0, 1):
            raise ValueError(f"binary_label must be 0 or 1, got {self.binary_label}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported split: {self.split}")


@dataclass(frozen=True)
class FrameManifestRow(CsvRowMixin):
    task_id: str
    video_id: str
    frame_idx: int
    frame_source_index: int
    image_path: str
    face_found: int
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None
    width: int
    height: int
    manipulation_type: str
    binary_label: int
    split: str

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task_id: {self.task_id}")
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")
        if self.binary_label not in (0, 1):
            raise ValueError(f"binary_label must be 0 or 1, got {self.binary_label}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.face_found not in (0, 1):
            raise ValueError(f"face_found must be 0 or 1, got {self.face_found}")


@dataclass(frozen=True)
class CalibrationRow(CsvRowMixin):
    task_id: str
    video_id: str
    frame_idx: int
    image_path: str
    manipulation_type: str
    split: str

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task_id: {self.task_id}")
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")
        if self.split != "val":
            raise ValueError(f"Calibration split must be 'val', got {self.split}")


@dataclass(frozen=True)
class FramePredictionRow(CsvRowMixin):
    task_id: str
    quant_id: str
    split: str
    video_id: str
    frame_idx: int
    image_path: str
    manipulation_type: str
    binary_label: int
    score_fake: float
    pred_label: int
    latency_ms: float
    device_name: str
    runtime: str

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task_id: {self.task_id}")
        if self.quant_id not in QUANTIZATION_IDS:
            raise ValueError(f"Unsupported quant_id: {self.quant_id}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")
        if self.binary_label not in (0, 1):
            raise ValueError(f"binary_label must be 0 or 1, got {self.binary_label}")
        if self.pred_label not in (0, 1):
            raise ValueError(f"pred_label must be 0 or 1, got {self.pred_label}")


@dataclass(frozen=True)
class VideoPredictionRow(CsvRowMixin):
    task_id: str
    quant_id: str
    split: str
    video_id: str
    manipulation_type: str
    binary_label: int
    num_frames_used: int
    mean_score_fake: float
    majority_vote_pred: int
    fps: float
    device_name: str
    runtime: str

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task_id: {self.task_id}")
        if self.quant_id not in QUANTIZATION_IDS:
            raise ValueError(f"Unsupported quant_id: {self.quant_id}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.manipulation_type not in MANIPULATION_TYPES:
            raise ValueError(f"Unsupported manipulation_type: {self.manipulation_type}")
        if self.binary_label not in (0, 1):
            raise ValueError(f"binary_label must be 0 or 1, got {self.binary_label}")
        if self.majority_vote_pred not in (0, 1):
            raise ValueError(
                f"majority_vote_pred must be 0 or 1, got {self.majority_vote_pred}"
            )


@dataclass(frozen=True)
class FailureMetadataRow(CsvRowMixin):
    video_id: str
    frame_idx: int
    image_path: str
    fp32_score_fake: float
    int8_score_fake: float
    binary_label: int
    selection_reason: str

    def __post_init__(self) -> None:
        if self.binary_label not in (0, 1):
            raise ValueError(f"binary_label must be 0 or 1, got {self.binary_label}")


def validate_row_sequence(rows: Iterable[CsvRowMixin], cls: Type[CsvRowMixin]) -> None:
    cls.validate_rows(list(rows))
