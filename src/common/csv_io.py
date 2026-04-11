from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Type, TypeVar

from src.common.schemas import CsvRowMixin

T = TypeVar("T", bound=CsvRowMixin)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: Iterable[T], row_type: Type[T]) -> None:
    materialized: List[T] = list(rows)
    row_type.validate_rows(materialized)
    ensure_parent_dir(path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_type.field_names())
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.to_dict())


def read_rows(path: Path, row_type: Type[T]) -> List[T]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in row_type.field_names() if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return [row_type.from_dict(row) for row in reader]
