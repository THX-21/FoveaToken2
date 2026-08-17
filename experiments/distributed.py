from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}"

    def shard(self, values: Sequence[T]) -> list[T]:
        return list(values[self.rank :: self.world_size])

    def barrier(self) -> None:
        if self.enabled:
            torch.distributed.barrier()

    def rank_path(self, path: str | Path) -> Path:
        destination = Path(path)
        return destination.with_name(
            f"{destination.stem}.rank{self.rank}{destination.suffix}"
        )


_CONTEXT: DistributedContext | None = None


def distributed_context() -> DistributedContext:
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            f"invalid distributed environment: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process experiment execution requires CUDA")
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
    _CONTEXT = DistributedContext(rank, local_rank, world_size)
    return _CONTEXT


def merge_rank_jsonl(path: str | Path, *, key: str, remove_shards: bool = True) -> int:
    """Merge rank-local JSONL files into one de-duplicated primary file."""

    destination = Path(path)
    records: dict[str, dict[str, Any]] = {}
    if destination.exists():
        _read_jsonl_into(destination, key, records)
    shards = sorted(destination.parent.glob(f"{destination.stem}.rank*{destination.suffix}"))
    for shard in shards:
        _read_jsonl_into(shard, key, records)
    if shards:
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records.values():
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination)
        if remove_shards:
            for shard in shards:
                shard.unlink()
    return len(records)


def merge_rank_text(path: str | Path) -> None:
    """Append rank-local text files in rank order and remove the shards."""

    destination = Path(path)
    shards = sorted(destination.parent.glob(f"{destination.stem}.rank*{destination.suffix}"))
    if not shards:
        return
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("wb") as output:
        if destination.exists():
            output.write(destination.read_bytes())
        for shard in shards:
            output.write(shard.read_bytes())
    temporary.replace(destination)
    for shard in shards:
        shard.unlink()


def _read_jsonl_into(
    path: Path,
    key: str,
    records: dict[str, dict[str, Any]],
) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing string key {key!r} in {path}")
        records[value] = record
