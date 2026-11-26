"""Versioned LoRA adapter publication, activation and stale-adapter GC.

An adapter is a *model name* for the rollout engine: naming an old adapter
serves the old policy (and its cached KV blocks). Publishing in place is
therefore unsafe. This module mirrors the pinned TRL async GRPO behavior:

* publish ``{name}-v{N}`` into a fresh immutable directory;
* load the adapter on the rollout **before** advancing the model version;
* keep versions ``> N - (max_staleness + 1)`` servable (in-flight requests may
  still name them);
* unload the version exactly at that boundary and delete its files one version
  later, because the engine resolves ``lora_path`` lazily and a missing path is
  a fatal engine error, not a 404.

All state is persisted in ``manifest.json`` so a restarted controller resumes
the same monotonic version sequence and never resurrects a deleted adapter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Sequence

import tensorplay as tp

from .weights import DigestMismatch, VersionMismatch

MANIFEST = "manifest.json"
ADAPTER_FILE = "adapter.mega"
FORMAT = 1


@dataclasses.dataclass(frozen=True, slots=True)
class AdapterArtifact:
    name: str
    version: int
    path: str
    digest: str
    rank: int | None = None
    created_at: float = 0.0

    @property
    def lora_name(self) -> str:
        return f"{self.name}-v{self.version}"


def _digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


@dataclasses.dataclass
class AdapterRegistry:
    """Filesystem-backed, versioned adapter lifecycle manager."""

    root: str | Path
    name: str = "policy"
    max_staleness: int = 1
    max_adapters: int | None = None
    load: Callable[[AdapterArtifact], None] | None = None
    unload: Callable[[AdapterArtifact], None] | None = None
    load_hook: Callable[[str, int], None] | None = None
    unload_hook: Callable[[str, int], None] | None = None
    active_version: int = 0
    adapters: dict[int, AdapterArtifact] = dataclasses.field(default_factory=dict)
    _unloaded: set[int] = dataclasses.field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        if self.max_adapters is not None and self.max_adapters < 1:
            raise ValueError("max_adapters must be positive")
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _manifest_path(self) -> Path:
        return self.root / MANIFEST

    def _restore(self) -> None:
        path = self._manifest_path()
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT:
            raise ValueError(f"unsupported adapter registry format: {raw.get('format')}")
        if raw.get("name") != self.name:
            raise ValueError(f"adapter registry name mismatch: {raw.get('name')}")
        self.active_version = int(raw["active_version"])
        self.adapters = {
            int(version): AdapterArtifact(**artifact)
            for version, artifact in raw["adapters"].items()
        }
        for artifact in list(self.adapters.values()):
            directory = Path(artifact.path)
            if not directory.is_dir() or _digest(directory) != artifact.digest:
                raise DigestMismatch(f"adapter {artifact.lora_name} failed checksum on resume")
            if artifact.version > self.active_version:
                # Crash between publish and activate: the engine never loaded
                # this adapter, so drop it rather than reviving it on resume.
                shutil.rmtree(directory, ignore_errors=True)
                del self.adapters[artifact.version]
        self._unloaded = {
            version
            for version in self.adapters
            if version <= self.active_version - (self.max_staleness + 1)
        }
        self._persist()

    def _persist(self) -> None:
        _write_json(
            self._manifest_path(),
            {
                "format": FORMAT,
                "name": self.name,
                "active_version": self.active_version,
                "adapters": {
                    str(version): dataclasses.asdict(artifact)
                    for version, artifact in sorted(self.adapters.items())
                },
            },
        )

    @property
    def latest_version(self) -> int:
        return max(self.adapters, default=0)

    def path_for(self, version: int) -> Path:
        return self.root / f"{self.name}-v{version}"

    def publish(
        self, state: dict | str | Path, *, rank: int | None = None
    ) -> AdapterArtifact:
        """Atomically publish a new immutable adapter version."""
        version = self.latest_version + 1
        if version <= self.active_version:
            raise VersionMismatch("cannot publish a version below the active one")
        target = self.path_for(version)
        if target.exists():
            raise VersionMismatch(f"adapter version {version} already exists")
        staging = self.root / f".{self.name}-v{version}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        if isinstance(state, (str, Path)):
            source = Path(state)
            if source.is_dir():
                for item in source.rglob("*"):
                    if item.is_file():
                        destination = staging / item.relative_to(source)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, destination)
            else:
                shutil.copy2(source, staging / ADAPTER_FILE)
        else:
            tp.save(state, staging / ADAPTER_FILE)
        digest = _digest(staging)
        (staging / "adapter.json").write_text(
            json.dumps(
                {"name": self.name, "version": version, "rank": rank, "digest": digest},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        digest = _digest(staging)
        os.replace(staging, target)
        artifact = AdapterArtifact(
            name=self.name,
            version=version,
            path=str(target.resolve()),
            digest=digest,
            rank=rank,
            created_at=time.time(),
        )
        self.adapters[version] = artifact
        self._persist()
        return artifact

    def activate(self, version: int) -> AdapterArtifact:
        """Load an adapter on the rollout, then advance the active version."""
        artifact = self.adapters.get(version)
        if artifact is None:
            raise VersionMismatch(f"adapter version {version} is not published")
        if version <= self.active_version:
            raise VersionMismatch(f"version {version} is not newer than active {self.active_version}")
        if _digest(Path(artifact.path)) != artifact.digest:
            raise DigestMismatch(f"adapter {artifact.lora_name} changed on disk")
        if self.load is not None:
            self.load(artifact)
        if self.load_hook is not None:
            self.load_hook(artifact.lora_name, artifact.path)
        # The version moves only after the load succeeded: rollout workers name
        # their adapter from the model version, so an early bump serves stale
        # KV blocks under a new policy's name.
        self.active_version = version
        self._persist()
        return artifact

    def garbage_collect(self) -> list[str]:
        """Unload and delete adapters outside the in-flight staleness window."""
        stale = self.active_version - (self.max_staleness + 1)
        removed: list[str] = []

        def unload_if_loaded(version: int) -> None:
            if version <= 0 or version in self._unloaded or version not in self.adapters:
                return
            artifact = self.adapters[version]
            if self.unload is not None:
                self.unload(artifact)
            if self.unload_hook is not None:
                self.unload_hook(artifact.lora_name, artifact.path)
            self._unloaded.add(version)
            removed.append(artifact.lora_name)

        unload_if_loaded(stale)
        # The engine resolves paths lazily, so unload at N-(max_staleness+1)
        # but keep the files for one more sync before deleting them.
        file_cutoff = stale - 1
        for version in sorted(self.adapters):
            if version == self.active_version or version > file_cutoff:
                continue
            shutil.rmtree(self.adapters[version].path, ignore_errors=True)
            del self.adapters[version]
            removed.append(f"files-v{version}")
        if self.max_adapters is not None:
            keep = set(
                range(
                    max(1, self.active_version - self.max_staleness),
                    self.active_version + 1,
                )
            )
            for version in sorted(self.adapters):
                if len(self.adapters) <= self.max_adapters:
                    break
                if version in keep:
                    continue
                unload_if_loaded(version)
                shutil.rmtree(self.adapters[version].path, ignore_errors=True)
                del self.adapters[version]
                removed.append(f"files-v{version}")
        if removed:
            self._persist()
        return removed

    def reload(self, state: dict | str | Path, *, rank: int | None = None) -> AdapterArtifact:
        """Publish -> activate -> GC, the adapter equivalent of reload_weights."""
        artifact = self.publish(state, rank=rank)
        self.activate(artifact.version)
        self.garbage_collect()
        return artifact

    def status(self) -> dict:
        return {
            "active_version": self.active_version,
            "published": sorted(self.adapters),
            "servable": [
                version
                for version in self.adapters
                if version > self.active_version - (self.max_staleness + 1)
            ],
        }


class VersionedAdapterRollout:
    """Rollout decorator that turns ``reload_weights`` into a versioned publish.

    ``generate`` and every other attribute are forwarded to the wrapped rollout
    unchanged, so controllers keep using the narrow ``RolloutService`` trait.
    """

    def __init__(
        self,
        inner,
        registry: AdapterRegistry,
        *,
        rank_from_path=None,
    ) -> None:
        self.inner = inner
        self.registry = registry
        self.rank_from_path = rank_from_path

    def generate(self, request):
        return self.inner.generate(request)

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        artifact = self.registry.reload(path)
        if expected_digest is not None and artifact.digest != expected_digest:
            raise DigestMismatch("adapter digest does not match the expected digest")

    def __getattr__(self, name):
        return getattr(self.inner, name)


__all__ = [
    "AdapterArtifact",
    "AdapterRegistry",
    "VersionedAdapterRollout",
]
