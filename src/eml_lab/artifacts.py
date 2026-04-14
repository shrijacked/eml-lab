"""Artifact manifests shared across experiment entrypoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactFile:
    label: str
    path: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactManifest:
    root_dir: str
    manifest_path: str
    files: tuple[ArtifactFile, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "root_dir": self.root_dir,
            "manifest_path": self.manifest_path,
            "files": [artifact.to_dict() for artifact in self.files],
            "metadata": self.metadata,
        }


def write_artifact_manifest(
    root_dir: str | Path,
    *,
    files: list[ArtifactFile],
    metadata: Mapping[str, object] | None = None,
    filename: str = "manifest.json",
) -> ArtifactManifest:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / filename
    manifest = ArtifactManifest(
        root_dir=str(root),
        manifest_path=str(manifest_path),
        files=tuple(files),
        metadata=dict(metadata or {}),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest
