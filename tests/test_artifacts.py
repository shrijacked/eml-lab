import json
from pathlib import Path

from eml_lab.artifacts import ArtifactFile, ArtifactManifest, write_artifact_manifest


def test_write_artifact_manifest_round_trips(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")

    manifest = write_artifact_manifest(
        tmp_path,
        files=[ArtifactFile(label="payload", path=str(payload), kind="json")],
        metadata={"kind": "test"},
    )

    assert isinstance(manifest, ArtifactManifest)
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["metadata"]["kind"] == "test"
    assert data["files"][0]["label"] == "payload"
