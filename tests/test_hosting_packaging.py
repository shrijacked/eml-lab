from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_packaging_files_exist() -> None:
    assert (ROOT / "requirements.txt").read_text(encoding="utf-8").strip() == "-e ."
    assert (ROOT / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.11"
    assert (ROOT / ".streamlit" / "config.toml").exists()
    assert (ROOT / "docs" / "hosted-demo.md").exists()


def test_dockerfile_runs_streamlit_app() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "COPY .streamlit ./.streamlit" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "python -m pip install --no-cache-dir -e ." in dockerfile
    assert "streamlit" in dockerfile
    assert "src/eml_lab/app.py" in dockerfile
    assert "8501" in dockerfile


def test_dockerignore_excludes_local_artifacts() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".venv" in ignored
    assert "runs" in ignored
    assert "paper.pdf" in ignored


def test_ci_preinstalls_cpu_torch_before_package_install() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    cpu_index = "https://download.pytorch.org/whl/cpu"
    torch_install = f"python -m pip install --index-url {cpu_index} torch"
    package_install = 'python -m pip install -e ".[dev]"'

    assert torch_install in workflow
    assert package_install in workflow
    assert workflow.index(torch_install) < workflow.index(package_install)
