import pytest
import subprocess
from pathlib import Path
import sys


@pytest.fixture
def run_bench():
    def _run(config_path, run_type, workspace):
        project_root = Path(__file__).parent.parent
        cmd = [
            "python",
            "src/main.py",
            "--config",
            str(config_path),
            "--run_type",
            run_type,
            "--output_dir",
            str(workspace / "outputs"),
            "--database_path",
            str(workspace / "crispy_fishstick.db"),
            "--log_file",
            str(project_root / "test" / "logs" / "tmp.log"),
            "--log_level",
            "DEBUG",
        ]

        # Use subprocess.Popen to stream logs in real-time
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Print each line to stdout so pytest can capture/display it
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # delete the log file if it exists
        log_file_path = project_root / "test" / "logs" / "tmp.log"
        if log_file_path.exists():
            log_file_path.unlink()

        return process

    return _run


@pytest.fixture
def workspace(tmp_path):
    """Creates and cleans up directories for every test."""
    out = tmp_path / "outputs"
    out.mkdir()
    return tmp_path
