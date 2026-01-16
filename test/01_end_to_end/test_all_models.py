import pytest
from pathlib import Path

# 1. Define the path to your configs
CONFIG_DIR = Path(__file__).parent / "configs"
# 2. Automatically find all .yaml files
CONFIG_FILES = list(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_execution(config_path, workspace, run_bench):
    """
    This test will run once for EVERY yaml file found in the directory.
    'ids' ensures the test output shows the filename (e.g., [scNODE.yaml]).
    """
    result = run_bench(config_path, "auto_train_test", workspace)

    assert result.returncode == 0

    output_files = list((workspace / "outputs").iterdir())
    assert len(output_files) > 0, f"Config {config_path.name} produced no output."

    # let's verify that we have:
    # 1. database object
    # 2. metric outputs
    # 3. dataset pickle
    # 4. model config
    required_files = [
        "dataset.pkl",
        "model_config.yaml",
    ]

    # TODO: read from the database as well
