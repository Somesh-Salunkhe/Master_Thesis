import os
from pathlib import Path

def get_project_root() -> Path:
    """Finds the project root by looking for anchor directories."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir() or (parent / "src").is_dir() or (parent / "scripts").is_dir():
            return parent
    return Path(os.getcwd()).resolve()

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_output_dir() -> Path:
    return get_project_root() / "output"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
