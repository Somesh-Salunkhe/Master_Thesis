import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Finds the project root by looking for key directories.
    Returns a pathlib.Path object.
    """
    # Start from the directory of this file
    current = Path(__file__).resolve()
    
    # Climb up until we find 'configs' or 'src'
    for parent in current.parents:
        if (parent / "configs").is_dir() or (parent / "src").is_dir():
            return parent
    
    # Fallback to current working directory if not found
    return Path(os.getcwd()).resolve()

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_output_dir() -> Path:
    return get_project_root() / "outputs"

def get_config_dir() -> Path:
    return get_project_root() / "configs"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
