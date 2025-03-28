from pathlib import Path


def find_project_root() -> Path | None:
    current = Path(".").resolve()

    while True:
        if (current / ".git").exists():
            return current

        if current.parent == current:
            print("WARNING: No .git dir found")
            return current

        current = current.parent
