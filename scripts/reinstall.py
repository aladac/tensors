#!/usr/bin/env python3
"""Reinstall tensors locally and on junkpile with hash-suffixed versioning."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INIT_FILE = PROJECT_ROOT / "tensors" / "__init__.py"
JUNKPILE_HOST = "chi@junkpile"
JUNKPILE_PATH = "/opt/tensors"


def get_version() -> str:
    """Get current version from __init__.py."""
    content = INIT_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Could not find __version__ in tensors/__init__.py")
    return match.group(1)


def get_base_version(version: str) -> str:
    """Strip any +hash suffix from version."""
    return version.split("+", maxsplit=1)[0]


def get_git_hash() -> str:
    """Get short git hash of HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def set_version(new_version: str) -> None:
    """Update version in __init__.py."""
    content = INIT_FILE.read_text()
    content = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"', content)
    INIT_FILE.write_text(content)
    print(f"  Updated tensors/__init__.py to {new_version}")


def run(cmd: list[str], *, check: bool = True, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, cwd=cwd or PROJECT_ROOT)


def git_commit_version(version: str) -> bool:
    """Commit version change if there are changes."""
    # Check if there are changes
    result = run(["git", "diff", "--quiet", "tensors/__init__.py"], check=False, capture=True)
    if result.returncode == 0:
        return False  # No changes

    run(["git", "add", "tensors/__init__.py"])
    run(["git", "commit", "-m", f"Version {version}"])
    return True


def git_push_if_ahead() -> bool:
    """Push to remote if ahead."""
    result = run(["git", "status", "--porcelain", "-b"], capture=True)
    if "ahead" in result.stdout:
        run(["git", "push"])
        return True
    return False


def install_local() -> None:
    """Install locally with uv."""
    print("\n[3/5] Installing locally...")
    run(["uv", "pip", "install", "-e", "."], check=True)


def sync_to_junkpile() -> None:
    """Sync project to junkpile."""
    print("\n[4/5] Syncing to junkpile...")
    excludes = [
        ".git",
        ".venv",
        "__pycache__",
        "*.pyc",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "*.egg-info",
        "node_modules",
        ".tmp",
    ]
    cmd = ["rsync", "-avz", "--delete"]
    for exc in excludes:
        cmd.extend(["--exclude", exc])
    cmd.extend([f"{PROJECT_ROOT}/", f"{JUNKPILE_HOST}:{JUNKPILE_PATH}/"])
    run(cmd)


def install_junkpile() -> None:
    """Install on junkpile."""
    print("\n[5/5] Installing on junkpile...")
    run(["ssh", JUNKPILE_HOST, f"cd {JUNKPILE_PATH} && pip install -e '.[server]'"])


def main() -> int:
    """Main entry point."""
    current = get_version()
    base_version = get_base_version(current)
    git_hash = get_git_hash()
    new_version = f"{base_version}+{git_hash}"

    print("\n=== Tensors Reinstall ===")
    print(f"  Current version: {current}")
    print(f"  Base version:    {base_version}")
    print(f"  Git hash:        {git_hash}")
    print(f"  New version:     {new_version}")

    if current == new_version:
        print("\n[1/5] Version already current, skipping update")
    else:
        print(f"\n[1/5] Updating version to {new_version}...")
        set_version(new_version)

        print("\n[2/5] Committing version change...")
        if git_commit_version(new_version):
            print("  Committed.")
            git_push_if_ahead()
        else:
            print("  No changes to commit.")

    try:
        install_local()
        sync_to_junkpile()
        install_junkpile()
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}")
        return 1

    # Verify installation
    print("\n=== Verification ===")
    run(["uv", "run", "tsr", "--version"])

    print(f"\n Done! tensors {new_version} installed locally and on junkpile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
