"""Reject generated/private artifacts and broken local documentation links."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

FORBIDDEN_PARTS = {
    ".cursor",
    ".idea",
    ".vscode",
    "__pycache__",
    "CMakeFiles",
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "CTestTestfile.cmake",
}
FORBIDDEN_PREFIXES = ("._",)
FORBIDDEN_SUFFIXES = (
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".dll",
    ".dylib",
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
)
ALLOWED_SUFFIX_FILES = {
    "motion_retargeting/data/Solo8/solo8_motion_data.pt",
}
ALLOWED_LARGE_FILES = {"data/disc_data_c6.npy"}
MAX_TRACKED_BYTES = 10 * 1024 * 1024

# Frozen evaluator bytes are part of the recorded provenance and must not be
# rewritten solely for source-cleanliness checks.
TEXT_SCAN_EXEMPT = {
    "evaluation/quant_eval_v5.py",
    "evaluation/run_quant_eval_v5.py",
}
SELF_PATH = "tools/check_repo_hygiene.py"
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MACHINE_LOCAL_MARKERS = ("/home/che/", "C:\\Users\\")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [item.decode() for item in result.stdout.split(b"\0") if item]


def check_markdown_links(root: Path, rel: str, text: str, problems: list[str]) -> None:
    source = root / rel
    for raw_target in MARKDOWN_LINK_RE.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        if not target:
            continue
        candidate = (root / target.lstrip("/")) if target.startswith("/") else (source.parent / target)
        if not candidate.exists():
            problems.append(f"broken local markdown link in {rel}: {raw_target}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []

    for rel in tracked_files(root):
        path = root / rel
        parts = set(Path(rel).parts)
        name = path.name

        if parts & FORBIDDEN_PARTS:
            problems.append(f"forbidden generated/private path: {rel}")
        if name in FORBIDDEN_NAMES:
            problems.append(f"forbidden private/generated file: {rel}")
        if name.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"forbidden OS metadata: {rel}")
        if name.endswith(FORBIDDEN_SUFFIXES) and rel not in ALLOWED_SUFFIX_FILES:
            problems.append(f"forbidden generated binary/checkpoint: {rel}")

        if not path.is_file():
            continue

        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES and rel not in ALLOWED_LARGE_FILES:
            problems.append(f"tracked file exceeds 10 MiB: {rel} ({size} bytes)")

        if path.suffix not in TEXT_SUFFIXES or rel in TEXT_SCAN_EXEMPT:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if rel != SELF_PATH:
            for marker in MACHINE_LOCAL_MARKERS:
                if marker in text:
                    problems.append(f"machine-local path in {rel}: {marker}")

        if path.suffix == ".md":
            check_markdown_links(root, rel, text, problems)

    if problems:
        print("Repository hygiene check failed:")
        for problem in sorted(set(problems)):
            print(f"  - {problem}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
