"""Read a Substance archive (.sbsar): a 7z archive of a graph's files."""

import tempfile
from pathlib import Path

import py7zr


def read_archive(sbsar: Path) -> dict[str, bytes]:
    """Return the archive's files by name (posix paths inside it)."""
    with tempfile.TemporaryDirectory() as tmp, py7zr.SevenZipFile(sbsar) as archive:
        archive.extractall(tmp)
        return {
            p.relative_to(tmp).as_posix(): p.read_bytes()
            for p in Path(tmp).rglob("*")
            if p.is_file()
        }


def member(files: dict[str, bytes], suffix: str) -> bytes:
    """Return the archive file whose name ends in ``suffix``."""
    for name, data in files.items():
        if name.endswith(suffix):
            return data
    msg = f"no {suffix} file in the archive; not an .sbsar?"
    raise ValueError(msg)
