"""Substance's own library, cooked once and kept, for recognizing its nodes.

sbsar2ptex.recognize finds library nodes in a compiled graph by matching
the atomic nodes each library graph cooks to. The library's sources ship
with Substance 3D Designer (resources/packages/*.sbs); ``load`` cooks them
with its sbscooker into a cache folder the first time (a few minutes), and
keeps the templates read from them in a pickle, so later runs only load it.

The cache is per user: LOCALAPPDATA/sbsar2ptex/library on Windows, else
XDG_CACHE_HOME (or ~/.cache)/sbsar2ptex/library; ``--library`` picks
another.
"""

import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sbsar2ptex.recognize import Template
from sbsar2ptex.recognize import templates

COOKER_PLACES = (
    Path("C:/Program Files/Adobe/Adobe Substance 3D Designer"),
    Path("C:/Program Files/Adobe/Adobe Substance 3D Automation Toolkit"),
    Path("C:/Program Files/Allegorithmic/Substance Designer"),
    Path("/Applications/Adobe Substance 3D Designer.app/Contents/MacOS"),
    Path("/opt/Adobe/Adobe_Substance_3D_Designer"),
    Path("/opt/Adobe/Adobe_Substance_3D_Automation_Toolkit"),
)
"""Where sbscooker and the library packages are looked for."""

PICKLE = "templates.pickle"
FAILED = "failed.txt"
"""Packages that would not cook, one name a line."""
VERSION = 1
"""Bump when Template or the matching changes, to rebuild cached pickles."""


def cache_dir() -> Path:
    """Return the default library cache folder."""
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "sbsar2ptex" / "library"


def find_cooker(given: str | None = None) -> Path | None:
    """Return sbscooker: the one given, on PATH, or in an install folder."""
    if given:
        return Path(given)
    on_path = shutil.which("sbscooker")
    if on_path:
        return Path(on_path)
    for place in COOKER_PLACES:
        for name in ("sbscooker.exe", "sbscooker"):
            if (place / name).is_file():
                return place / name
    return None


def packages(cooker: Path) -> Path | None:
    """Return the library packages folder of the install ``cooker`` is in."""
    for place in (cooker.parent, *cooker.parents):
        folder = place / "resources" / "packages"
        if folder.is_dir():
            return folder
    return None


def cook(cooker: Path, folder: Path, cache: Path, jobs: int = 6) -> list[str]:
    """Cook every package in ``folder`` not cooked yet into ``cache``.

    Returns the packages that failed (some need assets or are not graphs);
    they are listed in FAILED, not to be tried again.
    """
    cache.mkdir(parents=True, exist_ok=True)
    failed_file = cache / FAILED
    tried = set(failed_file.read_text().split()) if failed_file.exists() else set()
    todo = [
        s
        for s in sorted(folder.glob("*.sbs"))
        if not (cache / f"{s.stem}.sbsar").exists() and s.stem not in tried
    ]

    def one(source: Path) -> tuple[str, int]:
        result = subprocess.run(  # noqa: S603 (the user's own Designer)
            [str(cooker), str(source), "--output-path", str(cache)],
            capture_output=True,
            check=False,
            timeout=900,
        )
        return source.stem, result.returncode

    if todo:
        print(f"cooking {len(todo)} library packages into {cache} (once)...")
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(one, todo))
    failed = [name for name, code in results if code]
    if failed:
        failed_file.write_text("\n".join(sorted(tried | set(failed))) + "\n")
    return failed


def load(
    cache: Path | None = None, cooker: str | None = None, *, refresh: bool = False
) -> list[Template]:
    """Return the library's templates, cooking and caching them if needed.

    An empty list when there is no cache and no Designer to make one.
    """
    cache = cache or cache_dir()
    stored = cache / PICKLE
    if stored.exists() and not refresh:
        with stored.open("rb") as f:
            version, kept = pickle.load(f)  # noqa: S301 (our own cache)
        if version == VERSION:
            return kept
    found = find_cooker(cooker)
    folder = packages(found) if found else None
    if found is not None and folder is not None:
        failed = cook(found, folder, cache)
        if failed:
            print(f"  {len(failed)} packages did not cook (skipped)")
    out: list[Template] = []
    for path in sorted(cache.glob("*.sbsar")):
        try:
            out += templates(path)
        except (ValueError, KeyError, IndexError, OSError):
            continue
    if out:
        with stored.open("wb") as f:
            pickle.dump((VERSION, out), f)
    return out
