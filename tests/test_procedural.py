"""The procedural material wiring and the library cache."""

import pickle

from sbsar2ptex import library
from sbsar2ptex.procedural import material


def test_material_wires_slots_and_converts_normals() -> None:
    nodes, wires = material(
        {"albedo": ("a", 0), "normal": ("n", 1)}, normal_directx=True, size=11
    )
    names = {n["name"]: n for n in nodes}
    assert names["normal_convert"]["parameters"]["op"] == 1
    assert {"from": "a", "from_port": 0, "to": "Material", "to_port": 0} in wires
    assert {"from": "n", "from_port": 1, "to": "normal_convert", "to_port": 0} in wires
    assert {
        "from": "normal_convert", "from_port": 0, "to": "Material", "to_port": 4,
    } in wires  # fmt: skip


def test_library_load_reads_its_cache(tmp_path) -> None:
    with (tmp_path / library.PICKLE).open("wb") as f:
        pickle.dump((library.VERSION, ["kept"]), f)
    assert library.load(tmp_path, cooker="no-such-cooker") == ["kept"]


def test_library_skips_packages_that_failed(tmp_path) -> None:
    (tmp_path / library.FAILED).write_text("broken\n")
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "broken.sbs").write_text("")
    assert library.cook(tmp_path / "no-cooker", packages, tmp_path) == []
