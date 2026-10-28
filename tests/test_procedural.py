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
    (tmp_path / library.COOKED).write_text(f"{library.COOK_VERSION}\n")
    (tmp_path / library.FAILED).write_text("broken\n")
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "broken.sbs").write_text("")
    assert library.cook(tmp_path / "no-cooker", packages, tmp_path) == []


def test_library_cooked_another_way_is_cooked_again(tmp_path) -> None:
    (tmp_path / library.COOKED).write_text("1\n")
    (tmp_path / "old.sbsar").write_bytes(b"")
    (tmp_path / library.FAILED).write_text("broken\n")
    packages = tmp_path / "packages"
    packages.mkdir()
    library.cook(tmp_path / "no-cooker", packages, tmp_path)
    assert not (tmp_path / "old.sbsar").exists()
    assert not (tmp_path / library.FAILED).exists()


def test_matches_are_kept_per_graph_and_library(tmp_path, monkeypatch) -> None:
    calls = []

    def recognize(nodes, lib):
        calls.append(nodes)
        return []

    monkeypatch.setattr(library, "recognize", recognize)
    assert library.recognized(b"graph", [], [], tmp_path) == []
    assert library.recognized(b"graph", [], [], tmp_path) == []
    assert len(calls) == 1  # the second from the cache
    library.recognized(b"another graph", [], [], tmp_path)
    assert len(calls) == 2


def test_excluded_graphs_are_published() -> None:
    hidden = (
        '<graph><identifier v="old"/><graphOutputs><graphoutput><identifier v="o"/>'
        '<uid v="7"/></graphoutput></graphOutputs>'
        '<root><disabled v="1"/><rootOutputs/></root></graph>'
    )
    rootless = (
        '<graph><identifier v="helper"/><graphOutputs><graphoutput>'
        '<identifier v="o"/><uid v="9"/></graphoutput></graphOutputs></graph>'
    )
    text = library.publishable(hidden + rootless)
    assert library.DISABLED not in text
    assert '<root><rootOutputs><rootOutput><output v="9"/>' in text
