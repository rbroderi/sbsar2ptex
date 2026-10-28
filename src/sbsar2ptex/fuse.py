"""Fuse shader nodes inside a library node into fewer Material Maker nodes.

Material Maker builds one shader from everything up to a buffer, each node a
function called wherever it is read, so a node read by one other node only
may as well be that node's helper: the same code, the same arithmetic, the
same pixels, one node fewer. Inside a recognized library node (or one of the
author's repeated units) that changes nothing a user could edit in Substance
either, which shows such a node as one; every knob stays a live parameter of
the fused node (renamed with a prefix, labelled as before), and the graph's
inputs and the instance's parameters keep their names, so what drives them
still does.
"""

import re
from typing import Any

from sbsar2ptex.translate import GRAPH_PARAM
from sbsar2ptex.translate import INSTANCE_PARAM

SHARED = (GRAPH_PARAM, INSTANCE_PARAM)
"""Parameters shared by name across nodes: never renamed."""


def _plain(node: dict[str, Any]) -> bool:
    """Whether ``node`` is a shader of the form translate.shader makes."""
    if node.get("type") != "shader":
        return False
    m = node["shader_model"]
    outs = m.get("outputs", [])
    return (
        not m.get("code")
        and len(outs) == 1
        and outs[0].get(outs[0]["type"]) == "$(name)_fct($uv)"
        and all(i.get("function") for i in m.get("inputs", []))
    )


def _merge_globals(a: str, b: str) -> str | None:
    """Return the global code of a node fusing two; None if they could clash.

    Material Maker includes a global block once per distinct text, so the
    fused node may carry one of the two only (or both equal): a block joining
    two would repeat a helper another node's block also defines.
    """
    if not a or a == b:
        return b
    if not b:
        return a
    return None


def _rename(text: str, names: list[str], prefix: str, *, calls: bool) -> str:
    """Prefix ``$name`` references (``calls``: only ``$name(``, inputs)."""
    for n in sorted(names, key=len, reverse=True):
        tail = r"(?=\()" if calls else r"(?![\w(])"
        text = re.sub(r"\$" + re.escape(n) + tail, f"${prefix}{n}", text)
    return text


def fuse_into(
    reader: dict[str, Any], port: int, inner: dict[str, Any], prefix: str
) -> dict[str, Any] | None:
    """Return ``reader`` with ``inner`` (its input ``port``) as a helper; None if not.

    The fused node's inputs are the reader's, less ``port``, then the inner
    node's, prefixed; see fuse for the wiring.
    """
    rm, im = reader["shader_model"], inner["shader_model"]
    shared = _merge_globals(im["global"], rm["global"])
    port_in = rm["inputs"][port]
    if shared is None or port_in["type"] != im["outputs"][0]["type"]:
        return None
    own = [p["name"] for p in im["parameters"] if not p["name"].startswith(SHARED)]
    ins = [i["name"] for i in im["inputs"]]
    tag = prefix.rstrip("_")
    body = im["instance"].replace("$(name)", f"$(name)_{tag}")
    body = _rename(body, ins, prefix, calls=True)
    body = _rename(body, own, prefix, calls=False)
    call = re.sub(
        r"\$" + re.escape(port_in["name"]) + r"(?=\()",
        f"$(name)_{tag}_fct",
        rm["instance"],
    )
    params = list(rm["parameters"])
    have = {p["name"] for p in params}
    values = dict(reader["parameters"])
    for p in im["parameters"]:
        name = p["name"] if p["name"].startswith(SHARED) else prefix + p["name"]
        if name not in have:
            params.append({**p, "name": name})
            have.add(name)
            values[name] = inner["parameters"].get(p["name"], p.get("default"))
    inputs = [i for k, i in enumerate(rm["inputs"]) if k != port]
    inputs += [{**i, "name": prefix + i["name"]} for i in im["inputs"]]
    return {
        **reader,
        "parameters": values,
        "shader_model": {
            **rm,
            "global": shared,
            "inputs": inputs,
            "instance": body + call,
            "parameters": params,
        },
    }


def fuse(
    nodes: list[dict[str, Any]], wires: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``nodes`` and ``wires`` (one level) with single-reader shaders fused.

    A shader whose one wire out goes to another shader at this level (not
    out of the level, nor to a buffer) becomes that shader's helper
    (fuse_into), until none is left to fuse.
    """
    nodes, wires = list(nodes), [dict(w) for w in wires]
    count = 0
    changed = True
    while changed:
        changed = False
        by_name = {n["name"]: n for n in nodes}
        outs: dict[str, list[dict[str, Any]]] = {}
        for w in wires:
            outs.setdefault(w["from"], []).append(w)
        for inner in list(nodes):
            ws = outs.get(inner["name"], [])
            if not _plain(inner) or len(ws) != 1 or ws[0]["from_port"]:
                continue
            w = ws[0]
            reader = by_name.get(w["to"])
            if reader is None or reader is inner or not _plain(reader):
                continue
            count += 1
            made = fuse_into(reader, w["to_port"], inner, f"u{count}_")
            if made is None:
                continue
            port, n_in = w["to_port"], len(reader["shader_model"]["inputs"])
            rewired = []
            for x in wires:
                if x is w:
                    continue
                if x["to"] == reader["name"] and x["to_port"] > port:
                    x = {**x, "to_port": x["to_port"] - 1}  # noqa: PLW2901
                elif x["to"] == inner["name"]:
                    x = {**x, "to": reader["name"], "to_port": n_in - 1 + x["to_port"]}  # noqa: PLW2901
                rewired.append(x)
            wires = rewired
            nodes = [made if n is reader else n for n in nodes if n is not inner]
            changed = True
            break
    return nodes, wires
