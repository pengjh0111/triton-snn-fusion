import operator
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class TemporalSpatialCanonicalizeStats:
    canonicalize_cat_chunk_removed: int = 0
    canonicalize_chunk_cat_removed: int = 0
    canonicalize_getitem_cat_removed: int = 0
    canonicalize_view_folded: int = 0
    canonicalize_dead_nodes_removed: int = 0
    iterations: int = 0
    final_cat_count: int = 0
    final_chunk_count: int = 0
    final_getitem_count: int = 0
    skipped: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def skip(self, reason: str, message: str):
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.log.append(f"SKIP[{reason}] {message}")


def _target_text(node: torch.fx.Node) -> str:
    return str(node.target)


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _getitem_index(node: torch.fx.Node):
    if not _is_getitem(node) or len(node.args) < 2:
        return None
    return node.args[1]


def _is_cat(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (node.target is torch.cat or "cat" in _target_text(node))


def _is_chunk(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (node.target is torch.chunk or "chunk" in _target_text(node))


def _chunk_input(node: torch.fx.Node):
    if not _is_chunk(node) or not node.args:
        return None
    return node.args[0] if isinstance(node.args[0], torch.fx.Node) else None


def _chunk_count(node: torch.fx.Node) -> Optional[int]:
    if not _is_chunk(node):
        return None
    if len(node.args) > 1:
        try:
            return int(node.args[1])
        except Exception:
            return None
    if "chunks" in node.kwargs:
        try:
            return int(node.kwargs["chunks"])
        except Exception:
            return None
    return None


def _cat_inputs(node: torch.fx.Node) -> Optional[List[torch.fx.Node]]:
    if not _is_cat(node) or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, (tuple, list)):
        return None
    if not all(isinstance(item, torch.fx.Node) for item in first):
        return None
    return list(first)


def _collect_chunk_getitems(chunk_node: torch.fx.Node) -> Optional[Dict[int, torch.fx.Node]]:
    out: Dict[int, torch.fx.Node] = {}
    for user in list(chunk_node.users):
        if not _is_getitem(user):
            return None
        idx = _getitem_index(user)
        if not isinstance(idx, int):
            return None
        out[idx] = user
    return out


def _replace_chunk_of_cat(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for chunk in list(gm.graph.nodes):
        cat = _chunk_input(chunk)
        if not isinstance(cat, torch.fx.Node) or not _is_cat(cat):
            continue
        inputs = _cat_inputs(cat)
        count = _chunk_count(chunk)
        getitems = _collect_chunk_getitems(chunk)
        if inputs is None or count != len(inputs) or getitems is None:
            continue
        if sorted(getitems) != list(range(len(inputs))):
            continue
        for idx, original in enumerate(inputs):
            getitems[idx].replace_all_uses_with(original)
        for idx in sorted(getitems, reverse=True):
            if len(getitems[idx].users) == 0:
                gm.graph.erase_node(getitems[idx])
        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)
        if len(cat.users) == 0:
            gm.graph.erase_node(cat)
        stats.canonicalize_cat_chunk_removed += 1
        changed = True
    return changed


def _replace_cat_of_chunk(gm: torch.fx.GraphModule, stats: TemporalSpatialCanonicalizeStats) -> bool:
    changed = False
    for cat in list(gm.graph.nodes):
        if not _is_cat(cat):
            continue
        inputs = _cat_inputs(cat)
        if not inputs:
            continue
        if not all(_is_getitem(item) and isinstance(item.args[0], torch.fx.Node) for item in inputs):
            continue
        chunk = inputs[0].args[0]
        if not isinstance(chunk, torch.fx.Node) or not _is_chunk(chunk):
            continue
        if any(item.args[0] is not chunk for item in inputs):
            continue
        if [_getitem_index(item) for item in inputs] != list(range(len(inputs))):
            continue
        count = _chunk_count(chunk)
        source = _chunk_input(chunk)
        if count != len(inputs) or source is None:
            continue
        cat.replace_all_uses_with(source)
        if len(cat.users) == 0:
            gm.graph.erase_node(cat)
        for item in reversed(inputs):
            if len(item.users) == 0:
                gm.graph.erase_node(item)
        if len(chunk.users) == 0:
            gm.graph.erase_node(chunk)
        stats.canonicalize_chunk_cat_removed += 1
        stats.canonicalize_getitem_cat_removed += len(inputs)
        changed = True
    return changed


def _count_nodes(gm: torch.fx.GraphModule):
    cats = chunks = getitems = 0
    for node in gm.graph.nodes:
        if _is_cat(node):
            cats += 1
        elif _is_chunk(node):
            chunks += 1
        elif _is_getitem(node):
            getitems += 1
    return cats, chunks, getitems


def _count_all_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for _ in gm.graph.nodes)


def canonicalize_temporal_spatial_ir(
    gm: torch.fx.GraphModule,
    *,
    max_iter: int = 8,
    dump_dir: Optional[Path] = None,
    strict: bool = False,
) -> TemporalSpatialCanonicalizeStats:
    stats = TemporalSpatialCanonicalizeStats()
    try:
        for iteration in range(max_iter):
            stats.iterations = iteration + 1
            changed = False
            changed |= _replace_cat_of_chunk(gm, stats)
            changed |= _replace_chunk_of_cat(gm, stats)
            before = _count_all_nodes(gm)
            gm.graph.eliminate_dead_code()
            after = _count_all_nodes(gm)
            if after < before:
                stats.canonicalize_dead_nodes_removed += before - after
                changed = True
            gm.graph.lint()
            gm.recompile()
            if not changed:
                break
        stats.final_cat_count, stats.final_chunk_count, stats.final_getitem_count = _count_nodes(gm)
        message = (
            "[CANONICALIZE] "
            f"cat_chunk_removed={stats.canonicalize_cat_chunk_removed} "
            f"chunk_cat_removed={stats.canonicalize_chunk_cat_removed} "
            f"getitem_cat_removed={stats.canonicalize_getitem_cat_removed} "
            f"dead={stats.canonicalize_dead_nodes_removed} "
            f"final_cat={stats.final_cat_count} final_chunk={stats.final_chunk_count} "
            f"final_getitem={stats.final_getitem_count}"
        )
        stats.log.append(message)
        print(message)
    except Exception as exc:
        if strict:
            raise
        stats.skip("exception", str(exc))
        print(f"[CANONICALIZE][SKIP] {exc}")
        traceback.print_exc()
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / "temporal_spatial_canonicalize.txt").write_text(
            "\n".join(stats.log + [f"stats={asdict(stats)}"]) + "\n",
            encoding="utf-8",
        )
    return stats
