from __future__ import annotations

from bisect import bisect_left, bisect_right

from stardust.tree.atom import AtomIndex, Node, SpanOffset


class OffsetMap:
    def __init__(self) -> None:
        self._src_starts: list[int] = []
        self._dst_starts: list[int] = []
        self._lengths: list[int] = []

    def add(self, src_start: int, dst_start: int, length: int) -> None:
        self._src_starts.append(src_start)
        self._dst_starts.append(dst_start)
        self._lengths.append(length)

    def to_dst(self, src_start: int, src_end: int) -> SpanOffset:
        i = bisect_right(self._src_starts, src_start) - 1
        delta = self._dst_starts[i] - self._src_starts[i] if i >= 0 else 0
        return SpanOffset(start=src_start + delta, end=src_end + delta)

    def to_src(self, dst_start: int, dst_end: int) -> SpanOffset:
        i = bisect_right(self._dst_starts, dst_start) - 1
        delta = self._src_starts[i] - self._dst_starts[i] if i >= 0 else 0
        return SpanOffset(start=dst_start + delta, end=dst_end + delta)


class IntervalTree:
    def __init__(self, index: AtomIndex) -> None:
        atoms = [index.nodes[a] for a in index.atoms]
        self._atoms = sorted(atoms, key=lambda n: n.clean_offset.start)
        self._starts = [n.clean_offset.start for n in self._atoms]
        self._ends = [n.clean_offset.end for n in self._atoms]

    def find_overlapping(self, start: int, end: int) -> list[Node]:
        lo = bisect_left(self._ends, start)
        hi = bisect_right(self._starts, end)
        return [
            self._atoms[i]
            for i in range(lo, hi)
            if self._starts[i] < end and self._ends[i] > start
        ]


def get_ancestors(node_id: int, index: AtomIndex) -> list[Node]:
    ancestors: list[Node] = []
    current = index.nodes[node_id]
    while current.parent_id is not None:
        parent = index.nodes[current.parent_id]
        ancestors.append(parent)
        current = parent
    return list(reversed(ancestors))


def get_leaves(node_id: int, index: AtomIndex) -> list[int]:
    children = index.children.get(node_id, [])
    if not children:
        return [node_id]
    leaves: list[int] = []
    for child_id in children:
        leaves.extend(get_leaves(child_id, index))
    return leaves


def get_subtree(node_id: int, index: AtomIndex) -> list[int]:
    result = [node_id]
    for child_id in index.children.get(node_id, []):
        result.extend(get_subtree(child_id, index))
    return result
