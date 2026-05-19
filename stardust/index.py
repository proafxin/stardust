from bisect import bisect_right

from stardust.tree.atom import SpanOffset


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
