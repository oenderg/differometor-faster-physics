"""Separate fixed-unit equations from an optical system's static CSC pattern."""

import numpy as np


def build_scatter_plan(indices: np.ndarray):
    """Group indices by per-target occurrence rank (exact np.add.at order).

    Returns a tuple of ``(src_positions_or_None, dst_targets)`` pairs with
    ``np.intp`` index arrays. When every target is unique, returns
    ``((None, targets),)`` so callers can add without slicing updates.
    """
    idx = np.asarray(indices, dtype=np.intp)
    if idx.size == 0:
        return ()
    if idx.size == 1 or (idx[1:] > idx[:-1]).all():
        return ((None, idx),)
    order = np.argsort(idx, kind="stable")
    ordered = idx[order]
    first = np.empty(ordered.size, dtype=bool)
    first[0] = True
    first[1:] = ordered[1:] != ordered[:-1]
    if first.all():
        return ((None, idx),)
    starts = np.maximum.accumulate(np.where(first, np.arange(ordered.size), 0))
    rank = np.empty(idx.size, dtype=np.intp)
    rank[order] = np.arange(ordered.size) - starts
    max_rank = int(rank.max())
    batches = []
    for r in range(max_rank + 1):
        pos = np.flatnonzero(rank == r)
        batches.append((pos, idx[pos]))
    return tuple(batches)


def scatter_add_axis1(dst: np.ndarray, plan: tuple, updates: np.ndarray) -> None:
    """Scatter-add ``updates`` into ``dst`` along axis 1 in exact ufunc.at order."""
    if not plan:
        return
    if len(plan) == 1 and plan[0][0] is None:
        dst[:, plan[0][1]] += updates
        return
    for pos, targets in plan:
        dst[:, targets] += updates[:, pos]


def scatter_add_1d(dst: np.ndarray, plan: tuple, updates: np.ndarray) -> None:
    """Scatter-add 1-D ``updates`` into 1-D ``dst`` in exact ufunc.at order."""
    if not plan:
        return
    if len(plan) == 1 and plan[0][0] is None:
        dst[plan[0][1]] += updates
        return
    for pos, targets in plan:
        dst[targets] += updates[pos]


def scatter_add_axis0_T(dst: np.ndarray, plan: tuple, updates: np.ndarray) -> None:
    """Scatter-add ``updates.T`` (shape ``(M, B)``) into ``dst`` along axis 0."""
    if not plan:
        return
    if len(plan) == 1 and plan[0][0] is None:
        dst[plan[0][1]] += updates.T
        return
    for pos, targets in plan:
        dst[targets] += updates[:, pos].T


class _SingletonLayer:
    __slots__ = ("coupling_slice", "sources", "targets", "target_plan", "source_plan")

    def __init__(self, coupling_slice, sources, targets):
        self.coupling_slice = coupling_slice
        self.sources = np.asarray(sources, dtype=np.intp)
        self.targets = np.asarray(targets, dtype=np.intp)
        self.target_plan = build_scatter_plan(self.targets)
        self.source_plan = build_scatter_plan(self.sources)


def _apply_step(couplings, sl, source, target_plan, buf):
    if source.size == 0:
        return
    coeff = couplings[:, sl]
    updates = buf[:, source]
    if buf.ndim == 2:
        np.multiply(coeff, updates, out=updates)
    else:
        np.multiply(coeff[:, :, None], updates, out=updates)
    scatter_add_axis1(buf, target_plan, updates)


class _OpticalLinks:
    """Eliminate independent unit-diagonal rows with one incoming coefficient.

    Since the eliminated diagonal block is identity, the Schur correction is
    just the product of its outgoing and incoming couplings. The same stored
    coefficients recover the original solution for normal and adjoint solves.
    """

    __slots__ = (
        "variables",
        "predecessors",
        "incoming_slice",
        "targets",
        "outgoing_variables",
        "outgoing_slice",
        "raw_outgoing",
        "incoming_for_edge",
        "base_slots",
        "base_destinations",
        "fill_only_destinations",
        "correction_plan",
        "target_plan",
        "predecessor_plan",
        "variable_plan",
        "outgoing_variable_plan",
    )

    def __init__(
        self,
        variables,
        predecessors,
        incoming_slice,
        targets,
        outgoing_variables,
        outgoing_slice,
        raw_outgoing,
        incoming_for_edge,
        base_slots,
        base_destinations,
        corrections,
    ):
        self.variables = np.asarray(variables, dtype=np.intp)
        self.predecessors = np.asarray(predecessors, dtype=np.intp)
        self.incoming_slice = incoming_slice
        self.targets = np.asarray(targets, dtype=np.intp)
        self.outgoing_variables = np.asarray(outgoing_variables, dtype=np.intp)
        self.outgoing_slice = outgoing_slice
        self.raw_outgoing = np.asarray(raw_outgoing, dtype=np.intp)
        self.incoming_for_edge = np.asarray(incoming_for_edge, dtype=np.intp)
        self.base_slots = np.asarray(base_slots, dtype=np.intp)
        self.base_destinations = np.asarray(base_destinations, dtype=np.intp)
        corr_idx = np.asarray(corrections, dtype=np.intp)
        self.fill_only_destinations = np.setdiff1d(corr_idx, self.base_destinations).astype(np.intp)
        self.correction_plan = build_scatter_plan(corr_idx)
        self.target_plan = build_scatter_plan(self.targets)
        self.predecessor_plan = build_scatter_plan(self.predecessors)
        self.variable_plan = build_scatter_plan(self.variables)
        self.outgoing_variable_plan = build_scatter_plan(self.outgoing_variables)

    def condense(self, values, matrix):
        if self.fill_only_destinations.size:
            matrix[:, self.fill_only_destinations] = 0
        matrix[:, self.base_destinations] = values[:, self.base_slots]
        if self.raw_outgoing.size:
            correction = values[:, self.raw_outgoing]
            np.negative(correction, out=correction)
            correction *= values[:, self.incoming_for_edge]
            scatter_add_axis1(matrix, self.correction_plan, correction)

    def prepare(self, couplings, buf, trans):
        if trans == "N":
            _apply_step(couplings, self.outgoing_slice, self.outgoing_variables, self.target_plan, buf)
        else:
            _apply_step(couplings, self.incoming_slice, self.variables, self.predecessor_plan, buf)

    def restore(self, couplings, buf, trans):
        if trans == "N":
            _apply_step(couplings, self.incoming_slice, self.predecessors, self.variable_plan, buf)
        else:
            _apply_step(couplings, self.outgoing_slice, self.targets, self.outgoing_variable_plan, buf)


class SignalBlockPlan:
    """Order singleton substitutions around the remaining coupled block."""

    __slots__ = (
        "leading",
        "trailing",
        "core",
        "rows",
        "indptr",
        "base_slots",
        "coupling_slots",
        "links",
        "buffer_size",
        "_tail_offset",
        "_split_offset",
    )

    def __init__(self, leading, trailing, core, rows, indptr, base_slots, coupling_slots, links, split_offset):
        self.leading = leading
        self.trailing = trailing
        self.core = np.asarray(core, dtype=np.intp)
        self.rows = rows
        self.indptr = indptr
        self.base_slots = np.asarray(base_slots, dtype=np.intp)
        self.coupling_slots = np.asarray(coupling_slots, dtype=np.intp)
        self.links = links
        self._tail_offset = int(rows.size)
        self._split_offset = int(split_offset)
        self.buffer_size = self._tail_offset + int(self.coupling_slots.size)

    def pack(self, values):
        numeric = np.empty((values.shape[0], self.buffer_size), dtype=values.dtype)
        tail_offset = self._tail_offset
        core_matrix = numeric[:, :tail_offset]
        if self.links is None:
            core_matrix[:] = values[:, self.base_slots]
        else:
            self.links.condense(values, core_matrix)
        # Pre-negate packed couplings once; every prepare/restore step multiplies by -coupling.
        np.negative(values[:, self.coupling_slots], out=numeric[:, tail_offset:])
        return numeric

    def prepare(self, values, rhs, trans, copy=True):
        result = rhs.copy() if copy else rhs
        buf = result[:, :, 0] if (result.ndim == 3 and result.shape[2] == 1) else result
        if trans == "N":
            couplings = values[:, self._tail_offset : self._tail_offset + self._split_offset]
            for layer in self.leading:
                _apply_step(couplings, layer.coupling_slice, layer.sources, layer.target_plan, buf)
            if self.links is not None:
                self.links.prepare(couplings, buf, trans)
        else:
            half = values[:, self._tail_offset + self._split_offset :]
            couplings = half.conj() if trans == "H" else half
            for layer in self.trailing:
                _apply_step(couplings, layer.coupling_slice, layer.sources, layer.target_plan, buf)
            if self.links is not None:
                self.links.prepare(couplings, buf, trans)
        return result

    def restore(self, values, result, trans):
        buf = result[:, :, 0] if (result.ndim == 3 and result.shape[2] == 1) else result
        if trans == "N":
            couplings = values[:, self._tail_offset + self._split_offset :]
            if self.links is not None:
                self.links.restore(couplings, buf, trans)
            for layer in reversed(self.trailing):
                _apply_step(couplings, layer.coupling_slice, layer.targets, layer.source_plan, buf)
        else:
            half = values[:, self._tail_offset : self._tail_offset + self._split_offset]
            couplings = half.conj() if trans == "H" else half
            if self.links is not None:
                self.links.restore(couplings, buf, trans)
            for layer in reversed(self.leading):
                _apply_step(couplings, layer.coupling_slice, layer.targets, layer.source_plan, buf)


def plan_signal_blocks(size, rows, cols, *, unit_slots):
    """Peel source and sink singletons, keeping all cycles in the coupled block."""
    diagonal = rows == cols
    diagonal_slots = np.full(size, -1, dtype=np.intp)
    diagonal_slots[rows[diagonal]] = np.flatnonzero(diagonal)
    unit_diagonal = diagonal_slots >= 0
    unit_diagonal[unit_diagonal] = unit_slots[diagonal_slots[unit_diagonal]]
    active = np.ones(size, dtype=bool)

    def peel(source, target):
        layers = []
        while True:
            edges = active[source] & active[target] & ~diagonal
            incoming = np.bincount(target[edges], minlength=size)
            peelable = active & unit_diagonal & (incoming == 0)
            direct = np.flatnonzero(peelable)
            if direct.size == 0:
                return layers
            coupling = edges & peelable[source]
            layers.append((np.flatnonzero(coupling), source[coupling], target[coupling]))
            active[direct] = False

    raw_leading = peel(cols, rows)
    raw_trailing = peel(rows, cols)
    raw_links, core, factored_rows, pointers, base_slots = _plan_optical_links(
        size, rows, cols, diagonal, active, unit_diagonal
    )
    if not (raw_leading or raw_trailing or raw_links is not None):
        return None

    # Pack into two disjoint halves:
    #   Half A (offset 0..split): leading layers + links.outgoing  (used by prepare("N") and restore("T"/"H"))
    #   Half B (offset 0..end-split relative): trailing layers + links.incoming (used by prepare("T"/"H") and restore("N"))
    half_a_parts = []
    off_a = 0
    leading_list = []
    for slots, src, dst in raw_leading:
        n = int(slots.size)
        half_a_parts.append(slots)
        leading_list.append(_SingletonLayer(slice(off_a, off_a + n), src, dst))
        off_a += n
    leading = tuple(leading_list)

    half_b_parts = []
    off_b = 0
    trailing_list = []
    for slots, src, dst in raw_trailing:
        n = int(slots.size)
        half_b_parts.append(slots)
        trailing_list.append(_SingletonLayer(slice(off_b, off_b + n), src, dst))
        off_b += n
    trailing = tuple(trailing_list)

    links = None
    if raw_links is not None:
        (
            variables, predecessors, incoming_slots,
            target, source, edge_slots,
            incoming_for_edge, link_base_slots, base_dest, fill_dest,
        ) = raw_links
        n_out = int(edge_slots.size)
        out_slice = slice(off_a, off_a + n_out)
        half_a_parts.append(edge_slots)
        off_a += n_out

        n_in = int(incoming_slots.size)
        in_slice = slice(off_b, off_b + n_in)
        half_b_parts.append(incoming_slots)
        off_b += n_in

        links = _OpticalLinks(
            variables, predecessors, in_slice,
            target, source, out_slice,
            edge_slots, incoming_for_edge, link_base_slots, base_dest, fill_dest,
        )

    all_parts = half_a_parts + half_b_parts
    couplings = np.concatenate(all_parts).astype(np.intp, copy=False) if all_parts else np.empty(0, dtype=np.intp)
    return SignalBlockPlan(leading, trailing, core, factored_rows, pointers, base_slots, couplings, links, off_a)


def _plan_optical_links(size, rows, cols, diagonal, active, unit_diagonal):
    in_block = active[rows] & active[cols]
    off_diagonal = np.flatnonzero(in_block & ~diagonal)
    indegree = np.bincount(rows[off_diagonal], minlength=size)
    single_edges = off_diagonal[indegree[rows[off_diagonal]] == 1]
    predecessor = np.full(size, -1, dtype=np.intp)
    predecessor[rows[single_edges]] = cols[single_edges]
    incoming = np.full(size, -1, dtype=np.intp)
    incoming[rows[single_edges]] = single_edges

    blocked = np.zeros(size, dtype=bool)
    # A two-edge feedback loop would change a core diagonal. Keep that loop in LU.
    return_edges = rows[off_diagonal] == predecessor[cols[off_diagonal]]
    blocked[cols[off_diagonal[return_edges]]] = True
    candidates = np.flatnonzero(active & (indegree == 1) & unit_diagonal)
    eliminate = np.zeros(size, dtype=bool)
    for variable in candidates:
        parent = predecessor[variable]
        if not blocked[variable] and not eliminate[parent]:
            eliminate[variable] = True
            blocked[parent] = True

    kept = active & ~eliminate
    core = np.flatnonzero(kept)
    remap = np.cumsum(kept, dtype=np.int32) - 1
    base_slots = np.flatnonzero(kept[rows] & kept[cols])
    base_keys = remap[cols[base_slots]] * np.int64(core.size) + remap[rows[base_slots]]
    raw_links = None
    keys = base_keys
    if np.any(eliminate):
        variables = np.flatnonzero(eliminate)
        edge_slots = off_diagonal[eliminate[cols[off_diagonal]]]
        target = rows[edge_slots]
        source = cols[edge_slots]
        fill_keys = remap[predecessor[source]] * np.int64(core.size) + remap[target]
        keys = np.union1d(base_keys, fill_keys)
        raw_links = (
            variables, predecessor[variables], incoming[variables],
            target, source, edge_slots,
            incoming[source], base_slots,
            np.searchsorted(keys, base_keys), np.searchsorted(keys, fill_keys),
        )

    factored_rows = (keys % core.size).astype(np.int32) if core.size else np.empty(0, dtype=np.int32)
    counts = np.bincount(keys // core.size, minlength=core.size) if core.size else np.empty(0, dtype=np.int32)
    pointers = np.empty(core.size + 1, dtype=np.int32)
    pointers[0] = 0
    np.cumsum(counts, out=pointers[1:])
    return raw_links, core, factored_rows, pointers, base_slots
