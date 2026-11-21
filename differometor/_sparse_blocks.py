"""Separate fixed-unit equations from an optical system's static CSC pattern."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class _SingletonLayer:
    coupling_slots: np.ndarray
    sources: np.ndarray
    targets: np.ndarray


def _scale_for_rhs(values, rhs):
    return values[..., None] if rhs.ndim == 3 else values


@dataclass(frozen=True, slots=True)
class _OpticalLinks:
    """Eliminate independent unit-diagonal rows with one incoming coefficient.

    Since the eliminated diagonal block is identity, the Schur correction is
    just the product of its outgoing and incoming couplings. The same stored
    coefficients recover the original solution for normal and adjoint solves.
    """

    variables: np.ndarray
    predecessors: np.ndarray
    incoming: np.ndarray
    targets: np.ndarray
    outgoing_variables: np.ndarray
    outgoing: np.ndarray
    raw_outgoing: np.ndarray
    incoming_for_edge: np.ndarray
    base_slots: np.ndarray
    base_destinations: np.ndarray
    corrections: np.ndarray

    def condense(self, values, matrix):
        matrix.fill(0)
        matrix[:, self.base_destinations] = values[:, self.base_slots]
        correction = -values[:, self.raw_outgoing] * values[:, self.incoming_for_edge]
        np.add.at(matrix, (slice(None), self.corrections), correction)

    def prepare(self, values, result, trans):
        if trans == "N":
            source, target, slots = self.outgoing_variables, self.targets, self.outgoing
        else:
            source, target, slots = self.variables, self.predecessors, self.incoming
        coefficients = values[:, slots]
        if trans == "H":
            coefficients = coefficients.conj()
        updates = -_scale_for_rhs(coefficients, result) * result[:, source]
        np.add.at(result, (slice(None), target), updates)

    def restore(self, values, result, trans):
        if trans == "N":
            source, target, slots = self.predecessors, self.variables, self.incoming
        else:
            source, target, slots = self.targets, self.outgoing_variables, self.outgoing
        coefficients = values[:, slots]
        if trans == "H":
            coefficients = coefficients.conj()
        updates = -_scale_for_rhs(coefficients, result) * result[:, source]
        np.add.at(result, (slice(None), target), updates)


@dataclass(frozen=True, slots=True)
class SignalBlockPlan:
    """Order singleton substitutions around the remaining coupled block."""

    leading: tuple[_SingletonLayer, ...]
    trailing: tuple[_SingletonLayer, ...]
    core: np.ndarray
    rows: np.ndarray
    indptr: np.ndarray
    base_slots: np.ndarray
    coupling_slots: np.ndarray
    links: _OpticalLinks | None

    @property
    def buffer_size(self):
        return self.rows.size + self.coupling_slots.size

    def pack(self, values):
        numeric = np.empty((values.shape[0], self.buffer_size), dtype=values.dtype)
        core_matrix = numeric[:, :self.rows.size]
        if self.links is None:
            core_matrix[:] = values[:, self.base_slots]
        else:
            self.links.condense(values, core_matrix)
        numeric[:, self.rows.size:] = values[:, self.coupling_slots]
        return numeric

    def prepare(self, values, rhs, trans):
        result = rhs.copy()
        layers = self.leading if trans == "N" else self.trailing
        for layer in layers:
            coupling = values[:, layer.coupling_slots]
            if trans == "H":
                coupling = coupling.conj()
            updates = -_scale_for_rhs(coupling, rhs) * result[:, layer.sources]
            np.add.at(result, (slice(None), layer.targets), updates)
        if self.links is not None:
            self.links.prepare(values, result, trans)
        return result

    def restore(self, values, result, trans):
        if self.links is not None:
            self.links.restore(values, result, trans)
        layers = self.trailing if trans == "N" else self.leading
        for layer in reversed(layers):
            coupling = values[:, layer.coupling_slots]
            if trans == "H":
                coupling = coupling.conj()
            updates = -_scale_for_rhs(coupling, result) * result[:, layer.targets]
            np.add.at(result, (slice(None), layer.sources), updates)


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
                return tuple(layers)
            coupling = edges & peelable[source]
            layers.append(
                _SingletonLayer(np.flatnonzero(coupling), source[coupling], target[coupling])
            )
            active[direct] = False

    leading = peel(cols, rows)
    trailing = peel(rows, cols)
    links, core, factored_rows, pointers, base_slots = _plan_optical_links(
        size, rows, cols, diagonal, active, unit_diagonal
    )
    if not (leading or trailing or links is not None):
        return None

    used = [layer.coupling_slots for layer in (*leading, *trailing)]
    if links is not None:
        used.extend((links.incoming, links.outgoing))
    couplings = np.unique(np.concatenate(used))
    packed = np.empty(rows.size, dtype=np.intp)
    packed[couplings] = factored_rows.size + np.arange(couplings.size)

    def packed_layers(layers):
        return tuple(replace(layer, coupling_slots=packed[layer.coupling_slots]) for layer in layers)

    if links is not None:
        links = replace(links, incoming=packed[links.incoming], outgoing=packed[links.outgoing])
    return SignalBlockPlan(
        packed_layers(leading), packed_layers(trailing), core, factored_rows, pointers, base_slots, couplings, links
    )


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
    link_plan = None
    keys = base_keys
    if np.any(eliminate):
        variables = np.flatnonzero(eliminate)
        edge_slots = off_diagonal[eliminate[cols[off_diagonal]]]
        target = rows[edge_slots]
        source = cols[edge_slots]
        fill_keys = remap[predecessor[source]] * np.int64(core.size) + remap[target]
        keys = np.union1d(base_keys, fill_keys)
        link_plan = _OpticalLinks(
            variables, predecessor[variables], incoming[variables],
            target, source, edge_slots, edge_slots, incoming[source], base_slots,
            np.searchsorted(keys, base_keys), np.searchsorted(keys, fill_keys),
        )

    factored_rows = (keys % core.size).astype(np.int32) if core.size else np.empty(0, dtype=np.int32)
    counts = np.bincount(keys // core.size, minlength=core.size) if core.size else np.empty(0, dtype=np.int32)
    pointers = np.empty(core.size + 1, dtype=np.int32)
    pointers[0] = 0
    np.cumsum(counts, out=pointers[1:])
    return link_plan, core, factored_rows, pointers, base_slots
