"""
differometor.builder
====================

This module turns a `differometor.setups.Setup` graph into:

1. A *carrier* linear system matrix (complex-valued).
2. A *signal* linear system matrix that contains two sidebands plus explicit signal degrees of freedom.
3. A *noise* matrix and selection vectors for quantum noise readout.
4. A set of *instructions* that describe how to update matrix entries when parameters change.

The central idea is:

- Build sparse update instructions that map from a parameter vector to specific matrix entries.
- Keep indexing conventions consistent across carrier, signal, and noise representations.
- Allow optional parameter linking, where some parameters are computed from others by user-provided functions.

Indexing conventions
--------------------

Carrier system:
- `carrier_matrix` has shape `(system_size, system_size + 1)`.
  The last column is the right-hand side.

Signal system:
- `signal_matrix` has shape `(system_size * 2 + signal_size, system_size * 2 + signal_size + 1)`.
  The last column is the right-hand side.
- The first `system_size` block is the upper sideband.
- The next `system_size` block is the lower sideband.
- The final `signal_size` rows and columns represent explicit signal inputs.

Noise system:
- `noise_matrix` matches `signal_matrix[:, :-1]` shape.
- `noise_selection_vectors` encodes which fields are read out by noise detectors.

The code below is intentionally explicit about shapes and index transforms, because silent off-by-one
or block-shift errors are difficult to debug in linearized optical simulations.
"""

import numpy as np
import jax.numpy as jnp
from differometor.setups import Setup
from collections import defaultdict
from differometor.components import (
    C_LIGHT,
    F,
    DEFAULT_REFRACTIVE_INDEX,
    DEFAULT_PROPERTIES,
    FUNCTIONS,
    signal_function,
    laser_np,
    laser,
    vacuum_quantum_noise,
    squeezer,
    nothing_matrix,
    directional_beamsplitter_matrix,
    beamsplitter_matrix,
    mirror_matrix,
    _reflectivity_left_np,
    _reflectivity_right_np,
    _transmissivity_np,
    space_modulation,
    space_lower,
    space_modulation_lower,
    space,
    laser_amplitude_modulation,
    laser_frequency_modulation,
    laser_frequency_modulation_lower,
    susceptibility,
    force_calculation_left,
    force_calculation_right,
    surface,
    loss_quantum_noise,
    corrected_optomechanical_phase_left,
    corrected_optomechanical_phase_right,
    dummy_function,
)

# ---------------------------------------------------------------------
# Global state for parameter linking
# ---------------------------------------------------------------------

LINKING_FUNCTIONS = []
"""
Global registry of parameter-linking functions.

A linking function is a callable that is evaluated externally to compute the current value
of a "linked" parameter. The builder stores indices into this list, so downstream code can
apply linking without needing to serialize callables.

Notes
-----
- This is module-level state. If you build multiple setups in the same process and rely on
  strict reproducibility of indices, you should reset this list before each build.
"""


_IDX_FORCE_RIGHT = FUNCTIONS.index(force_calculation_right)
_IDX_LASER = FUNCTIONS.index(laser)
_IDX_SURFACE = FUNCTIONS.index(surface)
_IDX_SPACE = FUNCTIONS.index(space)
_IDX_SPACE_MOD = FUNCTIONS.index(space_modulation)
_IDX_SIGNAL = FUNCTIONS.index(signal_function)
_IDX_DUMMY = FUNCTIONS.index(dummy_function)
_IDX_VACUUM_NOISE = FUNCTIONS.index(vacuum_quantum_noise)
_IDX_LOSS_NOISE = FUNCTIONS.index(loss_quantum_noise)
_IDX_SPACE_LOWER = FUNCTIONS.index(space_lower)
_IDX_LASER_AMP_MOD = FUNCTIONS.index(laser_amplitude_modulation)
_IDX_LASER_FREQ_MOD = FUNCTIONS.index(laser_frequency_modulation)
_IDX_SUSCEPTIBILITY = FUNCTIONS.index(susceptibility)
_IDX_FORCE_LEFT = FUNCTIONS.index(force_calculation_left)
_IDX_OM_PHASE_LEFT = FUNCTIONS.index(corrected_optomechanical_phase_left)
_IDX_OM_PHASE_RIGHT = FUNCTIONS.index(corrected_optomechanical_phase_right)
_IDX_SPACE_MOD_LOWER = FUNCTIONS.index(space_modulation_lower)
_IDX_SQUEEZER = FUNCTIONS.index(squeezer)
_IDX_LASER_FREQ_MOD_LOWER = FUNCTIONS.index(laser_frequency_modulation_lower)

_MIRROR_OUT_COL = np.array([0, 1, 1, 2], dtype=np.int32)
_MIRROR_ROW_OFF = np.array([1, 1, 3, 3], dtype=np.int32)
_MIRROR_COL_OFF = np.array([0, 2, 0, 2], dtype=np.int32)
_MIRROR_NOISE_OUT_COL = np.array([0, 0], dtype=np.int32)
_MIRROR_NOISE_OFF = np.array([1, 3], dtype=np.int32)

_BS_OUT_COL = np.array([0, 1, 0, 1, 1, 2, 1, 2], dtype=np.int32)
_BS_ROW_OFF = np.array([1, 1, 3, 3, 5, 5, 7, 7], dtype=np.int32)
_BS_COL_OFF = np.array([2, 4, 0, 6, 0, 6, 2, 4], dtype=np.int32)
_BS_NOISE_OUT_COL = np.array([0, 0, 0, 0], dtype=np.int32)
_BS_NOISE_OFF = np.array([1, 3, 5, 7], dtype=np.int32)

_MIRROR_SIG_OUT_COL = np.concatenate((_MIRROR_OUT_COL, _MIRROR_OUT_COL))
_MIRROR_SIG_NOISE_OUT_COL = np.concatenate((_MIRROR_NOISE_OUT_COL, _MIRROR_NOISE_OUT_COL))
_BS_SIG_OUT_COL = np.concatenate((_BS_OUT_COL, _BS_OUT_COL))
_BS_SIG_NOISE_OUT_COL = np.concatenate((_BS_NOISE_OUT_COL, _BS_NOISE_OUT_COL))
_SURFACE_FUNC_LIST = [_IDX_SURFACE]
_LOSS_NOISE_FUNC_LIST = [_IDX_LOSS_NOISE]
_VACUUM_INP_PADDED = np.zeros((1, 7), dtype=np.int32)
_VACUUM_OUT_COL_2 = np.zeros(2, dtype=np.int32)
_VACUUM_FUNC_LIST = [_IDX_VACUUM_NOISE]
_SIG_OUT_COL_1 = np.zeros(1, dtype=np.int32)
_SIG_COL_M1 = np.array([-1], dtype=np.int32)
_SIG_FUNC_LIST = [_IDX_SIGNAL]
_LASER_FUNC_LIST = [_IDX_LASER]
_SQUEEZER_OUT_COL = np.array([0, 1, 2, 3], dtype=np.int32)
_SQUEEZER_FUNC_LIST = [_IDX_SQUEEZER]
_SPACE_CAR_OUT_COL = np.zeros(2, dtype=np.int32)
_SPACE_CAR_FUNC_LIST = [_IDX_SPACE]
_SPACE_SIG_OUT_ROW = np.array([0, 0, 1, 1], dtype=np.int32)
_SPACE_SIG_OUT_COL = np.zeros(4, dtype=np.int32)
_SPACE_SIG_FUNC_LIST = [_IDX_SPACE, _IDX_SPACE_LOWER]
_SPACE_MOD_OUT_COL_1 = np.zeros(4, dtype=np.int32)
_SPACE_MOD_OUT_ROW_1 = np.array([0, 0, 1, 1], dtype=np.int32)
_SPACE_MOD_FUNC_LIST_1 = [_IDX_SPACE_MOD, _IDX_SPACE_MOD_LOWER]
_FM_FUNC_LIST = [
    _IDX_SUSCEPTIBILITY,
    _IDX_FORCE_LEFT,
    _IDX_FORCE_RIGHT,
    _IDX_OM_PHASE_LEFT,
    _IDX_OM_PHASE_RIGHT,
]
_FM_MIRROR_OUT_ROW = np.array([0, 1, 1, 2, 2, 3, 4, 3, 4, 1, 1, 2, 2], dtype=np.int32)
_FM_MIRROR_OUT_COL = np.zeros(13, dtype=np.int32)
_FM_MIRROR_CAR_OFF = np.array([0, 1, 2, 3, 0, 2, 0, 2, 0, 1, 2, 3], dtype=np.int32)
_FM_BS_OUT_ROW = np.array(
    [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2],
    dtype=np.int32,
)
_FM_BS_OUT_COL = np.zeros(25, dtype=np.int32)
_FM_BS_CAR_OFF = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 2, 0, 6, 4, 2, 0, 6, 4, 0, 1, 2, 3, 4, 5, 6, 7],
    dtype=np.int32,
)
_NOTHING_ROW_OFF = np.array([1, 3], dtype=np.int32)
_NOTHING_COL_OFF = np.array([2, 0], dtype=np.int32)
_NOTHING_VALS = np.array([-1.0, -1.0], dtype=np.complex128)
_DBS_ROW_OFF = np.array([1, 3, 5, 7], dtype=np.int32)
_DBS_COL_OFF = np.array([2, 6, 0, 4], dtype=np.int32)
_DBS_VALS = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.complex128)


def _expand_sidebands(rows: np.ndarray, cols: np.ndarray, s: int, extra1: np.ndarray, extra2: np.ndarray | None = None):
    """Expand (rows, cols) and aligned extra arrays across upper/lower sidebands."""
    r_lt = rows < s
    c_lt = cols < s
    if r_lt.all() and c_lt.all():
        new_rows = np.concatenate((rows, rows + s))
        new_cols = np.concatenate((cols, cols + s))
        new_e1 = np.concatenate((extra1, extra1))
        new_e2 = np.concatenate((extra2, extra2)) if extra2 is not None else None
        return new_rows, new_cols, new_e1, new_e2

    base_r = np.where(r_lt, rows, rows + s)
    base_c = np.where(c_lt, cols, cols + s)

    only_r = r_lt & ~c_lt
    only_c = ~r_lt & c_lt
    both = r_lt & c_lt

    r_parts = [base_r]
    c_parts = [base_c]
    e1_parts = [extra1]
    e2_parts = [extra2] if extra2 is not None else None

    if only_r.any():
        r_parts.append(base_r[only_r] + s)
        c_parts.append(base_c[only_r])
        e1_parts.append(extra1[only_r])
        if e2_parts is not None:
            e2_parts.append(extra2[only_r])

    if only_c.any():
        r_parts.append(base_r[only_c])
        c_parts.append(base_c[only_c] + s)
        e1_parts.append(extra1[only_c])
        if e2_parts is not None:
            e2_parts.append(extra2[only_c])

    if both.any():
        r_parts.append(base_r[both] + s)
        c_parts.append(base_c[both] + s)
        e1_parts.append(extra1[both])
        if e2_parts is not None:
            e2_parts.append(extra2[both])

    if len(r_parts) == 1:
        return base_r, base_c, extra1, extra2
    return (
        np.concatenate(r_parts),
        np.concatenate(c_parts),
        np.concatenate(e1_parts),
        np.concatenate(e2_parts) if e2_parts is not None else None,
    )


def set_instructions(
    instructions: dict,
    node: str,
    function_input_indices: np.ndarray,
    output_column_indices: np.ndarray,
    system_matrix_row_indices: np.ndarray,
    system_matrix_column_indices: np.ndarray,
    function_indices: list,
    output_row_indices: np.ndarray | None = None,
    carrier_indices: np.ndarray | None = None,
    carrier_row_indices: np.ndarray | None = None,
    carrier_column_indices: np.ndarray | None = None,
    system_size_for_sidebands: int | None = None,
) -> None:
    """Store standardized update instructions for one logical component."""
    n_rows, n_cols = function_input_indices.shape
    if n_cols < 7:
        padded = np.zeros((n_rows, 7), dtype=np.int32)
        padded[:, :n_cols] = function_input_indices
        function_input_indices = padded
    elif n_cols != 7:
        raise AssertionError("Function input indices have to be standardized to length 7.")

    if len(output_column_indices) != len(system_matrix_row_indices):
        raise AssertionError("Output column indices have to match the length of system matrix row indices.")

    if system_size_for_sidebands is not None:
        (
            system_matrix_row_indices,
            system_matrix_column_indices,
            output_column_indices,
            output_row_indices,
        ) = _expand_sidebands(
            system_matrix_row_indices,
            system_matrix_column_indices,
            system_size_for_sidebands,
            output_column_indices,
            output_row_indices,
        )
        if carrier_indices is not None:
            (
                carrier_row_indices,
                carrier_column_indices,
                carrier_indices,
                _,
            ) = _expand_sidebands(
                carrier_row_indices,
                carrier_column_indices,
                system_size_for_sidebands,
                carrier_indices,
                None,
            )

    entry = {
        "function_input_indices": function_input_indices,
        "output_column_indices": output_column_indices,
        "system_matrix_row_indices": system_matrix_row_indices,
        "system_matrix_column_indices": system_matrix_column_indices,
        "function_indices": function_indices,
    }
    if output_row_indices is not None:
        entry["output_row_indices"] = output_row_indices
    if carrier_indices is not None:
        entry["carrier_indices"] = carrier_indices
        entry["carrier_row_indices"] = carrier_row_indices
        entry["carrier_column_indices"] = carrier_column_indices
    instructions[node] = entry


def parameter_linking(
    new_parameters: list,
    indices_to_link: list,
    linked_names: list,
    linking_function_indices: list,
    parameters: list,
) -> None:
    """
    Append parameters to the parameter list while supporting "linked" parameters.

    A "linked" parameter is represented by a tuple `(name, function)` where:
    - `name` is stored in `linked_names` and later converted to an index in `parameter_names`.
    - `function` is appended to `LINKING_FUNCTIONS` and referenced by an integer index.

    Parameters
    ----------
    new_parameters:
        List whose items are either:
        - A numeric value to append directly, or
        - A tuple `(str, callable)` describing a linked parameter.
    indices_to_link:
        List updated in-place with indices in `parameters` that correspond to linked parameters.
    linked_names:
        List updated in-place with the names of linked parameters.
    linking_function_indices:
        List updated in-place with integer indices into `LINKING_FUNCTIONS`.
    parameters:
        Parameter list updated in-place. Linked parameters are stored as placeholder zeros.

    Raises
    ------
    AssertionError
        If a linked parameter tuple is malformed.
    """
    for new_parameter in new_parameters:
        if type(new_parameter) is not tuple:
            parameters.append(new_parameter)
        else:
            assert type(new_parameter[0]) is str, "Linked parameter name has to be a string."
            assert callable(new_parameter[1]), "Linked parameter value has to be callable."

            # Store a placeholder; the real value is computed via the linking function later.
            parameters.append(0)

            # Register the linking function and store the index mapping.
            LINKING_FUNCTIONS.append(new_parameter[1])
            linking_function_indices.append(len(LINKING_FUNCTIONS) - 1)
            indices_to_link.append(len(parameters) - 1)
            linked_names.append(new_parameter[0])


def build(setup: Setup, as_jax: bool = True, _sparse_signal: bool = False):
    """
    Build carrier, signal, and noise matrices plus update instructions from a setup graph.

    Parameters
    ----------
    setup:
        A `Setup` graph that contains:
        - Nodes with at least a `"component"` key.
        - Edges representing spaces (propagation paths) with optional `"properties"` dict.

    Returns
    -------
    instructions_tuple:
        `(carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names)`
        where each instruction dictionary maps a component key to a standardized instruction set.
    matrices_tuple:
        `(parameters, carrier_matrix, signal_matrix, noise_matrix, noise_selection_vectors,
          qhd_parameter_indices, qhd_placing_indices, linked_indices, linking_function_indices, indices_to_link)`
        as `jax.numpy` arrays.
    metadata_tuple:
        `(detector_indices, mirror_indices, beamsplitter_indices, isolator_indices, port_to_index)`
        where the first four entries are integer arrays and `port_to_index` maps string keys of the form
        `"component.port.direction"` to carrier indices.

    Notes
    -----
    - `parameters` is shaped `(1, N)` to match later vectorized update logic.
    - If no explicit signal exists, a dummy signal degree of freedom is created to avoid conditional
      branches in later JAX computations.

    Raises
    ------
    ValueError
        If invalid graph connections are detected (for example detector as edge source).
    """
    system_size = 0
    signal_size = 0
    matrix_positions = {}
    carrier_instructions = {}
    signal_instructions = {}

    # Parameter vector starts with:
    # - Carrier frequency at index 0
    # - Default refractive index at index 1
    # - Default alpha value of 0 at index 2
    parameters = [F, DEFAULT_REFRACTIVE_INDEX, 0]
    parameter_names = ["", "", ""]

    signal_components = []
    space_to_signals = defaultdict(list)
    laser_to_signals = defaultdict(list)
    parameter_positions = {}
    noise_instructions = {}
    all_ports = set([])
    used_ports = set([])
    quantum_detector_number = 0
    free_masses = []
    signal_frequency_position = 0
    surfaces_to_refractive_index_parameter_position = defaultdict(dict)
    port_to_index = {}
    mirror_indices = []
    beamsplitter_indices = []
    isolator_indices = []
    indices_to_link = []
    linked_names = []
    linking_function_indices = []

    def load_defaults(data: dict, component: str | None = None) -> None:
        """
        Ensure `data["properties"]` exists and includes defaults for missing keys.

        Parameters
        ----------
        data:
            Node or edge attribute dictionary to update in-place.
        component:
            Component name to select defaults from. If not provided, `data["component"]` is used.
        """
        if component is None:
            component = data["component"]
        if "properties" not in data:
            data["properties"] = DEFAULT_PROPERTIES[component].copy()
        else:
            default_dict = DEFAULT_PROPERTIES[component].copy()
            default_dict.update(data["properties"])
            data["properties"] = default_dict

    # -----------------------------------------------------------------
    # First edge loop: determine system size contributions from spaces
    # and register space parameters early (needed by surfaces).
    # -----------------------------------------------------------------
    for (source, target, data) in setup.edges(data=True):
        load_defaults(data, "space")

        # Normalize missing port configuration for edges.
        if "source_port" not in data:
            data["source_port"] = "right"
        if "target_port" not in data:
            data["target_port"] = "left"

        source_node = setup.nodes[source]
        target_node = setup.nodes[target]

        # Add space parameters (length and refractive index).
        parameter_positions[f"{source}_{target}"] = len(parameters)
        parameter_linking(
            [data["properties"]["length"], data["properties"]["refractive_index"]],
            indices_to_link,
            linked_names,
            linking_function_indices,
            parameters,
        )
        parameter_names += [f"{source}_{target}_length", f"{source}_{target}_refractive_index"]

        # Track refractive index parameter positions at surfaces.
        if source_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[source][data["source_port"]] = len(parameters) - 1
        if target_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[target][data["target_port"]] = len(parameters) - 1

        # Beamsplitter refractive index constraints:
        # left and top share one index, right and bottom share the other.
        port_mapping = {"right": "right", "left": "left", "top": "left", "bottom": "right"}

        if source_node["component"] == "beamsplitter":
            port = port_mapping[data["source_port"]]
            if (
                port in surfaces_to_refractive_index_parameter_position[source]
                and parameters[surfaces_to_refractive_index_parameter_position[source][port]] != parameters[-1]
            ):
                raise ValueError(
                    "Beamsplitter has to have the same refractive index on left and top and right and bottom respectively."
                )
            surfaces_to_refractive_index_parameter_position[source][port] = len(parameters) - 1

        if target_node["component"] == "beamsplitter":
            port = port_mapping[data["target_port"]]
            if (
                port in surfaces_to_refractive_index_parameter_position[target]
                and parameters[surfaces_to_refractive_index_parameter_position[target][port]] != parameters[-1]
            ):
                raise ValueError(
                    "Beamsplitter has to have the same refractive index on left and top and right and bottom respectively."
                )
            surfaces_to_refractive_index_parameter_position[target][port] = len(parameters) - 1

        # Space-connected lasers and detectors create explicit "space endpoint" nodes in the matrix.
        if source_node["component"] in ["laser", "squeezer"]:
            matrix_positions[f"{source}_{target}_source"] = system_size
            system_size += 2  # each space endpoint contributes an input and output index
            all_ports.add(f"{source}_{target}.left")
            source_node["target"] = f"{source}_{target}_source"
        elif source_node["component"] in ["detector", "qnoised"]:
            raise ValueError("Detectors can only be targets of edges.")

        if target_node["component"] in ["detector", "qnoised"]:
            matrix_positions[f"{source}_{target}_target"] = system_size
            system_size += 2
            all_ports.add(f"{source}_{target}.right")
            target_node["target"] = f"{source}_{target}_target"
            target_node["direction"] = "out"
        elif target_node["component"] in ["laser", "squeezer"]:
            raise ValueError("Lasers can only be sources of edges.")

    # -----------------------------------------------------------------
    # First node loop: place submatrices and count signal degrees of freedom.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        load_defaults(data)

        # Place matrix components on the diagonal.
        if data["component"] in MATRIX_SIZES:
            matrix_size = MATRIX_SIZES[data["component"]]
            matrix_positions[node] = system_size

            if data["component"] == "mirror":
                surface_indices = mirror_indices
            elif data["component"] == "beamsplitter":
                surface_indices = beamsplitter_indices
            elif data["component"] == "directional_beamsplitter":
                surface_indices = isolator_indices
            else:
                surface_indices = []

            surface_indices.extend(range(system_size, system_size + matrix_size))
            system_size += matrix_size

            # Track all potential ports to later mark unused ports as vacuum-noise inputs.
            if data["component"] in ["mirror"]:
                all_ports.add(node + ".left")
                all_ports.add(node + ".right")
            if data["component"] in ["beamsplitter"]:
                all_ports.add(node + ".top")
                all_ports.add(node + ".bottom")
                all_ports.add(node + ".left")
                all_ports.add(node + ".right")

        # Normalize defaults for sources and sinks.
        if data["component"] in ["laser", "squeezer", "detector", "qnoised"]:
            if "port" not in data:
                data["port"] = "left"
            if "direction" not in data:
                data["direction"] = "in"

        # Squeezers are treated as signal components because their parameters are varied in the signal run.
        if data["component"] == "squeezer":
            signal_components.append(node)

        # Quantum detector counting for noise readout vectors.
        if data["component"] in ["qnoised", "qhd"]:
            if ("auxiliary" not in data) or ("auxiliary" in data and not data["auxiliary"]):
                quantum_detector_number += 1

        # Explicit signal inputs (for example modulation sources).
        if data["component"] == "signal":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            try:
                target_component = setup.nodes[data["target"]]["component"]
                if target_component == "laser":
                    laser_to_signals[data["target"]].append(node)
            except KeyError:
                # Target is a space (spaces are edges, not nodes).
                space_to_signals[data["target"]].append(node)
            signal_size += 1

        # Free mass introduces two signal degrees of freedom (force and displacement).
        if data["component"] == "free_mass":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            signal_size += 2
            free_masses.append(node)

        # Signal frequency parameter (used for sideband computations).
        if data["component"] == "frequency":
            signal_components.append(node)
            signal_frequency_position = len(parameters)
            parameters.append(data["properties"]["frequency"])
            parameter_names.append(f"{node}_frequency")

    detectors = {}

    # Carrier matrix: identity on the left block, with a separate right-hand-side column.
    if _sparse_signal:
        carrier_matrix = None
        c_diag_idx = np.arange(system_size, dtype=np.int32)
        c_coo_r = [c_diag_idx]
        c_coo_c = [c_diag_idx]
        c_coo_v = [np.ones(system_size, dtype=np.complex128)]
        c_rhs = np.zeros(system_size, dtype=np.complex128)
    else:
        carrier_matrix = np.zeros((system_size, system_size + 1), dtype=complex)
        np.fill_diagonal(carrier_matrix, 1)

    # Ensure signal_size is at least 1 to avoid conditional logic later.
    if signal_size == 0:
        signal_size = 1

    # Signal matrix includes two carrier-sized blocks plus appended signal block, plus a right-hand-side column.
    signal_dim = system_size * 2 + signal_size
    if _sparse_signal:
        signal_matrix = signal_dim
    else:
        signal_matrix = np.zeros((signal_dim, signal_dim + 1), dtype=complex)
        np.fill_diagonal(signal_matrix, 1)

    # Noise matrix has the same shape as the signal matrix without the right-hand side.
    noise_matrix = np.zeros((signal_dim, signal_dim)) if as_jax else None
    noise_detectors = {}

    # Selection vectors map fields to detector readouts for quantum noise calculations.
    noise_selection_vectors = np.zeros((quantum_detector_number, 1, system_size * 2 + signal_size), dtype=complex)
    noise_detector_count = 0

    detector_indices = []

    # Quantum homodyne detection (QHD) needs per-detector phase rotation information.
    qhd_parameter_indices = []
    qhd_placing_indices = []

    # -----------------------------------------------------------------
    # Second node loop: resolve laser and detector indices (depends on target placement).
    # Also fill matrices and build update instructions.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        # -----------------------------
        # Signals
        # -----------------------------
        if data["component"] == "signal":
            signal_index = matrix_positions[node]

            # Add signal amplitude and phase parameters.
            parameter_linking(
                [data["properties"]["amplitude"], data["properties"]["phase"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_amplitude", f"{node}_phase"]

            # Signal acts as a right-hand-side injection into the explicit signal block.
            sig_inp = np.zeros((1, 7), dtype=np.int32)
            sig_inp[0, 0] = len(parameters) - 2
            sig_inp[0, 1] = len(parameters) - 1
            signal_instructions[node] = {
                "function_input_indices": sig_inp,
                "output_column_indices": _SIG_OUT_COL_1,
                "system_matrix_row_indices": np.array([system_size * 2 + signal_index], dtype=np.int32),
                "system_matrix_column_indices": _SIG_COL_M1,
                "function_indices": _SIG_FUNC_LIST,
            }

        # -----------------------------
        # Lasers, detectors, squeezers
        # -----------------------------
        if data["component"] in ["laser", "detector", "qnoised", "squeezer"]:
            # Locate the target's base index in the system matrix.
            try:
                target_index = matrix_positions[data["target"]]
            except KeyError:
                # The target is a synthetic space endpoint created in the edge loop.
                suffix = "_source" if data["port"] == "left" else "_target"
                target_index = matrix_positions[data["target"] + suffix]

            # Compute port offset within the target submatrix.
            try:
                port_offset = PORT_DICTS[setup.nodes[data["target"]]["component"]][data["port"]]
            except KeyError:
                # Space endpoints have a single implicit port.
                port_offset = 0

            if data["direction"] not in ["in", "out"]:
                raise ValueError("Direction has to be either in or out.")

            # Determine if target is a diagonal matrix component (mirror, beamsplitter, etc.).
            target_is_matrix_component = False
            try:
                target_is_matrix_component = setup.nodes[data["target"]]["component"] in MATRIX_SIZES
            except KeyError:
                pass

            # Direction ordering differs for space endpoints versus explicit components.
            if not target_is_matrix_component:
                direction_offset = 1 if data["direction"] == "in" else 0
            else:
                direction_offset = 0 if data["direction"] == "in" else 1

            component_index = target_index + port_offset + direction_offset
            matrix_positions[node] = component_index

            # Ordinary detectors.
            if data["component"] == "detector":
                sideband = data.get("sideband", "upper")
                if sideband == "lower":
                    component_index += system_size
                detectors[node] = component_index
                detector_indices.append(component_index)

            # Quantum-noise detectors contribute to selection vectors.
            if data["component"] == "qnoised":
                if ("auxiliary" not in data) or ("auxiliary" in data and not data["auxiliary"]):
                    noise_detectors[node] = noise_detector_count
                    noise_selection_vectors[noise_detector_count, 0, component_index] = np.sqrt(2)
                    noise_selection_vectors[noise_detector_count, 0, component_index + system_size] = np.sqrt(2)
                    noise_detector_count += 1

            # Lasers inject a carrier field into the right-hand side and also emit vacuum noise.
            if data["component"] == "laser":
                target_port = data["target"].replace("_source", "").replace("_target", "") + "." + data["port"]
                used_ports.add(target_port)

                laser_val = laser_np(**data["properties"])
                if _sparse_signal:
                    c_rhs[component_index] = laser_val
                else:
                    carrier_matrix[component_index, system_size] = laser_val

                parameter_linking(
                    [data["properties"]["power"], data["properties"]["phase"]],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_power", f"{node}_phase"]

                laser_inp = np.zeros((1, 7), dtype=np.int32)
                laser_inp[0, 0] = len(parameters) - 2
                laser_inp[0, 1] = len(parameters) - 1
                carrier_instructions[node] = {
                    "function_input_indices": laser_inp,
                    "output_column_indices": _SIG_OUT_COL_1,
                    "system_matrix_row_indices": np.array([component_index], dtype=np.int32),
                    "system_matrix_column_indices": _SIG_COL_M1,
                    "function_indices": _LASER_FUNC_LIST,
                }

                laser_noise_idx = np.array([component_index, component_index + system_size], dtype=np.int32)
                noise_instructions[node] = {
                    "function_input_indices": _VACUUM_INP_PADDED,
                    "output_column_indices": _VACUUM_OUT_COL_2,
                    "system_matrix_row_indices": laser_noise_idx,
                    "system_matrix_column_indices": laser_noise_idx,
                    "function_indices": _VACUUM_FUNC_LIST,
                }

            # Squeezers transform vacuum noise entering the attached port.
            if data["component"] == "squeezer":
                target_port = data["target"].replace("_source", "").replace("_target", "") + "." + data["port"]
                used_ports.add(target_port)

                parameter_linking(
                    [data["properties"]["db"], data["properties"]["angle"]],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_db", f"{node}_angle"]

                sq_inp = np.zeros((1, 7), dtype=np.int32)
                sq_inp[0, 0] = len(parameters) - 2
                sq_inp[0, 1] = len(parameters) - 1
                noise_instructions[node] = {
                    "function_input_indices": sq_inp,
                    "output_column_indices": _SQUEEZER_OUT_COL,
                    "system_matrix_row_indices": np.array(
                        [component_index, component_index, component_index + system_size, component_index + system_size],
                        dtype=np.int32,
                    ),
                    "system_matrix_column_indices": np.array(
                        [component_index, component_index + system_size, component_index + system_size, component_index],
                        dtype=np.int32,
                    ),
                    "function_indices": _SQUEEZER_FUNC_LIST,
                }

        # -----------------------------
        # Quantum homodyne detection (QHD)
        # -----------------------------
        if data["component"] == "qhd":
            parameter_linking(
                [data["properties"]["phase"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_phase"]

            detector1_index = matrix_positions[data["detector1"]]
            detector2_index = matrix_positions[data["detector2"]]

            qhd_parameter_indices.extend([len(parameters) - 1, len(parameters) - 1])
            qhd_placing_indices.extend(
                [[noise_detector_count, 0, detector2_index], [noise_detector_count, 0, detector2_index + system_size]]
            )

            noise_selection_vectors[noise_detector_count, 0, detector1_index] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector1_index + system_size] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index + system_size] = np.sqrt(2)
            noise_detector_count += 1

        # -----------------------------
        # Passive matrix components
        # -----------------------------
        if data["component"] == "nothing":
            nothing_index = matrix_positions[node]
            if _sparse_signal:
                c_coo_r.append(nothing_index + _NOTHING_ROW_OFF)
                c_coo_c.append(nothing_index + _NOTHING_COL_OFF)
                c_coo_v.append(_NOTHING_VALS)
            else:
                matrix = nothing_matrix()
                carrier_matrix[nothing_index : nothing_index + matrix.shape[0], nothing_index : nothing_index + matrix.shape[1]] = matrix

        if data["component"] == "directional_beamsplitter":
            directional_beamsplitter_index = matrix_positions[node]
            if _sparse_signal:
                c_coo_r.append(directional_beamsplitter_index + _DBS_ROW_OFF)
                c_coo_c.append(directional_beamsplitter_index + _DBS_COL_OFF)
                c_coo_v.append(_DBS_VALS)
            else:
                matrix = directional_beamsplitter_matrix()
                carrier_matrix[
                    directional_beamsplitter_index : directional_beamsplitter_index + matrix.shape[0],
                    directional_beamsplitter_index : directional_beamsplitter_index + matrix.shape[1],
                ] = matrix

        # -----------------------------
        # Mirrors and beamsplitters
        # -----------------------------
        if data["component"] in ["mirror", "beamsplitter"]:
            component_index = matrix_positions[node]
            parameter_positions[node] = len(parameters)

            if data["component"] == "mirror":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)

                props = data["properties"]
                loss_v = props["loss"]
                refl_v = props["reflectivity"]
                tuning_v = props["tuning"]
                nl_v = parameters[refractive_index_left_position]
                nr_v = parameters[refractive_index_right_position]
                if _sparse_signal:
                    abs_r = (1.0 - loss_v) * refl_v
                    abs_t = (1.0 - loss_v) * (1.0 - refl_v)
                    r_l = _reflectivity_left_np(abs_r, tuning_v, F, nl_v)
                    t_v = _transmissivity_np(abs_t, tuning_v, F, nl_v, nr_v)
                    r_r = _reflectivity_right_np(abs_r, tuning_v, F, nl_v, nr_v)
                else:
                    matrix = mirror_matrix(loss_v, refl_v, tuning_v, F, nl_v, nr_v)

                parameter_linking(
                    [loss_v, refl_v, tuning_v],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning"]
                loss_position = len(parameters) - 3

                function_input_indices = np.array(
                    [[len(parameters) - 3, len(parameters) - 2, len(parameters) - 1, 0, refractive_index_left_position, refractive_index_right_position, 2]],
                    dtype=np.int32,
                )
                signal_function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 3,
                            len(parameters) - 2,
                            len(parameters) - 1,
                            signal_frequency_position,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            2,
                        ]
                    ],
                    dtype=np.int32,
                )

                output_column_indices = _MIRROR_OUT_COL
                system_matrix_row_indices = component_index + _MIRROR_ROW_OFF
                system_matrix_column_indices = component_index + _MIRROR_COL_OFF

                sig_output_column_indices = _MIRROR_SIG_OUT_COL
                sig_noise_output_column_indices = _MIRROR_SIG_NOISE_OUT_COL
                noise_system_matrix_indices = component_index + _MIRROR_NOISE_OFF

                if _sparse_signal:
                    c_coo_r.append(system_matrix_row_indices)
                    c_coo_c.append(system_matrix_column_indices)
                    c_coo_v.append(np.array([r_l, t_v, t_v, r_r], dtype=np.complex128))

            elif data["component"] == "beamsplitter":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)

                props = data["properties"]
                loss_v = props["loss"]
                refl_v = props["reflectivity"]
                tuning_v = props["tuning"]
                alpha_v = props["alpha"]
                nl_v = parameters[refractive_index_left_position]
                nr_v = parameters[refractive_index_right_position]
                if _sparse_signal:
                    abs_r = (1.0 - loss_v) * refl_v
                    abs_t = (1.0 - loss_v) * (1.0 - refl_v)
                    r_l = _reflectivity_left_np(abs_r, tuning_v, F, nl_v, alpha_v)
                    t_v = _transmissivity_np(abs_t, tuning_v, F, nl_v, nr_v, alpha_v)
                    r_r = _reflectivity_right_np(abs_r, tuning_v, F, nl_v, nr_v, alpha_v)
                else:
                    matrix = beamsplitter_matrix(loss_v, refl_v, tuning_v, F, nl_v, nr_v, alpha_v)

                parameter_linking(
                    [loss_v, refl_v, tuning_v, alpha_v],
                    indices_to_link,
                    linked_names,
                    linking_function_indices,
                    parameters,
                )
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning", f"{node}_alpha"]
                loss_position = len(parameters) - 4

                function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 4,
                            len(parameters) - 3,
                            len(parameters) - 2,
                            0,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            len(parameters) - 1,
                        ]
                    ],
                    dtype=np.int32,
                )
                signal_function_input_indices = np.array(
                    [
                        [
                            len(parameters) - 4,
                            len(parameters) - 3,
                            len(parameters) - 2,
                            signal_frequency_position,
                            refractive_index_left_position,
                            refractive_index_right_position,
                            len(parameters) - 1,
                        ]
                    ],
                    dtype=np.int32,
                )

                output_column_indices = _BS_OUT_COL
                system_matrix_row_indices = component_index + _BS_ROW_OFF
                system_matrix_column_indices = component_index + _BS_COL_OFF

                sig_output_column_indices = _BS_SIG_OUT_COL
                sig_noise_output_column_indices = _BS_SIG_NOISE_OUT_COL
                noise_system_matrix_indices = component_index + _BS_NOISE_OFF

                if _sparse_signal:
                    c_coo_r.append(system_matrix_row_indices)
                    c_coo_c.append(system_matrix_column_indices)
                    c_coo_v.append(np.array([r_l, t_v, r_l, t_v, t_v, r_r, t_v, r_r], dtype=np.complex128))

            # Initialize the carrier matrix diagonal block for the surface.
            if not _sparse_signal:
                carrier_matrix[
                    component_index : component_index + matrix.shape[0],
                    component_index : component_index + matrix.shape[1],
                ] = matrix

            # Carrier update instructions for the surface (function_input_indices already width 7).
            carrier_instructions[node] = {
                "function_input_indices": function_input_indices,
                "output_column_indices": output_column_indices,
                "system_matrix_row_indices": system_matrix_row_indices,
                "system_matrix_column_indices": system_matrix_column_indices,
                "function_indices": _SURFACE_FUNC_LIST,
            }

            # Signal & noise instructions: all port indices are strictly < system_size.
            signal_instructions[node] = {
                "function_input_indices": signal_function_input_indices,
                "output_column_indices": sig_output_column_indices,
                "system_matrix_row_indices": np.concatenate((system_matrix_row_indices, system_matrix_row_indices + system_size)),
                "system_matrix_column_indices": np.concatenate((system_matrix_column_indices, system_matrix_column_indices + system_size)),
                "function_indices": _SURFACE_FUNC_LIST,
            }

            loss_inp = np.zeros((1, 7), dtype=np.int32)
            loss_inp[0, 0] = loss_position
            noise_idx_2 = np.concatenate((noise_system_matrix_indices, noise_system_matrix_indices + system_size))
            noise_instructions[node] = {
                "function_input_indices": loss_inp,
                "output_column_indices": sig_noise_output_column_indices,
                "system_matrix_row_indices": noise_idx_2,
                "system_matrix_column_indices": noise_idx_2,
                "function_indices": _LOSS_NOISE_FUNC_LIST,
            }

    # -----------------------------------------------------------------
    # Apply laser field modulations (needs final indices of both lasers and signals).
    # -----------------------------------------------------------------
    for node in laser_to_signals:
        laser_index = matrix_positions[node]

        signal_matrix_indices = []
        function_indices = []

        for signal in laser_to_signals[node]:
            signal_index = matrix_positions[signal]
            signal_matrix_indices += [system_size * 2 + signal_index] * 2

            target_property = setup.nodes[signal]["target_property"]
            if target_property == "amplitude":
                function_indices.extend([_IDX_LASER_AMP_MOD, _IDX_LASER_AMP_MOD])
            elif target_property == "frequency":
                function_indices.extend([_IDX_LASER_FREQ_MOD, _IDX_LASER_FREQ_MOD_LOWER])
            else:
                raise ValueError("Target property for laser modulation has to be either amplitude or frequency.")

        laser_signal_number = len(laser_to_signals[node])

        system_matrix_row_indices = np.array([laser_index, laser_index + system_size] * laser_signal_number)
        system_matrix_column_indices = np.array(signal_matrix_indices)

        set_instructions(
            signal_instructions,
            node,
            function_input_indices=np.tile(
                np.array([[signal_frequency_position], [signal_frequency_position]]),
                (laser_signal_number, 1),
            ),
            output_column_indices=np.array([0, 0] * laser_signal_number),
            output_row_indices=np.arange(laser_signal_number * 2),
            system_matrix_row_indices=system_matrix_row_indices,
            system_matrix_column_indices=system_matrix_column_indices,
            function_indices=function_indices,
            carrier_indices=np.array([laser_index, laser_index] * laser_signal_number),
            carrier_row_indices=system_matrix_row_indices,
            carrier_column_indices=system_matrix_column_indices,
        )

    # -----------------------------------------------------------------
    # Free mass coupling entries (needs surface parameters and indices).
    # -----------------------------------------------------------------
    for node in free_masses:
        data = setup.nodes[node]
        try:
            component_index = matrix_positions[data["target"]]
            parameter_position = parameter_positions[data["target"]]
            target_component = setup.nodes[data["target"]]["component"]
        except KeyError:
            raise ValueError("Free mass has to be connected to either a mirror or a beamsplitter.")

        if target_component == "mirror":
            fm = matrix_positions[node] + 2 * system_size

            parameter_linking(
                [data["properties"]["mass"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_mass"]

            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)

            c = component_index
            cs = c + system_size
            exp_rows = np.array(
                [fm + 1, fm, fm, fm, fm, c + 1, c + 3, cs + 1, cs + 3, fm, fm, fm, fm],
                dtype=np.int32,
            )
            exp_cols = np.array(
                [fm, c, c + 1, c + 2, c + 3, fm + 1, fm + 1, fm + 1, fm + 1, cs, cs + 1, cs + 2, cs + 3],
                dtype=np.int32,
            )
            signal_instructions[node] = {
                "function_input_indices": np.array(
                    [
                        [signal_frequency_position, len(parameters) - 1, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0],
                        [2, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, 2, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, 2],
                    ],
                    dtype=np.int32,
                ),
                "output_row_indices": _FM_MIRROR_OUT_ROW,
                "output_column_indices": _FM_MIRROR_OUT_COL,
                "system_matrix_row_indices": exp_rows,
                "system_matrix_column_indices": exp_cols,
                "function_indices": _FM_FUNC_LIST,
                "carrier_indices": c + _FM_MIRROR_CAR_OFF,
                "carrier_row_indices": exp_rows[1:],
                "carrier_column_indices": exp_cols[1:],
            }

        elif target_component == "beamsplitter":
            fm = matrix_positions[node] + 2 * system_size

            parameter_linking(
                [data["properties"]["mass"]],
                indices_to_link,
                linked_names,
                linking_function_indices,
                parameters,
            )
            parameter_names += [f"{node}_mass"]

            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)

            c = component_index
            cs = c + system_size
            exp_rows = np.array(
                [
                    fm + 1,
                    fm, fm, fm, fm, fm, fm, fm, fm,
                    c + 1, c + 3, c + 5, c + 7,
                    cs + 1, cs + 3, cs + 5, cs + 7,
                    fm, fm, fm, fm, fm, fm, fm, fm,
                ],
                dtype=np.int32,
            )
            exp_cols = np.array(
                [
                    fm,
                    c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6, c + 7,
                    fm + 1, fm + 1, fm + 1, fm + 1,
                    fm + 1, fm + 1, fm + 1, fm + 1,
                    cs, cs + 1, cs + 2, cs + 3, cs + 4, cs + 5, cs + 6, cs + 7,
                ],
                dtype=np.int32,
            )
            signal_instructions[node] = {
                "function_input_indices": np.array(
                    [
                        [signal_frequency_position, len(parameters) - 1, 0, 0, 0, 0, 0],
                        [parameter_position + 3, 0, 0, 0, 0, 0, 0],
                        [parameter_position + 3, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, parameter_position + 3, 0],
                        [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, parameter_position + 3],
                    ],
                    dtype=np.int32,
                ),
                "output_row_indices": _FM_BS_OUT_ROW,
                "output_column_indices": _FM_BS_OUT_COL,
                "system_matrix_row_indices": exp_rows,
                "system_matrix_column_indices": exp_cols,
                "function_indices": _FM_FUNC_LIST,
                "carrier_indices": c + _FM_BS_CAR_OFF,
                "carrier_row_indices": exp_rows[1:],
                "carrier_column_indices": exp_cols[1:],
            }

    # -----------------------------------------------------------------
    # Edge loop: connect components through spaces and add space instructions.
    # -----------------------------------------------------------------
    for (source, target, data) in setup.edges(data=True):
        source_node = setup.nodes[source]
        target_node = setup.nodes[target]

        if source_node["component"] in MATRIX_SIZES:
            used_ports.add(source + "." + data["source_port"])
        if target_node["component"] in MATRIX_SIZES:
            used_ports.add(target + "." + data["target_port"])

        source_index = matrix_positions[source]
        target_index = matrix_positions[target]

        # Determine input and output indices for source and target ports.
        try:
            source_port_index = PORT_DICTS[setup.nodes[source]["component"]][data["source_port"]]
            source_input_index = source_index + source_port_index
            source_output_index = source_index + source_port_index + 1
        except KeyError:
            # Laser is represented at an output index when connected through a space endpoint.
            source_input_index = source_index - 1
            source_output_index = source_index

        try:
            target_port_index = PORT_DICTS[setup.nodes[target]["component"]][data["target_port"]]
            target_input_index = target_index + target_port_index
            target_output_index = target_index + target_port_index + 1
        except KeyError:
            # Detector is represented at an output index when connected through a space endpoint.
            target_input_index = target_index + 1
            target_output_index = target_index

        # Initialize the carrier matrix with the space propagation entry (pure NumPy).
        if F == 0:
            space_entry = -1.0 + 0.0j
        else:
            space_entry = -np.exp(
                -1j * 2.0 * np.pi * F * data["properties"]["length"] * data["properties"]["refractive_index"] / C_LIGHT
            )
        space_key = f"{source}_{target}"
        sp_rows = np.array([source_input_index, target_input_index], dtype=np.int32)
        sp_cols = np.array([target_output_index, source_output_index], dtype=np.int32)
        if _sparse_signal:
            c_coo_r.append(sp_rows)
            c_coo_c.append(sp_cols)
            c_coo_v.append(np.array([space_entry, space_entry], dtype=np.complex128))
        else:
            carrier_matrix[source_input_index, target_output_index] = space_entry
            carrier_matrix[target_input_index, source_output_index] = space_entry

        parameter_position = parameter_positions[space_key]

        sp_car_inp = np.zeros((1, 7), dtype=np.int32)
        sp_car_inp[0, 1] = parameter_position
        sp_car_inp[0, 2] = parameter_position + 1
        carrier_instructions[space_key] = {
            "function_input_indices": sp_car_inp,
            "output_column_indices": _SPACE_CAR_OUT_COL,
            "system_matrix_row_indices": sp_rows,
            "system_matrix_column_indices": sp_cols,
            "function_indices": _SPACE_CAR_FUNC_LIST,
        }

        # Signal run: update both sidebands. Lower sideband uses `space_lower`.
        sp_sig_inp = np.zeros((2, 7), dtype=np.int32)
        sp_sig_inp[:, 0] = signal_frequency_position
        sp_sig_inp[:, 1] = parameter_position
        sp_sig_inp[:, 2] = parameter_position + 1
        signal_instructions[space_key] = {
            "function_input_indices": sp_sig_inp,
            "output_row_indices": _SPACE_SIG_OUT_ROW,
            "output_column_indices": _SPACE_SIG_OUT_COL,
            "system_matrix_row_indices": np.array(
                [
                    source_input_index,
                    target_input_index,
                    source_input_index + system_size,
                    target_input_index + system_size,
                ],
                dtype=np.int32,
            ),
            "system_matrix_column_indices": np.array(
                [
                    target_output_index,
                    source_output_index,
                    target_output_index + system_size,
                    source_output_index + system_size,
                ],
                dtype=np.int32,
            ),
            "function_indices": _SPACE_SIG_FUNC_LIST,
        }

        # Space strain signals inject into the signal connector columns.
        if space_key in space_to_signals:
            sigs = space_to_signals[space_key]
            space_signal_number = len(sigs)
            if space_signal_number == 1:
                sig_col = system_size * 2 + matrix_positions[sigs[0]]
                mod_rows = np.array(
                    [source_input_index, target_input_index, source_input_index + system_size, target_input_index + system_size],
                    dtype=np.int32,
                )
                mod_cols = np.array([sig_col, sig_col, sig_col, sig_col], dtype=np.int32)
                signal_instructions[f"{space_key}_modulation"] = {
                    "function_input_indices": sp_sig_inp,
                    "output_column_indices": _SPACE_MOD_OUT_COL_1,
                    "output_row_indices": _SPACE_MOD_OUT_ROW_1,
                    "system_matrix_row_indices": mod_rows,
                    "system_matrix_column_indices": mod_cols,
                    "function_indices": _SPACE_MOD_FUNC_LIST_1,
                    "carrier_indices": np.array(
                        [source_input_index, target_input_index, source_input_index, target_input_index],
                        dtype=np.int32,
                    ),
                    "carrier_row_indices": mod_rows,
                    "carrier_column_indices": mod_cols,
                }
            else:
                signal_matrix_indices = []
                for signal in sigs:
                    signal_index = matrix_positions[signal]
                    signal_matrix_indices += [system_size * 2 + signal_index] * 4

                system_matrix_row_indices = np.array(
                    [source_input_index, target_input_index, source_input_index + system_size, target_input_index + system_size]
                    * space_signal_number,
                    dtype=np.int32,
                )
                system_matrix_column_indices = np.array(signal_matrix_indices, dtype=np.int32)
                signal_instructions[f"{space_key}_modulation"] = {
                    "function_input_indices": np.tile(sp_sig_inp, (space_signal_number, 1)),
                    "output_column_indices": np.zeros(4 * space_signal_number, dtype=np.int32),
                    "output_row_indices": np.repeat(np.arange(space_signal_number * 2, dtype=np.int32), 2),
                    "system_matrix_row_indices": system_matrix_row_indices,
                    "system_matrix_column_indices": system_matrix_column_indices,
                    "function_indices": _SPACE_MOD_FUNC_LIST_1 * space_signal_number,
                    "carrier_indices": np.array(
                        [source_input_index, target_input_index, source_input_index, target_input_index]
                        * space_signal_number,
                        dtype=np.int32,
                    ),
                    "carrier_row_indices": system_matrix_row_indices,
                    "carrier_column_indices": system_matrix_column_indices,
                }

    # Insert carrier matrix into both sideband blocks of the signal matrix.
    if not _sparse_signal:
        signal_matrix[:system_size, :system_size] = carrier_matrix[:, :system_size]
        signal_matrix[system_size:-signal_size, system_size:-signal_size - 1] = carrier_matrix[:, :system_size]

    # -----------------------------------------------------------------
    # Mark unused ports and inject vacuum noise at those ports.
    # -----------------------------------------------------------------
    for node_port in all_ports - used_ports:
        node, port = node_port.split(".")
        try:
            port_offset = PORT_DICTS[setup.nodes[node]["component"]][port]
        except KeyError:
            port_offset = 0

        node_index = matrix_positions[node]
        port_index = node_index + port_offset

        port_pair = np.array([port_index, port_index + system_size], dtype=np.int32)
        noise_instructions[node_port] = {
            "function_input_indices": _VACUUM_INP_PADDED,
            "output_column_indices": _VACUUM_OUT_COL_2,
            "system_matrix_row_indices": port_pair,
            "system_matrix_column_indices": port_pair,
            "function_indices": _VACUUM_FUNC_LIST,
        }

    # -----------------------------------------------------------------
    # Build port-to-index lookup for carrier ports.
    # -----------------------------------------------------------------
    for node, data in setup.nodes(data=True):
        component = data.get("component")
        if component not in PORT_DICTS:
            continue

        node_index = matrix_positions[node]
        for port_name, port_offset in PORT_DICTS[component].items():
            port_to_index[f"{node}.{port_name}.in"] = node_index + port_offset
            port_to_index[f"{node}.{port_name}.out"] = node_index + port_offset + 1

    # Convert linked parameter names to parameter index positions.
    param_name_to_idx = {name: i for i, name in enumerate(parameter_names)}
    linked_indices = [param_name_to_idx[name] for name in linked_names]
    if not as_jax:
        carrier_out = (
            (system_size, np.concatenate(c_coo_r), np.concatenate(c_coo_c), np.concatenate(c_coo_v), c_rhs)
            if _sparse_signal
            else carrier_matrix
        )
        return (
            (carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names),
            (
                np.array([parameters], dtype=np.complex128),
                carrier_out,
                signal_matrix,
                noise_matrix,
                noise_selection_vectors,
                np.array(qhd_parameter_indices, dtype=np.int32),
                np.array(qhd_placing_indices, dtype=np.int32),
                np.array(linked_indices, dtype=np.int32),
                np.array(linking_function_indices, dtype=np.int32),
                np.array(indices_to_link, dtype=np.int32),
            ),
            (
                jnp.asarray(np.asarray(detector_indices, dtype=np.int64)),
                jnp.asarray(np.asarray(mirror_indices, dtype=np.int64)),
                jnp.asarray(np.asarray(beamsplitter_indices, dtype=np.int64)),
                jnp.asarray(np.asarray(isolator_indices, dtype=np.int64)),
                port_to_index,
            ),
        )

    if len(linked_indices) == 0:
        linked_indices = [0]
        indices_to_link = [0]
        LINKING_FUNCTIONS.append(lambda x: x)
        linking_function_indices = [0]

    return (
        (carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names),
        (
            jnp.array([parameters], dtype=complex),
            jnp.array(carrier_matrix),
            jnp.array(signal_matrix),
            jnp.array(noise_matrix, dtype=complex),
            jnp.array(noise_selection_vectors, dtype=complex),
            jnp.array(qhd_parameter_indices, dtype=int),
            jnp.array(qhd_placing_indices, dtype=int),
            jnp.array(linked_indices, dtype=int),
            jnp.array(linking_function_indices, dtype=int),
            jnp.array(indices_to_link, dtype=int),
        ),
        (
            jnp.array(detector_indices, dtype=int),
            jnp.array(mirror_indices, dtype=int),
            jnp.array(beamsplitter_indices, dtype=int),
            jnp.array(isolator_indices, dtype=int),
            port_to_index,
        ),
    )


def pairs_to_arrays(
    parameter_instructions: dict,
    signal_instructions: dict,
    noise_instructions: dict,
    signal_components: list,
    parameter_names: list,
    optimization_pairs: list | None = None,
    changing_pairs: list | None = None,
):
    """
    Convert instruction dictionaries into dense index arrays for vectorized JAX execution.

    This function gathers update instructions from:
    - Carrier updates (`parameter_instructions`)
    - Signal updates (`signal_instructions`)
    - Noise updates (`noise_instructions`)

    It then compiles them into stacks of arrays, which downstream code can use to:
    - Gather parameter values by index,
    - Call a batch of component functions,
    - Scatter results into matrices using integer index arrays.

    Parameters
    ----------
    parameter_instructions:
        Carrier update instructions keyed by component identifier.
    signal_instructions:
        Signal update instructions keyed by component identifier.
    noise_instructions:
        Noise update instructions keyed by component identifier.
    signal_components:
        List of component identifiers that belong to the signal solver update set.
    parameter_names:
        List of parameter names aligned with the parameter vector indices.
    optimization_pairs:
        List of pairs `(component, parameter)` or list of lists to represent shared optimization values.
    changing_pairs:
        List of pairs `(component, parameter)` indicating parameters changed over a sweep.

    Returns
    -------
    preparation_arrays:
        `(optimized_parameter_indices, optimization_value_indices,
          carrier_changing_parameter_indices, signal_changing_parameter_indices)`
    carrier_arrays:
        Tuple of arrays needed to update the carrier matrix.
    signal_arrays:
        Tuple of arrays needed to update the signal matrix, including carrier-solution coupling indices.
    noise_arrays:
        Tuple of arrays needed to update the noise matrix.

    Notes
    -----
    - If an instruction set is empty, a dummy update is produced to keep array shapes valid.
    - The signal arrays may include additional arrays that reference carrier solution entries.
    """
    optimized_parameter_indices = []
    optimization_value_indices = []

    carrier_components = []
    carrier_changing_parameter_indices = []
    carrier_arrays = defaultdict(list)

    signal_changing_parameter_indices = []
    signal_arrays = defaultdict(list)

    noise_components = []
    noise_arrays = defaultdict(list)

    if optimization_pairs is None:
        optimization_pairs = []
    if changing_pairs is None:
        changing_pairs = []

    signal_component_number = 0

    # Compile signal update arrays by iterating through all signal instructions.
    for instruction in signal_instructions.values():
        try:
            instruction["function_indices"]
        except KeyError:
            continue

        signal_arrays["function_input_indices"].append(instruction["function_input_indices"])
        signal_arrays["function_indices"].extend(instruction["function_indices"])
        signal_arrays["output_column_indices"].append(instruction["output_column_indices"])

        if "output_row_indices" in instruction:
            signal_arrays["output_row_indices"].append(instruction["output_row_indices"].flatten() + signal_component_number)
            individual_rows = len(np.unique(instruction["output_row_indices"]))
            signal_component_number += individual_rows
        else:
            signal_arrays["output_row_indices"].append(np.ones(len(instruction["output_column_indices"])) * signal_component_number)
            signal_component_number += 1

        signal_arrays["system_matrix_row_indices"].append(instruction["system_matrix_row_indices"])
        signal_arrays["system_matrix_column_indices"].append(instruction["system_matrix_column_indices"])

        if "carrier_indices" in instruction:
            signal_arrays["carrier_indices"].append(instruction["carrier_indices"])
            signal_arrays["carrier_row_indices"].append(instruction["carrier_row_indices"])
            signal_arrays["carrier_column_indices"].append(instruction["carrier_column_indices"])

    def append_information(instructions: dict, arrays: dict, component: str, components: list | None = None):
        """
        Helper to append one component's instruction data to a target arrays dictionary.

        Parameters
        ----------
        instructions:
            Instruction dictionary keyed by component.
        arrays:
            Destination dictionary of lists that will later be stacked into arrays.
        component:
            Component key to append.
        components:
            List used as a "seen" set to avoid adding the same component multiple times.

        Returns
        -------
        None
            This function mutates `arrays` and `components` in-place.
        """
        component_function_input_indices = instructions[component]["function_input_indices"]

        if component in components:
            return None
        components.append(component)

        arrays["function_input_indices"].append(component_function_input_indices)
        arrays["function_indices"].extend(instructions[component]["function_indices"])
        arrays["output_column_indices"].append(instructions[component]["output_column_indices"])
        arrays["output_row_indices"].append(
            np.ones(len(instructions[component]["output_column_indices"])) * (len(components) - 1)
        )
        arrays["system_matrix_row_indices"].append(instructions[component]["system_matrix_row_indices"])
        arrays["system_matrix_column_indices"].append(instructions[component]["system_matrix_column_indices"])

    # Compile noise arrays first (every noise instruction is always relevant).
    for component, instruction in noise_instructions.items():
        append_information(noise_instructions, noise_arrays, component, noise_components)

    # Compile optimized parameter indices and include any carrier-side components that need updating.
    for ix, optimization_pair in enumerate(optimization_pairs):
        if not isinstance(optimization_pair[0], list):
            optimized_component, optimized_parameter = optimization_pair
            try:
                optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                optimization_value_indices.append(ix)
            except ValueError:
                pass
        else:
            for optimized_component, optimized_parameter in optimization_pair:
                try:
                    optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                    optimization_value_indices.append(ix)
                except ValueError:
                    pass

        if optimized_component not in signal_components and optimized_component in parameter_instructions:
            append_information(parameter_instructions, carrier_arrays, optimized_component, carrier_components)

    # Compile changing parameter indices for sweeps.
    for changing_component, changing_parameter in changing_pairs:
        if changing_component in signal_components:
            signal_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
        else:
            carrier_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
            append_information(parameter_instructions, carrier_arrays, changing_component, carrier_components)

    def combine_arrays(arrays: dict, signal: bool = False):
        """
        Stack per-component lists into dense arrays. Provide a dummy update if empty.

        Parameters
        ----------
        arrays:
            Dictionary of lists populated by `append_information`.
        signal:
            If True, include carrier-solution coupling indices required by the signal solver.

        Returns
        -------
        tuple
            For carrier and noise: `(function_input_indices, function_indices, output_indices, system_matrix_indices)`
            For signal: plus carrier coupling arrays:
            `(function_input_indices, function_indices, output_indices, system_matrix_indices,
              carrier_solution_indices, carrier_solution_placing_indices)`
        """
        if len(arrays["function_input_indices"]) == 0:
            # Dummy update that leaves matrices unchanged.
            return (
                jnp.array([[0, 0, 0]]),
                jnp.array([_IDX_DUMMY]),
                jnp.array([[0], [0]]),
                jnp.array([[0], [0]]),
            )

        function_input_indices = np.vstack(arrays["function_input_indices"])
        output_indices = np.stack((np.concatenate(arrays["output_row_indices"]), np.concatenate(arrays["output_column_indices"])))
        system_matrix_indices = np.stack((np.concatenate(arrays["system_matrix_row_indices"]), np.concatenate(arrays["system_matrix_column_indices"])))

        if signal:
            # Dummy carrier coupling values if none are required.
            carrier_solution_indices = np.array([0])
            carrier_solution_placing_indices = np.array([[0], [-1]])

            if "carrier_indices" in arrays:
                carrier_solution_indices = np.concatenate(arrays["carrier_indices"])
                carrier_solution_placing_indices = np.stack(
                    (np.concatenate(arrays["carrier_row_indices"]), np.concatenate(arrays["carrier_column_indices"]))
                )

            return (
                jnp.array(function_input_indices, dtype=int),
                jnp.array(arrays["function_indices"], dtype=int),
                jnp.array(output_indices, dtype=int),
                jnp.array(system_matrix_indices, dtype=int),
                jnp.array(carrier_solution_indices, dtype=int),
                jnp.array(carrier_solution_placing_indices, dtype=int),
            )

        return (
            jnp.array(function_input_indices, dtype=int),
            jnp.array(arrays["function_indices"], dtype=int),
            jnp.array(output_indices, dtype=int),
            jnp.array(system_matrix_indices, dtype=int),
        )

    return (
        (
            jnp.array(optimized_parameter_indices, dtype=int),
            jnp.array(optimization_value_indices, dtype=int),
            jnp.array(carrier_changing_parameter_indices, dtype=int),
            jnp.array(signal_changing_parameter_indices, dtype=int),
        ),
        combine_arrays(carrier_arrays),
        combine_arrays(signal_arrays, signal=True),
        combine_arrays(noise_arrays),
    )


def prepare_arrays(
    optimized_parameter_indices,
    optimized_value_indices,
    carrier_changing_parameter_indices,
    signal_changing_parameter_indices,
    changing_values,
    parameters,
):
    """
    Ensure that all index and value arrays are non-empty and have consistent shapes.

    This function inserts "do-nothing" dummy indices and values when a particular feature is unused,
    so downstream vectorized code can run without conditional branches.

    Parameters
    ----------
    optimized_parameter_indices:
        Integer indices into the parameter vector for parameters under optimization.
    optimized_value_indices:
        Integer indices mapping each optimized parameter to an entry in an optimization value vector.
    carrier_changing_parameter_indices:
        Integer indices for parameters changed in the carrier solver sweep.
    signal_changing_parameter_indices:
        Integer indices for parameters changed in the signal solver sweep.
    changing_values:
        Values for the changing parameters. Shape is either `(R,)` or `(V, R)`, where:
        - `V` is number of changing parameters,
        - `R` is number of sweep points.
    parameters:
        Parameter array with shape `(1, N)`.

    Returns
    -------
    tuple
        `(optimized_parameters, optimized_parameter_indices, optimized_value_indices,
          carrier_changing_parameter_indices, carrier_changing_values,
          signal_changing_parameter_indices, signal_changing_values)`

    Raises
    ------
    ValueError
        If both carrier and signal changing indices are non-empty (unsupported combination),
        or if `changing_values` has an incompatible shape.
    """
    optimized_parameters = None

    # Ensure at least one optimized parameter index exists, using a dummy identity update if needed.
    if len(optimized_parameter_indices) == 0:
        optimized_parameter_indices = jnp.array([0])
        optimized_parameters = jnp.array(parameters[0][optimized_parameter_indices])
        optimized_value_indices = jnp.array([0])

    # Normalize changing_values to shape (V, R) if provided.
    if changing_values is not None:
        if len(changing_values.shape) == 1:
            changing_values = changing_values.reshape(1, -1)
        signal_changing_values = changing_values.copy()

    if len(carrier_changing_parameter_indices) != 0 and len(signal_changing_parameter_indices) != 0:
        raise ValueError("This combination of changing component-parameter pairs is not supported yet.")

    # Carrier changing defaults.
    if len(carrier_changing_parameter_indices) == 0:
        carrier_changing_parameter_indices = jnp.array([0])
        carrier_changing_values = jnp.array([parameters[0][carrier_changing_parameter_indices]])
    else:
        carrier_changing_values = changing_values
        if carrier_changing_values.shape[0] != len(carrier_changing_parameter_indices):
            raise ValueError(
                "The changing_values array needs to have shape (number of changing parameters, number of values). "
                "The changing_values array here has shape "
                + str(changing_values.shape)
                + ", but there are "
                + str(len(carrier_changing_parameter_indices))
                + " changing parameters."
            )

    # Signal changing defaults.
    if len(signal_changing_parameter_indices) == 0:
        signal_changing_parameter_indices = jnp.array([0])
        signal_changing_values = jnp.array([parameters[0][signal_changing_parameter_indices]])

    return (
        optimized_parameters,
        optimized_parameter_indices,
        optimized_value_indices,
        carrier_changing_parameter_indices,
        carrier_changing_values,
        signal_changing_parameter_indices,
        signal_changing_values,
    )


# ---------------------------------------------------------------------
# Support dictionaries
# ---------------------------------------------------------------------

MATRIX_SIZES = {
    "mirror": 4,
    "beamsplitter": 8,
    "nothing": 4,
    "directional_beamsplitter": 8,
}
"""
Map from component name to the size of its diagonal submatrix.

These sizes define how many consecutive indices in the carrier system matrix are reserved
for each component type.
"""

PORT_DICTS = {
    "mirror": {"left": 0, "right": 2},
    "beamsplitter": {"left": 0, "right": 4, "bottom": 6, "top": 2},
    "nothing": {"left": 0, "right": 2},
    "directional_beamsplitter": {"left": 0, "right": 4, "bottom": 6, "top": 2},
}
"""
Port offset definitions for each matrix component.

Offsets are counted within the component submatrix, moving counter-clockwise and alternating
between input and output, starting at the left input.

For example for a mirror (size 4):
- left port input is at offset 0, left port output is at offset 1
- right port input is at offset 2, right port output is at offset 3
"""


_FUNCTION_USED_ARGS = (
    3,  # 0: force_calculation_right
    2,  # 1: laser
    7,  # 2: surface
    3,  # 3: space
    3,  # 4: space_modulation
    2,  # 5: signal_function
    0,  # 6: dummy_function
    0,  # 7: vacuum_quantum_noise
    1,  # 8: loss_quantum_noise
    3,  # 9: space_lower
    0,  # 10: laser_amplitude_modulation
    1,  # 11: laser_frequency_modulation
    2,  # 12: susceptibility
    1,  # 13: force_calculation_left
    6,  # 14: corrected_optomechanical_phase_left
    7,  # 15: corrected_optomechanical_phase_right
    3,  # 16: space_modulation_lower
    2,  # 17: squeezer
    1,  # 18: laser_frequency_modulation_lower
)


def _pack_sparse_csc_coo(
    n: int,
    base_r: np.ndarray,
    base_c: np.ndarray,
    base_v: np.ndarray,
    base_rhs: np.ndarray | None,
    extra_rows: np.ndarray,
    extra_cols: np.ndarray,
):
    """Build static CSC structure directly from unique base COO entries + extra pattern."""
    base_keys = base_c.astype(np.int64) * n + base_r.astype(np.int64)
    if len(extra_rows) > 0:
        norm_c = np.where(extra_cols < 0, n, extra_cols)
        mat_mask = norm_c < n
        extra_keys = norm_c[mat_mask].astype(np.int64) * n + extra_rows[mat_mask].astype(np.int64)
        uniq_keys = np.unique(np.concatenate((base_keys, extra_keys)))
    else:
        uniq_keys = np.unique(base_keys)

    csc_cols = (uniq_keys // n).astype(np.int32)
    csc_rows = (uniq_keys % n).astype(np.int32)
    nnz_A = len(csc_rows)

    csc_indices = csc_rows
    counts = np.bincount(csc_cols, minlength=n)
    csc_indptr = np.empty(n + 1, dtype=np.int32)
    csc_indptr[0] = 0
    np.cumsum(counts, out=csc_indptr[1:])

    base_aug = np.zeros(nnz_A + n, dtype=np.complex128)
    if base_keys.size:
        base_aug[np.searchsorted(uniq_keys, base_keys)] = base_v
    if base_rhs is not None:
        base_aug[nnz_A:] = base_rhs

    return base_aug, csc_rows, csc_cols, csc_indices, csc_indptr, uniq_keys, nnz_A


def _pack_sparse_csc(base_matrix: np.ndarray, extra_rows: np.ndarray, extra_cols: np.ndarray):
    """Build static CSC structure for an augmented (N, N + 1) matrix in pure NumPy."""
    n = base_matrix.shape[0]
    base_A = base_matrix[:, :n]
    base_r, base_c = np.nonzero(base_A)
    return _pack_sparse_csc_coo(n, base_r, base_c, base_A[base_r, base_c], base_matrix[:, n], extra_rows, extra_cols)


def _map_rc_to_flat_pos(rows: np.ndarray, cols: np.ndarray, csc_keys: np.ndarray, nnz_A: int, n: int) -> np.ndarray:
    rhs = (cols < 0) | (cols == n)
    keys = cols.astype(np.int64) * n + rows
    return np.where(rhs, nnz_A + rows, np.searchsorted(csc_keys, keys)).astype(np.int32)


_FUNCTION_USED_ARGS_ARR = np.asarray(_FUNCTION_USED_ARGS, dtype=np.int32)
_CONSTANT_FUNCS = frozenset((6, 7, 10))


def _group_instructions_np(
    inp_indices: np.ndarray,
    func_indices: list[int],
    out_row: np.ndarray,
    out_col: np.ndarray,
    flat_pos: np.ndarray,
    *,
    sweep_param_set: set[int] | None = None,
    sweep_rows: np.ndarray | None = None,
    static_sweep_set: set[int] | None = None,
    bwd_map: np.ndarray | None = None,
):
    """Partition component update calls by (is_sweep, func_idx) into _OpGroup objects."""
    from differometor._sparse_kernel import build_op_group

    n_k = len(func_indices)
    if n_k == 0 or flat_pos.size == 0:
        return (), ()

    f_arr = np.asarray(func_indices, dtype=np.int32)
    if sweep_param_set:
        used_arr = _FUNCTION_USED_ARGS_ARR[f_arr]
        col_mask = np.arange(inp_indices.shape[1], dtype=np.int32)[None, :] < used_arr[:, None]
        if len(sweep_param_set) == 1:
            sw0 = next(iter(sweep_param_set))
            is_sweep = np.any((inp_indices == sw0) & col_mask, axis=1)
        else:
            sw_list = np.fromiter(sweep_param_set, dtype=inp_indices.dtype, count=len(sweep_param_set))
            is_sweep = np.any(np.isin(inp_indices, sw_list) & col_mask, axis=1)
    else:
        is_sweep = np.zeros(n_k, dtype=bool)

    # Bucket id: 0..18 for static, 19..37 for sweep
    bucket_id = f_arr + np.where(is_sweep, 19, 0)
    k_order = np.argsort(bucket_id, kind="stable")
    b_sorted = bucket_id[k_order]
    b_changes = np.empty(n_k, dtype=bool)
    b_changes[0] = True
    b_changes[1:] = b_sorted[1:] != b_sorted[:-1]
    b_starts = np.flatnonzero(b_changes)
    b_ends = np.empty_like(b_starts)
    b_ends[:-1] = b_starts[1:]
    b_ends[-1] = n_k
    uniq_buckets = b_sorted[b_starts]

    # Map each old_k -> (bucket_slot, local_k)
    old_k_to_slot = np.empty(n_k, dtype=np.intp)
    old_k_to_local = np.empty(n_k, dtype=np.intp)
    slot_k_lists = []
    for slot_i, (s, e) in enumerate(zip(b_starts, b_ends)):
        ks = k_order[s:e]
        slot_k_lists.append(ks)
        old_k_to_slot[ks] = slot_i
        old_k_to_local[ks] = np.arange(e - s, dtype=np.intp)

    # Group scatter entries by (slot, local_k) preserving original order within each local_k
    m_slots = old_k_to_slot[out_row]
    m_local = old_k_to_local[out_row]
    m_key = m_slots * np.int64(n_k) + m_local
    m_order = np.argsort(m_key, kind="stable")
    m_slots_sorted = m_slots[m_order]
    m_local_sorted = m_local[m_order]
    m_out_col_sorted = out_col[m_order]
    m_flat_pos_sorted = flat_pos[m_order]

    slot_counts = np.bincount(m_slots_sorted, minlength=len(uniq_buckets))
    slot_offsets = np.empty(len(uniq_buckets) + 1, dtype=np.intp)
    slot_offsets[0] = 0
    np.cumsum(slot_counts, out=slot_offsets[1:])

    static_groups = []
    sweep_groups = []
    for slot_i, bid in enumerate(uniq_buckets):
        ms = int(slot_offsets[slot_i])
        me = int(slot_offsets[slot_i + 1])
        if ms == me:
            continue
        bid_int = int(bid)
        is_sw_bucket = bid_int >= 19
        f_idx_int = bid_int - 19 if is_sw_bucket else bid_int
        used = _FUNCTION_USED_ARGS[f_idx_int]
        ks = slot_k_lists[slot_i]
        inp_sub = np.ascontiguousarray(inp_indices[ks, :used], dtype=np.intp)
        fp_slice = m_flat_pos_sorted[ms:me]
        bwd_fp = bwd_map[fp_slice] if (bwd_map is not None and f_idx_int not in _CONSTANT_FUNCS) else None
        n_out_cols = 3 if f_idx_int == 2 else (4 if f_idx_int == 17 else 1)
        grp = build_op_group(
            f_idx_int,
            inp_sub,
            m_local_sorted[ms:me],
            m_out_col_sorted[ms:me],
            fp_slice,
            n_out_cols,
            bwd_flat_pos=bwd_fp,
            sweep_rows=sweep_rows if is_sw_bucket else None,
            batch_sweep_set=sweep_param_set if is_sw_bucket else static_sweep_set,
        )
        if is_sw_bucket:
            sweep_groups.append(grp)
        else:
            static_groups.append(grp)

    return tuple(static_groups), tuple(sweep_groups)


def build_sparse_simulation_arrays(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
):
    """Build compact sparse CSC topology and minimal JAX arrays for `simulate`."""
    from differometor._sparse_kernel import SparseTopology, _CSCPattern, resolve_host_link
    from differometor._sparse_blocks import build_scatter_plan, plan_signal_blocks

    instructions, matrices, metadata = build(setup, as_jax=False, _sparse_signal=True)
    carrier_instructions, signal_instructions, noise_instructions, signal_components, parameter_names = instructions
    (
        parameters_np,
        carrier_matrix_np,
        signal_matrix_np,
        noise_matrix_np,
        noise_selection_vectors_np,
        qhd_parameter_indices_np,
        qhd_placing_indices_np,
        linked_indices_np,
        linking_function_indices_np,
        indices_to_link_np,
    ) = matrices

    param_name_to_idx = {name: i for i, name in enumerate(parameter_names)}
    signal_components_set = set(signal_components)

    if optimization_pairs is None:
        optimization_pairs = []
    if changing_pairs is None:
        changing_pairs = []

    optimized_parameter_indices = []
    optimization_value_indices = []
    carrier_components_seen = set()
    carrier_arrays_raw = defaultdict(list)
    carrier_changing_parameter_indices = []
    signal_changing_parameter_indices = []

    def append_info(instr_dict, target, comp, seen):
        if comp in seen:
            return
        seen.add(comp)
        inst = instr_dict[comp]
        target["function_input_indices"].append(inst["function_input_indices"])
        target["function_indices"].extend(inst["function_indices"])
        target["output_column_indices"].append(inst["output_column_indices"])
        target["output_row_indices"].append(np.full(len(inst["output_column_indices"]), len(seen) - 1, dtype=np.int32))
        target["system_matrix_row_indices"].append(inst["system_matrix_row_indices"])
        target["system_matrix_column_indices"].append(inst["system_matrix_column_indices"])

    noise_components_seen = set()
    noise_arrays_raw = defaultdict(list)
    for comp in noise_instructions:
        append_info(noise_instructions, noise_arrays_raw, comp, noise_components_seen)

    signal_arrays_raw = defaultdict(list)
    signal_component_number = 0
    for inst in signal_instructions.values():
        if "function_indices" not in inst:
            continue
        signal_arrays_raw["function_input_indices"].append(inst["function_input_indices"])
        signal_arrays_raw["function_indices"].extend(inst["function_indices"])
        signal_arrays_raw["output_column_indices"].append(inst["output_column_indices"])
        if "output_row_indices" in inst:
            signal_arrays_raw["output_row_indices"].append(inst["output_row_indices"].flatten() + signal_component_number)
            signal_component_number += inst["function_input_indices"].shape[0]
        else:
            signal_arrays_raw["output_row_indices"].append(
                np.full(len(inst["output_column_indices"]), signal_component_number, dtype=np.int32)
            )
            signal_component_number += 1
        signal_arrays_raw["system_matrix_row_indices"].append(inst["system_matrix_row_indices"])
        signal_arrays_raw["system_matrix_column_indices"].append(inst["system_matrix_column_indices"])
        if "carrier_indices" in inst:
            signal_arrays_raw["carrier_indices"].append(inst["carrier_indices"])
            signal_arrays_raw["carrier_row_indices"].append(inst["carrier_row_indices"])
            signal_arrays_raw["carrier_column_indices"].append(inst["carrier_column_indices"])

    for ix, opt_pair in enumerate(optimization_pairs):
        if not isinstance(opt_pair[0], list):
            opt_comp, opt_param = opt_pair
            idx = param_name_to_idx.get(f"{opt_comp}_{opt_param}")
            if idx is not None:
                optimized_parameter_indices.append(idx)
                optimization_value_indices.append(ix)
        else:
            for opt_comp, opt_param in opt_pair:
                idx = param_name_to_idx.get(f"{opt_comp}_{opt_param}")
                if idx is not None:
                    optimized_parameter_indices.append(idx)
                    optimization_value_indices.append(ix)
        if opt_comp not in signal_components_set and opt_comp in carrier_instructions:
            append_info(carrier_instructions, carrier_arrays_raw, opt_comp, carrier_components_seen)

    for ch_comp, ch_param in changing_pairs:
        idx = param_name_to_idx[f"{ch_comp}_{ch_param}"]
        if ch_comp in signal_components_set:
            signal_changing_parameter_indices.append(idx)
        else:
            carrier_changing_parameter_indices.append(idx)
            append_info(carrier_instructions, carrier_arrays_raw, ch_comp, carrier_components_seen)

    if len(carrier_changing_parameter_indices) != 0 and len(signal_changing_parameter_indices) != 0:
        raise ValueError("This combination of changing component-parameter pairs is not supported yet.")

    if changing_values is not None:
        ch_vals = jnp.asarray(np.atleast_2d(changing_values), dtype=jnp.float64)
    else:
        ch_vals = None

    empty_sweep = jnp.asarray(np.empty((0, 1), dtype=np.float64))
    if len(carrier_changing_parameter_indices) > 0:
        carrier_changing_values = ch_vals
        if carrier_changing_values.shape[0] != len(carrier_changing_parameter_indices):
            raise ValueError(
                "The changing_values array needs to have shape (number of changing parameters, number of values)."
            )
    else:
        carrier_changing_values = empty_sweep

    if len(signal_changing_parameter_indices) > 0:
        signal_changing_values = ch_vals
    else:
        signal_changing_values = empty_sweep

    car_ch_idx_np = np.asarray(carrier_changing_parameter_indices, dtype=np.intp)
    sig_ch_idx_np = np.asarray(signal_changing_parameter_indices, dtype=np.intp)
    car_ch_set = set(carrier_changing_parameter_indices) if carrier_changing_parameter_indices else None
    sig_ch_set = set(signal_changing_parameter_indices) if signal_changing_parameter_indices else None

    sig_sweep_rows = None
    if sig_ch_idx_np.size > 0:
        sig_sweep_rows = np.full(parameters_np.shape[1], -1, dtype=np.intp)
        sig_sweep_rows[sig_ch_idx_np] = np.arange(sig_ch_idx_np.size, dtype=np.intp)

    # --- Carrier CSC & grouped updates (direct COO, skipping dense cn x (cn+1)) ---
    cn, c_base_r, c_base_c, c_base_v, c_base_rhs = carrier_matrix_np
    if len(carrier_arrays_raw["function_input_indices"]) > 0:
        c_inp = np.vstack(carrier_arrays_raw["function_input_indices"]).astype(np.int32)
        c_out_r = np.concatenate(carrier_arrays_raw["output_row_indices"]).astype(np.int32)
        c_out_c = np.concatenate(carrier_arrays_raw["output_column_indices"]).astype(np.int32)
        c_mat_r = np.concatenate(carrier_arrays_raw["system_matrix_row_indices"]).astype(np.int32)
        c_mat_c = np.concatenate(carrier_arrays_raw["system_matrix_column_indices"]).astype(np.int32)
    else:
        c_inp = np.zeros((0, 7), dtype=np.int32)
        c_out_r = np.zeros((0,), dtype=np.int32)
        c_out_c = np.zeros((0,), dtype=np.int32)
        c_mat_r = np.zeros((0,), dtype=np.int32)
        c_mat_c = np.zeros((0,), dtype=np.int32)

    c_base_aug, c_rows, c_cols, c_indices, c_indptr, c_keys, c_nnz = _pack_sparse_csc_coo(
        cn, c_base_r, c_base_c, c_base_v, c_base_rhs, c_mat_r, c_mat_c
    )
    if len(c_mat_r) > 0:
        c_flat_pos = _map_rc_to_flat_pos(c_mat_r, c_mat_c, c_keys, c_nnz, cn)
        carrier_groups, _ = _group_instructions_np(
            c_inp, carrier_arrays_raw["function_indices"], c_out_r, c_out_c, c_flat_pos,
            sweep_param_set=None, static_sweep_set=car_ch_set,
        )
    else:
        carrier_groups = ()

    # --- Signal CSC & grouped updates (built directly in COO, skipping dense sn x (sn+1)) ---
    sn = int(signal_matrix_np)
    if len(signal_arrays_raw["function_input_indices"]) > 0:
        s_inp = np.vstack(signal_arrays_raw["function_input_indices"]).astype(np.int32)
        s_out_r = np.concatenate(signal_arrays_raw["output_row_indices"]).astype(np.int32)
        s_out_c = np.concatenate(signal_arrays_raw["output_column_indices"]).astype(np.int32)
        s_mat_r = np.concatenate(signal_arrays_raw["system_matrix_row_indices"]).astype(np.int32)
        s_mat_c = np.concatenate(signal_arrays_raw["system_matrix_column_indices"]).astype(np.int32)
    else:
        s_inp = np.zeros((0, 7), dtype=np.int32)
        s_out_r = np.zeros((0,), dtype=np.int32)
        s_out_c = np.zeros((0,), dtype=np.int32)
        s_mat_r = np.zeros((0,), dtype=np.int32)
        s_mat_c = np.zeros((0,), dtype=np.int32)

    if "carrier_indices" in signal_arrays_raw:
        s_car_idx = np.concatenate(signal_arrays_raw["carrier_indices"]).astype(np.intp)
        s_car_r = np.concatenate(signal_arrays_raw["carrier_row_indices"]).astype(np.int32)
        s_car_c = np.concatenate(signal_arrays_raw["carrier_column_indices"]).astype(np.int32)
        extra_s_r = np.concatenate([s_mat_r, s_car_r]) if len(s_mat_r) > 0 else s_car_r
        extra_s_c = np.concatenate([s_mat_c, s_car_c]) if len(s_mat_c) > 0 else s_car_c
    else:
        s_car_idx = np.zeros((0,), dtype=np.intp)
        s_car_r = np.zeros((0,), dtype=np.int32)
        s_car_c = np.zeros((0,), dtype=np.int32)
        extra_s_r = s_mat_r
        extra_s_c = s_mat_c

    s_diag = np.arange(2 * cn, sn, dtype=np.int64)
    s_base_r = np.concatenate((c_base_r, c_base_r + cn, s_diag))
    s_base_c = np.concatenate((c_base_c, c_base_c + cn, s_diag))
    s_base_v = np.concatenate((c_base_v, c_base_v, np.ones(s_diag.size, dtype=np.complex128)))
    s_base_aug, s_rows, s_cols, s_indices, s_indptr, s_keys, s_nnz = _pack_sparse_csc_coo(
        sn, s_base_r, s_base_c, s_base_v, None, extra_s_r, extra_s_c
    )
    if len(s_mat_r) > 0:
        s_flat_pos = _map_rc_to_flat_pos(s_mat_r, s_mat_c, s_keys, s_nnz, sn)
    else:
        s_flat_pos = np.zeros((0,), dtype=np.int32)

    if len(s_car_idx) > 0:
        s_car_flat_pos = _map_rc_to_flat_pos(s_car_r, s_car_c, s_keys, s_nnz, sn).astype(np.intp)
    else:
        s_car_flat_pos = np.zeros((0,), dtype=np.intp)

    # Identify active backward signal slots (only non-constant groups + carrier couplings)
    active_mask = np.zeros(s_nnz + sn, dtype=bool)
    if s_car_flat_pos.size > 0:
        active_mask[s_car_flat_pos] = True
    if len(s_mat_r) > 0:
        s_f_arr = np.asarray(signal_arrays_raw["function_indices"], dtype=np.int32)
        nonconst_k = (s_f_arr != 6) & (s_f_arr != 7) & (s_f_arr != 10)
        nonconst_m = nonconst_k[s_out_r]
        if nonconst_m.any():
            active_mask[s_flat_pos[nonconst_m]] = True

    bwd_a_pos = np.flatnonzero(active_mask[:s_nnz])
    bwd_rhs_rows = np.flatnonzero(active_mask[s_nnz:])
    bwd_n_act_a = int(bwd_a_pos.size)
    bwd_n_act = bwd_n_act_a + int(bwd_rhs_rows.size)
    bwd_map = np.full(s_nnz + sn, -1, dtype=np.intp)
    bwd_map[bwd_a_pos] = np.arange(bwd_n_act_a, dtype=np.intp)
    bwd_map[s_nnz + bwd_rhs_rows] = np.arange(bwd_n_act_a, bwd_n_act, dtype=np.intp)

    if len(s_mat_r) > 0:
        signal_groups_static, signal_groups_sweep = _group_instructions_np(
            s_inp, signal_arrays_raw["function_indices"], s_out_r, s_out_c, s_flat_pos,
            sweep_param_set=sig_ch_set, sweep_rows=sig_sweep_rows,
            static_sweep_set=car_ch_set, bwd_map=bwd_map,
        )
    else:
        signal_groups_static, signal_groups_sweep = (), ()

    bwd_sig_car_pos = bwd_map[s_car_flat_pos] if s_car_flat_pos.size > 0 else np.zeros((0,), dtype=np.intp)

    static_bwd_parts = [
        bwd_map[g.flat_pos] for g in signal_groups_static if g.const_val is None
    ]
    bwd_sig_static_pos = (
        np.unique(np.concatenate(static_bwd_parts)).astype(np.intp)
        if static_bwd_parts
        else np.zeros((0,), dtype=np.intp)
    )

    s_unit_slots = s_base_aug == 1
    if len(s_mat_r):
        s_unit_slots[s_flat_pos] = False
    s_unit_slots[s_car_flat_pos] = False

    conj_mask = np.zeros(s_nnz + sn, dtype=bool)
    conj_mask[:s_nnz] = (
        ((s_rows >= cn) & (s_rows < 2 * cn) & (s_cols >= cn) & (s_cols < 2 * cn))
        | ((s_rows >= 2 * cn) & (s_cols < cn))
        | ((s_rows >= cn) & (s_rows < 2 * cn) & (s_cols >= 2 * cn))
    )
    conj_mask[s_nnz + cn : s_nnz + 2 * cn] = True
    sweep_written_mask = np.zeros(s_nnz + sn, dtype=bool)
    if s_car_flat_pos.size > 0:
        sweep_written_mask[s_car_flat_pos] = True
    for g in signal_groups_sweep:
        sweep_written_mask[g.flat_pos] = True
    signal_static_conj_idx = np.flatnonzero(conj_mask & ~sweep_written_mask)
    signal_sweep_conj_idx = np.flatnonzero(conj_mask & sweep_written_mask)
    bwd_sig_conj_idx = bwd_map[np.flatnonzero(conj_mask & active_mask)]

    # --- Noise sparse COO & grouped updates ---
    if len(noise_arrays_raw["function_input_indices"]) > 0:
        n_inp = np.vstack(noise_arrays_raw["function_input_indices"]).astype(np.int32)
        n_out_r = np.concatenate(noise_arrays_raw["output_row_indices"]).astype(np.int32)
        n_out_c = np.concatenate(noise_arrays_raw["output_column_indices"]).astype(np.int32)
        n_mat_r = np.concatenate(noise_arrays_raw["system_matrix_row_indices"]).astype(np.int32)
        n_mat_c = np.concatenate(noise_arrays_raw["system_matrix_column_indices"]).astype(np.int32)

        _, first, inverse = np.unique(
            n_mat_r.astype(np.int64) * sn + n_mat_c, return_index=True, return_inverse=True
        )
        in_write_order = np.argsort(first)
        noise_rows = n_mat_r[first[in_write_order]].astype(np.intp)
        noise_cols = n_mat_c[first[in_write_order]].astype(np.intp)
        n_flat_pos = np.argsort(in_write_order)[inverse].astype(np.int32)
        noise_base_vals = (
            noise_matrix_np[noise_rows, noise_cols].astype(np.complex128)
            if noise_matrix_np is not None
            else np.zeros(len(noise_rows), dtype=np.complex128)
        )
        noise_groups_static, noise_groups_sweep = _group_instructions_np(
            n_inp, noise_arrays_raw["function_indices"], n_out_r, n_out_c, n_flat_pos,
            sweep_param_set=sig_ch_set, sweep_rows=sig_sweep_rows, static_sweep_set=car_ch_set,
        )
    else:
        noise_rows = np.zeros((0,), dtype=np.intp)
        noise_cols = np.zeros((0,), dtype=np.intp)
        noise_base_vals = np.zeros((0,), dtype=np.complex128)
        noise_groups_static, noise_groups_sweep = (), ()

    noise_grad_positions = np.concatenate((noise_rows, noise_cols))
    noise_grad_order = np.argsort(noise_grad_positions, kind="stable")
    noise_grad_indices, noise_grad_starts = np.unique(
        noise_grad_positions[noise_grad_order], return_index=True
    )

    # --- Linked parameter groups (resolved on host when possible; JAX fallback otherwise) ---
    linked_groups = {}
    host_linked_groups = ()
    if len(linked_indices_np) > 0:
        # Preserve original simulate.py write order: sorted by linking function index, stable within ties.
        link_order = np.argsort(linking_function_indices_np, kind="stable")
        ord_src = linked_indices_np[link_order]
        ord_fi = linking_function_indices_np[link_order]
        ord_dst = indices_to_link_np[link_order]
        # Keep only the final write to each destination, matching JAX .at[dst].set(vals).
        _, last_rev = np.unique(ord_dst[::-1], return_index=True)
        keep = np.sort(ord_dst.size - 1 - last_rev)
        ord_src = ord_src[keep]
        ord_fi = ord_fi[keep]
        ord_dst = ord_dst[keep]

        host_buckets: dict[tuple, tuple[list[int], list[int]]] = {}
        jax_buckets: dict[int, tuple[int, list[int], list[int]]] = {}
        all_host = True
        for src, f_i, dst in zip(ord_src, ord_fi, ord_dst):
            fi = int(f_i)
            fn = LINKING_FUNCTIONS[fi]
            pair_fns = resolve_host_link(fn)
            if pair_fns is None:
                all_host = False
            elif all_host:
                if pair_fns not in host_buckets:
                    host_buckets[pair_fns] = ([], [])
                host_buckets[pair_fns][0].append(int(src))
                host_buckets[pair_fns][1].append(int(dst))
            fn_id = id(fn)
            if fn_id not in jax_buckets:
                jax_buckets[fn_id] = (fi, [], [])
            jax_buckets[fn_id][1].append(int(src))
            jax_buckets[fn_id][2].append(int(dst))

        if all_host:
            host_linked_groups = tuple(
                (pair_fns[0], pair_fns[1], np.asarray(srcs, dtype=np.intp), np.asarray(dsts, dtype=np.intp))
                for pair_fns, (srcs, dsts) in host_buckets.items()
            )
        else:
            for fi, srcs, dsts in jax_buckets.values():
                linked_groups[fi] = (jnp.asarray(srcs, dtype=jnp.int32), jnp.asarray(dsts, dtype=jnp.int32))

    opt_param_idx_np = np.asarray(optimized_parameter_indices, dtype=np.intp)
    opt_val_idx_np = np.asarray(optimization_value_indices, dtype=np.intp)
    _, last_reversed = np.unique(opt_param_idx_np[::-1], return_index=True)
    opt_write_rows = np.sort(opt_param_idx_np.size - 1 - last_reversed)
    opt_param_idx_np = opt_param_idx_np[opt_write_rows]
    opt_val_idx_np = opt_val_idx_np[opt_write_rows]

    s_rows_i64 = s_rows.astype(np.intp)
    s_cols_i64 = s_cols.astype(np.intp)
    qhd_param_idx_p = np.asarray(qhd_parameter_indices_np, dtype=np.intp)
    signal_block_plan = plan_signal_blocks(sn, s_rows, s_cols, unit_slots=s_unit_slots[:s_nnz])
    carrier_csc_pattern = _CSCPattern(cn, c_indices, c_indptr)
    if signal_block_plan is None:
        signal_core_csc_pattern = _CSCPattern(sn, s_indices, s_indptr)
    else:
        signal_core_csc_pattern = _CSCPattern(
            signal_block_plan.core.size, signal_block_plan.rows, signal_block_plan.indptr
        )

    topology = SparseTopology(
        cn=cn,
        readout_ports=np.asarray(metadata[0], dtype=np.intp),
        sn=sn,
        nnz_c=c_nnz,
        nnz_s=s_nnz,
        nnz_n=len(noise_rows),
        num_detectors=noise_selection_vectors_np.shape[0],
        opt_param_idx=opt_param_idx_np,
        opt_val_idx=opt_val_idx_np,
        car_ch_idx=car_ch_idx_np,
        sig_ch_idx=sig_ch_idx_np,
        qhd_param_idx=qhd_param_idx_p,
        qhd_placing_idx=qhd_placing_indices_np,
        qhd_scatter_plan=build_scatter_plan(qhd_param_idx_p),
        host_linked_groups=host_linked_groups,
        carrier_base_aug=c_base_aug,
        carrier_csc_rows=c_rows.astype(np.intp),
        carrier_csc_cols=c_cols.astype(np.intp),
        carrier_csc_indices=c_indices,
        carrier_csc_indptr=c_indptr,
        carrier_csc_pattern=carrier_csc_pattern,
        carrier_groups=carrier_groups,
        signal_base_aug=s_base_aug,
        signal_csc_rows=s_rows_i64,
        signal_csc_cols=s_cols_i64,
        signal_csc_indices=s_indices,
        signal_csc_indptr=s_indptr,
        signal_core_csc_pattern=signal_core_csc_pattern,
        signal_block_plan=signal_block_plan,
        signal_groups_static=signal_groups_static,
        signal_groups_sweep=signal_groups_sweep,
        signal_carrier_indices=s_car_idx,
        signal_carrier_flat_pos=s_car_flat_pos,
        signal_carrier_scatter_plan=build_scatter_plan(s_car_idx),
        signal_static_conj_idx=signal_static_conj_idx,
        signal_sweep_conj_idx=signal_sweep_conj_idx,
        bwd_sig_n_act=bwd_n_act,
        bwd_sig_n_act_a=bwd_n_act_a,
        bwd_sig_a_rows=s_rows_i64[bwd_a_pos],
        bwd_sig_a_cols=s_cols_i64[bwd_a_pos],
        bwd_sig_rhs_rows=bwd_rhs_rows,
        bwd_sig_conj_idx=bwd_sig_conj_idx,
        bwd_sig_car_pos=bwd_sig_car_pos,
        bwd_sig_static_pos=bwd_sig_static_pos,
        noise_base_vals=noise_base_vals,
        noise_rows=noise_rows,
        noise_cols=noise_cols,
        noise_groups_static=noise_groups_static,
        noise_groups_sweep=noise_groups_sweep,
        noise_grad_order=noise_grad_order,
        noise_grad_starts=noise_grad_starts,
        noise_grad_indices=noise_grad_indices,
    )

    if len(optimized_parameter_indices) == 0:
        opt_params = jnp.asarray(np.empty(0, dtype=np.float64))
    else:
        opt_params = None

    simulation_arrays = {
        "optimized_parameters": opt_params,
        "parameters": jnp.asarray(parameters_np.real, dtype=jnp.float64),
        "carrier_changing_values": carrier_changing_values,
        "signal_changing_values": signal_changing_values,
        "noise_selection_vectors": jnp.asarray(noise_selection_vectors_np, dtype=jnp.complex128),
        "linked_groups": linked_groups,
        "topology": topology,
    }

    return simulation_arrays, metadata
