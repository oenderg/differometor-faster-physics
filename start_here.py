"""One complete optical setup simulation and gradient. Run: python start_here.py"""

import json
import os
from pathlib import Path
from time import perf_counter

setup_start = perf_counter()
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces

jax.config.update("jax_enable_compilation_cache", False)

# 1. Load the optical setup and all 101 reference frequencies.
data = Path(__file__).parent / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
setup = df.Setup.from_data(fixture)
frequencies = jnp.asarray(reference["frequencies"])
reference_sensitivity = jnp.asarray(reference["sensitivities"])

# 2. Select the 193 design variables and read their saved physical values.
properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
pairs = constrain_inter_grid_cell_spaces(setup.parameters, properties)
components = fixture["nodes"] | fixture["edges"]
values = []
for pair in pairs:
    # Coupled grid lengths share a value, so use the first member of the group.
    name, property_name = pair[0] if isinstance(pair[0], list) else pair
    values.append(components[name]["properties"][property_name])
parameters = jnp.asarray(values, dtype=jnp.float64)
arrays, detector_ports, *_ = df.run_build_step(
    setup, [("f", "frequency")], frequencies, pairs
)
detector_ports = jnp.asarray(detector_ports)
jax.block_until_ready((arrays, parameters, detector_ports, reference_sensitivity))
setup_seconds = perf_counter() - setup_start
print(f"Setup: {setup_seconds:.3f} s", flush=True)


# 3. Define one full simulation and the scalar quantity to differentiate.
def evaluate(parameters, arrays, detector_ports, reference_sensitivity):
    carrier, signal, noise = df.simulate(
        **dict(arrays, optimized_parameters=parameters)
    )
    # This saved optical setup has one signal detector.
    detector_signal = df.signal_detector(carrier, signal)[detector_ports].squeeze()
    sensitivity = noise / jnp.abs(detector_signal)
    powers = df.power_detector(carrier)
    loss = jnp.mean(jnp.log10(sensitivity / reference_sensitivity))
    return loss, (carrier, signal, noise, sensitivity, powers)


# 4. Compile the simulation and its gradient together.
# has_aux=True keeps the physical outputs alongside the scalar loss.
inputs = (parameters, arrays, detector_ports, reference_sensitivity)
start = perf_counter()
simulation_and_gradient = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
compiled = simulation_and_gradient.lower(*inputs).compile()
compile_seconds = perf_counter() - start
print(f"Compilation: {compile_seconds:.3f} s", flush=True)

# 5. Run once. Waiting is essential: JAX can otherwise return before it finishes.
start = perf_counter()
(loss, outputs), gradient = jax.block_until_ready(compiled(*inputs))
runtime_seconds = perf_counter() - start
print(f"Simulation + sensitivity + gradient: {runtime_seconds:.3f} s")
print(f"Total: {setup_seconds + compile_seconds + runtime_seconds:.3f} s")
print(f"Frequencies: {len(frequencies)}; gradient values: {gradient.size}")
print(f"Mean log sensitivity relative to reference: {float(loss):.3e}")
