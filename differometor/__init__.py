from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name: str):
    if name == "visualize_setup":
        from differometor.plot import visualize_setup
        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
