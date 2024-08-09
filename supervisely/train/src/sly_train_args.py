import sys
import sly_globals as g
import supervisely as sly


def init_script_arguments(state):
    sly.logger.debug(f"Initing script arguments with state: {state}")
    sys.argv = [sys.argv[0]]
    sys.argv.append(g.model_config_local_path)

    sys.argv.extend(["--workers", str(state["workersPerGPU"])])
    sys.argv.extend(["--batch-size", str(state["batchSizePerGPU"])])
    sys.argv.extend(["--weights", g.local_weights_path])

    if g.temp_model_path is not None:
        sys.argv.extend(["--temp-model-path", g.temp_model_path])
        sly.logger.debug(f"Added --temp-model-path argument with value {g.temp_model_path}")

    if state["continueTrain"]:
        sys.argv.extend(["--resume-exp", "exp"])
