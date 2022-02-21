import sys
import os
import sly_globals as g


def init_script_arguments(state):
    sys.argv = [sys.argv[0]]
    sys.argv.append(g.model_config_local_path)

    sys.argv.extend(["--workers", str(state["workersPerGPU"])])
    sys.argv.extend(["--batch-size", str(state["batchSizePerGPU"])])
    sys.argv.extend(["--weights", g.local_weights_path])
    if state["continueTrain"]:
        sys.argv.extend(["--resume-exp", "exp"])