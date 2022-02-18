import sys
import os
import sly_globals as g


def init_script_arguments(state):
    sys.argv.append(g.model_config_local_path)

    sys.argv.extend(["--workers", "8"])
    sys.argv.extend(["--batch-size", "8"])
    sys.argv.extend(["--weights", g.local_weights_path])