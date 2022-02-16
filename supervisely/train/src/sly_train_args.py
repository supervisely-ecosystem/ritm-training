import sys
import os
import sly_globals as g


def init_script_arguments(state):
    sys.argv.append(os.path.join(g.root_source_dir, "models", "iter_mask_supervisely", "hrnet18_itermask_3p.py"))

    sys.argv.extend(["--exp-name", "tmp_exp"])
    sys.argv.extend(["--workers", "2"])
    sys.argv.extend(["--batch-size", "2"])
    sys.argv.extend(["--weights", os.path.join(g.models_source_dir, "coco_lvis_h18_itermask.pth")])