import os
import argparse
import importlib.util

import torch
from isegm.utils.exp import init_experiment
import supervisely as sly


def main():
    args = parse_args()
    sly.logger.debug(f"Arguments parsed: {args}")
    if args.temp_model_path:
        sly.logger.debug(f"Arguments contain temp_model_path field, value: {args.temp_model_path}")
        model_script = load_module(args.temp_model_path)
    else:
        sly.logger.debug("Arguments do not contain temp_model_path field...")
        model_path = args.model_path
        sly.logger.debug(
            f"Model path from arguments: {model_path}, type of model_path: {type(model_path)}"
        )
        if model_path is None or model_path == "" or model_path == "None":
            sly.logger.warning(
                f"Argument model_path is empty in args. Will try to read custom_model_path: {args.custom_model_path}"
            )
            args.model_path = args.custom_model_path

        model_script = load_module(args.model_path)

    model_base_name = getattr(model_script, "MODEL_NAME", None)

    # TODO: dont forget
    args.distributed = "WORLD_SIZE" in os.environ
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")

    model_script.main(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str, help="Path to the model script.")
    parser.add_argument(
        "--custom-model-path", type=str, help="Path to the custom model script.", default=""
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Here you can specify the name of the experiment. "
        "It will be added as a suffix to the experiment folder.",
    )

    parser.add_argument("--workers", type=int, default=4, metavar="N", help="Dataloader threads.")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="You can override model batch size by specify positive number.",
    )

    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of GPUs. "
        'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
        'You should use either this argument or "--gpus".',
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        required=False,
        help='Ids of used GPUs. You should use either this argument or "--ngpus".',
    )

    parser.add_argument("--resume-exp", type=str, default=None)

    parser.add_argument(
        "--resume-prefix",
        type=str,
        default="latest",
        help="The prefix of the name of the checkpoint to be loaded.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="The number of the starting epoch from which training will continue. "
        "(it is important for correct logging and learning rate)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model weights will be loaded from the specified path if you use this argument.",
    )

    parser.add_argument(
        "--temp-model-path",
        type=str,
        default="",
        help="Do not use this argument (for internal purposes).",
    )

    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


def load_module(script_path):
    sly.logger.info(f"Loading model script from {script_path}...")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Model script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == "__main__":
    main()
