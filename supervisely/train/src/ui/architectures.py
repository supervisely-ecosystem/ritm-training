import os
from pathlib import Path
import sly_globals as g
import supervisely as sly


def get_models_list():
    return [
        {
            "config": "hrnet18_itermask_3p.py",
            "weightsFile": "sbd_h18_itermask.pth",
            "model": "(SBD) HRNet18 IT-M",
            "dataset": "SBD",
            "sbd_NoC_85": "3.39",
            "sbd_NoC_90": "5.43",
            "pascalVOC_NoC_85": "2.51",
            "COCO_MVal_90": "4.39",
            "params": "38.8 MB",
        },
        {
            "config": "hrnet18s_itermask_3p.py",
            "weightsFile": "coco_lvis_h18s_itermask.pth",
            "model": "(COCO) HRNet18s IT-M ",
            "dataset": "COCO + LVIS",
            "sbd_NoC_85": "4.04",
            "sbd_NoC_90": "6.48",
            "pascalVOC_NoC_85": "2.57",
            "COCO_MVal_90": "3.33",
            "params": "16.5 MB",
        },
        {
            "config": "hrnet18_itermask_3p.py",
            "weightsFile": "coco_lvis_h18_itermask.pth",
            "model": "(COCO) HRNet18 IT-M",
            "dataset": "COCO + LVIS",
            "sbd_NoC_85": "3.80",
            "sbd_NoC_90": "6.06",
            "pascalVOC_NoC_85": "2.28",
            "COCO_MVal_90": "2.98",
            "params": "38.8 MB",
        },
        {
            "config": "hrnet32_itermask_3p.py",
            "weightsFile": "coco_lvis_h32_itermask.pth",
            "model": "(COCO) HRNet32 IT-M",
            "dataset": "COCO + LVIS",
            "sbd_NoC_85": "3.59",
            "sbd_NoC_90": "5.71",
            "pascalVOC_NoC_85": "2.57",
            "COCO_MVal_90": "2.97",
            "params": "119 MB",
        },
    ]


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "params", "title": "Params", "subtitle": None},
        {"key": "sbd_NoC_85", "title": "SBD NoC 85%", "subtitle": None},
        {"key": "sbd_NoC_90", "title": "SBD NoC 90%", "subtitle": None},
        {"key": "pascalVOC_NoC_85", "title": "Pascal VOC NoC 85%", "subtitle": None},
        {"key": "COCO_MVal_90", "title": "COCO MVal NoC 90%", "subtitle": None},
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def init(data, state):
    models = get_models_list()
    data["models"] = models
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "(COCO) HRNet32 IT-M"
    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True

    state["weightsPath"] = ""
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


@g.my_app.callback("select_model")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_model(api: sly.Api, task_id, context, state, app_logger):
    if state["weightsInitialization"] == "custom":
        weights_path_remote = state["weightsPath"]
        sly.logger.debug(f"App uses custom weights: {weights_path_remote}")
        if not weights_path_remote.endswith(".pth"):
            raise ValueError(
                f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                f"Supported: '.pth'"
            )

        # get architecture type from previous UI state
        weights_parent_directory = str(Path(weights_path_remote).parents[1])
        sly.logger.debug(f"weights_parent_directory: {weights_parent_directory}")

        items_in_parent_directory = api.file.listdir(g.team_id, weights_parent_directory)
        sly.logger.debug(f"items_in_parent_directory: {items_in_parent_directory}")

        # Find first *.py file in the items_in_parent_directory.
        model_config_remote_path = None
        for item in items_in_parent_directory:
            if item.endswith(".py"):
                model_config_remote_path = os.path.join(weights_parent_directory, item)
                sly.logger.debug(f"Found model config: {model_config_remote_path}")
                break
        if not model_config_remote_path:
            raise FileNotFoundError(f"Model config not found in {weights_parent_directory}")

        # Download model config to local directory
        file_name = sly.fs.get_file_name_with_ext(model_config_remote_path)
        model_config_local_path = os.path.join(
            g.root_models_dir,
            "iter_mask_supervisely",
            file_name,
        )

        api.file.download(g.team_id, model_config_remote_path, model_config_local_path)
        sly.logger.debug(f"Model config downloaded to: {model_config_local_path}")

        g.temp_model_path = model_config_local_path
        sly.logger.debug(f"Save model config path to globals: {g.temp_model_path}")

        prev_state_path_remote = os.path.join(weights_parent_directory, "info/ui_state.json")
        sly.logger.debug(f"prev_state_path_remote: {prev_state_path_remote}")

        prev_state_path = os.path.join(g.my_app.data_dir, "ui_state.json")
        api.file.download(g.team_id, prev_state_path_remote, prev_state_path)
        sly.logger.debug(f"Prev state downloaded to: {prev_state_path}")

        prev_state = sly.json.load_json_file(prev_state_path)
        api.task.set_field(g.task_id, "state.selectedModel", prev_state["selectedModel"])
        sly.logger.debug(f"Readed json file: {prev_state}")
        sly.logger.debug(f"Current state: {state}")

        g.local_weights_path = os.path.join(
            g.models_source_dir, sly.fs.get_file_name_with_ext(weights_path_remote)
        )
        api.file.download(g.team_id, weights_path_remote, g.local_weights_path)
        sly.logger.debug(f"Weights downloaded to: {g.local_weights_path}")

    else:
        weights_file = None
        config_file = None
        for model in get_models_list():
            if state["selectedModel"] == model["model"]:
                weights_file = model["weightsFile"]
                config_file = model["config"]
        g.local_weights_path = os.path.join(g.models_source_dir, weights_file)
        g.model_config_local_path = os.path.join(
            g.root_source_dir, "models", "iter_mask_supervisely", config_file
        )

    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]
    g.api.app.set_fields(g.task_id, fields)
