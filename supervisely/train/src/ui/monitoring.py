import supervisely as sly
from functools import partial
from sly_train_progress import init_progress, _update_progress_ui
from supervisely.app.v1.widgets.chart import Chart
import sly_globals as g
import workflow as w
import os
import sys
import shutil
from train import main as train_model
from sly_train_args import init_script_arguments

# it is needed to use losses
from isegm.model.losses import *
import torch

_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None
    state["isValidation"] = False

    init_charts(data, state)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False
    state["preparingData"] = False

    state["started"] = False
    state["finishTrain"] = False
    state["continueTrain"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None

    state["addEpochs"] = 5


def init_charts(data, state):
    state["smoothing"] = 0.6

    g.sly_charts = {
        "lr": Chart(
            g.task_id,
            g.api,
            "data.chartLR",
            title="LR",
            series_names=["LR"],
            ydecimals=9,
            xdecimals=2,
        ),
        "loss": Chart(
            g.task_id,
            g.api,
            "data.chartLoss",
            title="Train Loss",
            series_names=["train"],
            smoothing=state["smoothing"],
            ydecimals=8,
            xdecimals=2,
        ),
        "val_iou": Chart(
            g.task_id,
            g.api,
            "data.chartIoU",
            title="Val Adaptive IoU",
            series_names=["val"],
            smoothing=state["smoothing"],
            ydecimals=6,
            xdecimals=2,
        ),
    }

    for current_chart in g.sly_charts.values():
        current_chart.init_data(data)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress(task_type: str):
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        # Don't trust monitor.len
        # if progress.total == 0:
        #     progress.set(monitor.bytes_read, monitor.len, report=False)
        # else:
        progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    dir_size = sly.fs.get_directory_size(g.artifacts_dir)
    progress = sly.Progress(
        "Upload directory with training artifacts to Team Files", dir_size, is_size=True
    )
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    model_dir = g.sly_ritm.framework_folder
    remote_artifacts_dir = f"{model_dir}/{g.task_id}_{g.project_info.name}"
    remote_weights_dir = os.path.join(remote_artifacts_dir, g.sly_ritm.weights_folder)

    res_dir = g.api.file.upload_directory(
        g.team_id, g.artifacts_dir, remote_artifacts_dir, progress_size_cb=progress_cb
    )

    # generate metadata file
    g.ritm_generated_metadata = g.sly_ritm.generate_metadata(
        app_name=g.sly_ritm.app_name,
        task_id=g.task_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=g.sly_ritm.weights_ext,
        project_name=g.project_info.name,
        task_type=task_type,
        config_path=None,
    )
    print(f"✅ Training artifacts successfully uploaded to: {res_dir}")

    return res_dir


def init_cfg(state):
    if state["continueTrain"]:
        g.train_cfg["num_epochs"] += state["addEpochs"]
    else:
        g.train_cfg["num_epochs"] = state["epochs"]
    g.train_cfg["input_size"] = (
        state["input_size"]["value"]["height"],
        state["input_size"]["value"]["width"],
    )
    g.train_cfg["checkpoint_interval"] = state["checkpointInterval"]
    g.train_cfg["visualization_interval"] = state["visualizationInterval"]
    g.train_cfg["max_num_points"] = state["maxNumPoints"]
    g.train_cfg["optimizer"] = state["optimizer"]
    g.train_cfg["step_lr_gamma"] = state["stepLrGamma"]

    g.train_cfg["optim_params"] = {}
    g.train_cfg["optim_params"]["lr"] = state["lr"]
    g.train_cfg["optim_params"]["weight_decay"] = state["weightDecay"]
    if state["optimizer"] in ["adam", "adamw"]:
        g.train_cfg["optim_params"]["betas"] = (state["beta1"], state["beta2"])
        g.train_cfg["optim_params"]["amsgrad"] = state["amsgrad"]
    if state["optimizer"] == "sgd":
        g.train_cfg["optim_params"]["momentum"] = state["momentum"]
        g.train_cfg["optim_params"]["nesterov"] = state["nesterov"]

    g.train_cfg["instance_loss"] = getattr(sys.modules[__name__], state["instanceLoss"])
    g.train_cfg["instance_aux_loss"] = getattr(sys.modules[__name__], state["instanceAuxLoss"])
    g.train_cfg["instance_loss_weight"] = state["instanceLossWeight"]
    g.train_cfg["instance_aux_loss_weight"] = state["instanceAuxLossWeight"]
    g.segmentation_type = state["segmentationType"]
    g.resize_mode = state["resizeMode"]
    g.crop_to_aspect_ratio = state["cropToAspectRatio"]


def get_preprocessing_pipelines():
    import imgaug.augmenters as iaa
    import augs

    width, height = g.train_cfg["input_size"][1], g.train_cfg["input_size"][0]
    aspect_ratio = width / height

    if g.resize_mode == "Random Crop":
        resize_augs = [
            iaa.PadToFixedSize(width=width, height=height),
            iaa.CropToFixedSize(width=width, height=height),
        ]
    elif g.resize_mode in ["Whole Image", "Crop to fit an instance object"]:
        resize_augs = []
        if g.crop_to_aspect_ratio:
            resize_augs.append(iaa.CropToAspectRatio(aspect_ratio))
        resize_augs.append(iaa.Resize({"width": width, "height": height}))
    else:
        raise Exception(f"Invalid resize_mode: {g.resize_mode}")

    if augs.augs_config_path is not None:
        augs_config = sly.json.load_json_file(augs.augs_config_path)
        train_augs = sly.imgaug_utils.build_pipeline(
            augs_config["pipeline"], random_order=augs_config["random_order"]
        )
    else:
        train_augs = iaa.Sequential([])

    train_augs.extend(resize_augs)
    val_augs = iaa.Sequential(resize_augs)

    return train_augs, val_augs


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    if not state["finishTrain"] and not state["continueTrain"]:
        g.api.app.set_field(task_id, "state.preparingData", True)
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        if os.path.exists(g.project_dir_seg):
            shutil.rmtree(g.project_dir_seg)
        os.makedirs(g.project_dir_seg)
        sly.Project.to_segmentation_task(
            g.project_dir,
            g.project_dir_seg,
            target_classes=state["selectedClasses"],
            segmentation_type="instance",
        )
        g.project_seg = sly.Project(g.project_dir_seg, sly.OpenMode.READ)
        g.seg_project_meta = g.project_seg.meta
        g.api.app.set_field(task_id, "state.preparingData", False)

    if not state["finishTrain"]:
        try:
            init_cfg(state)
            init_script_arguments(state)
            train_model()
            torch.cuda.empty_cache()
            w.workflow_input(api, g.project_info, state)
            fields = [
                {"field": "state.started", "payload": False},
                {"field": "state.continueTrain", "payload": True},
            ]
            g.api.app.set_fields(g.task_id, fields)

            app_logger.info(
                "ℹ️ Please press 'Finish Training' button to upload training results to Team Files."
            )
            g.my_app.show_modal_window(
                "Training is finished, app is still running and you can preview predictions dynamics over time."
                "Please stop app manually to upload the training results to Team Files."
            )
        except Exception as e:
            torch.cuda.empty_cache()
            g.api.app.set_field(task_id, "state.started", False)
            raise e  # app will handle this error and show modal window

    else:
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        task_type = state.get("segmentationType", None)
        remote_dir = upload_artifacts_and_log_progress(task_type)
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)
        w.workflow_output(api, g.ritm_generated_metadata, state)
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)

        g.my_app.stop()
