import supervisely as sly
from functools import partial
from sly_train_progress import init_progress, _update_progress_ui
from supervisely.app.v1.widgets.chart import Chart
import sly_globals as g
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
    state['finishTrain'] = False
    state["continueTrain"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None

    state["addEpochs"] = 5

def init_charts(data, state):
    state["smoothing"] = 0.6

    g.sly_charts = {
        'lr': Chart(g.task_id, g.api, "data.chartLR",
                                    title="LR", series_names=["LR"],
                                    ydecimals=9, xdecimals=2),
        'loss': Chart(g.task_id, g.api, "data.chartLoss",
                                      title="Train Loss", series_names=["train"],
                                      smoothing=state["smoothing"], ydecimals=8, xdecimals=2),
        'val_iou': Chart(g.task_id, g.api, "data.chartIoU",
                                        title="Val Adaptive UoI", series_names=["val"],
                                        smoothing=state["smoothing"], ydecimals=6, xdecimals=2)
    }

    for current_chart in g.sly_charts.values():
        current_chart.init_data(data)

def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)
    
    progress = sly.Progress("Upload directory with training artifacts to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    remote_dir = f"/RITM_training/{g.task_id}_{g.project_info.name}"
    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    
    return res_dir


def init_cfg(state):
    if state["continueTrain"]:
        g.train_cfg["num_epochs"] += state["addEpochs"]
    else:
        g.train_cfg["num_epochs"] = state["epochs"]
    g.train_cfg["input_size"] = (state["input_size"]["value"]["height"], state["input_size"]["value"]["width"])
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
    g.is_instance_segmentation = state["segmentationType"] == "instance"
    g.crop_objects = state["cropObjects"]
    sly.logger.debug(f'segmentationType={state["segmentationType"]}')


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
            g.project_dir, g.project_dir_seg,
            target_classes=state["selectedClasses"],
            segmentation_type='instance'
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

            fields = [
                {"field": "state.started", "payload": False},
                {"field": "state.continueTrain", "payload": True}
            ]
            g.api.app.set_fields(g.task_id, fields)

            g.my_app.show_modal_window(
            "Training is finished, app is still running and you can preview predictions dynamics over time."
            "Please stop app manually once you are finished with it.")
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

        remote_dir = upload_artifacts_and_log_progress()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)

        g.my_app.stop()
