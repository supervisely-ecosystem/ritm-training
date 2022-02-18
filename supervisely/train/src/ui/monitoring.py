import supervisely as sly
from functools import partial
from sly_train_progress import init_progress, _update_progress_ui
import sly_globals as g
import os
import shutil
from train import main as train_model
from sly_train_args import init_script_arguments

_open_lnk_name = "open_app.lnk"

def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None
    state["isValidation"] = False

    # init_charts(data, state)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False

    state["started"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None

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

@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    # init_class_charts_series(state)
    try:
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))

        # TODO: change classes to selected
        classes_json = g.project_meta.obj_classes.to_json()
        classes = [obj["title"] for obj in classes_json]
        os.makedirs(g.project_dir_seg, exist_ok=True)
        sly.Project.to_segmentation_task(
            g.project_dir, g.project_dir_seg,
            target_classes=classes,
            segmentation_type='instance'
        )
        shutil.rmtree(g.project_dir)
        g.project_seg = sly.Project(g.project_dir_seg, sly.OpenMode.READ)
        g.seg_project_meta = g.project_seg.meta

        init_script_arguments(state)
        cfg = train_model()
        # shutil.copyfile(os.path.join(cfg.CHECKPOINTS_PATH, ""), "")

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
    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.stop()