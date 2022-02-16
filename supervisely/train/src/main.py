import supervisely as sly
import sly_globals as g
import os
import random
import shutil
from train import main as train_model
from sly_train_args import init_script_arguments

def main():

    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    sly.download_project(g.api, g.project_id, g.project_dir,
                         cache=g.my_app.cache, save_image_info=False)
    g.project_dir_seg = os.path.join(g.my_app.data_dir, g.project_info.name + "_seg")

    sly.fs.mkdir(g.project_dir_seg)

    sly.Project.to_segmentation_task(
        g.project_dir, g.project_dir_seg,
        target_classes=["heart", "inner", "all", "obj"],
        segmentation_type='instance'
    )
    shutil.rmtree(g.project_dir)
    project_seg = sly.Project(g.project_dir_seg, sly.OpenMode.READ)
    g.seg_project_meta = project_seg.meta

    image_filenames = os.listdir(os.path.join(g.project_dir_seg, 'ds0', 'img'))
    random.shuffle(image_filenames)
    train_split = image_filenames[:14]
    val_split = image_filenames[14:]
    with open(os.path.join(g.project_dir_seg,  'train.txt'), 'w') as f:
        for item in train_split:
            f.write(f"{item}\n")

    with open(os.path.join(g.project_dir_seg, 'val.txt'), 'w') as f:
        for item in val_split:
            f.write(f"{item}\n")

    init_script_arguments(state)
    train_model()


    # ui.init(data, state)  # init data for UI widgets
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)