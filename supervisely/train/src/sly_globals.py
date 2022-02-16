import os
from pathlib import Path
import sys
import supervisely as sly


root_source_dir = str(Path(sys.argv[0]).parents[3])
print(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

models_source_dir = "/ritm_models"
print(f"Models source directory: {models_source_dir}")
sys.path.append(models_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
print(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
print(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)

models_configs_dir = os.path.join(root_source_dir, "configs")
print(f"Models configs directory: {models_configs_dir}")
sys.path.append(source_path)

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

# @TODO: for debug
sly.fs.clean_dir(my_app.data_dir)

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_dir = os.path.join(my_app.data_dir, "sly_project")
project_dir_seg = None

artifacts_dir = os.path.join(my_app.data_dir, "artifacts")
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)
visualizations_dir = os.path.join(artifacts_dir, "visualizations")
sly.fs.mkdir(visualizations_dir)

augs_config_path = os.path.join(info_dir, "augs_config.json")

local_weights_path = None
model_config_local_path = None

seg_project_meta = None