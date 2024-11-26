# This module contains functions that are used to configure the input and output of the workflow for the current app,
# and versioning feature that creates a project version before the task starts.


import supervisely as sly

from supervisely.api.file_api import FileInfo


def workflow_input(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    state: dict,
):
    try:
        project_version_id = api.project.version.create(
            project_info,
            f"Train RITM",
            f"This backup was created automatically by Supervisely before the Train RITM task with ID: {api.task_id}",
        )
    except Exception as e:
        sly.logger.warning(f"Failed to create a project version: {repr(e)}")
        project_version_id = None

    try:
        if project_version_id is None:
            project_version_id = (
                project_info.version.get("id", None) if project_info.version else None
            )
        api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
        if state["weightsPath"]:
            file_info = api.file.get_info_by_path(project_info.team_id, state["weightsPath"])
            api.app.workflow.add_input_file(file_info, model_weight=True)
        sly.logger.debug(
            f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}"
        )
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(
    api: sly.Api,
    mmdet_generated_metadata: dict,
    state: dict,
):
    try:
        checkpoints_list = mmdet_generated_metadata.get("checkpoints", [])
        if len(checkpoints_list) == 0:
            sly.logger.debug("Workflow Output: No checkpoints found. Cannot set workflow output.")
            best_filename_info = None
        else:
            best_checkpoints = []
            new_infos = [FileInfo(*checkpoint) for checkpoint in checkpoints_list]
            for info in new_infos:
                best_checkpoints.append(info) if "best" in info.name else None
            if len(best_checkpoints) > 1:
                best_filename_info = sorted(best_checkpoints, key=lambda x: x.name, reverse=True)[0]
            else:
                best_filename_info = best_checkpoints[0]

        module_id = api.task.get_info_by_id(api.task_id).get("meta", {}).get("app", {}).get("id")

        if state["weightsInitialization"] == "custom":
            node_custom_title = "Train Custom Model"
        else:
            node_custom_title = None
        node_settings = sly.WorkflowSettings(
            title=node_custom_title,
            url=(
                f"/apps/{module_id}/sessions/{api.task_id}"
                if module_id
                else f"apps/sessions/{api.task_id}"
            ),
            url_title="Show Results",
        )
        if best_filename_info:
            relation_settings = sly.WorkflowSettings(
                title="Checkpoints",
                icon="folder",
                icon_color="#FFA500",
                icon_bg_color="#FFE8BE",
                url=f"/files/{best_filename_info.id}/true",
                url_title="Open Folder",
            )
            meta = sly.WorkflowMeta(relation_settings, node_settings)
            api.app.workflow.add_output_file(best_filename_info, model_weight=True, meta=meta)
            sly.logger.debug(
                f"Workflow Output: Node custom title - {node_custom_title}, Best filename - {best_filename_info}"
            )
            sly.logger.debug(f"Workflow Output: Meta \n    {meta.as_dict}")
        else:
            sly.logger.debug(
                f"File {best_filename_info} not found in Team Files. Cannot set workflow output."
            )

    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
