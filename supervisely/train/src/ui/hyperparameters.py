import os
import supervisely as sly
import sly_globals as g


def init_general(state):
    state["epochs"] = 150
    state["gpusId"] = 0

    state["valInterval"] = 1
    state["logConfigInterval"] = 5
    state["input_size"] = {
        "value": {
            "width": 460,
            "height": 460,
            "proportional": True
        },
        "options": {
            "proportions": {
                "width": 100,
                "height": 100
            },
            "min": 64
        }
    }
    state["batchSizePerGPU"] = 4
    state["workersPerGPU"] = 2

def init(data, state):
    init_general(state)

    state["currentTab"] = "general"
    state["collapsed6"] = True
    state["disabled6"] = True
    state["done6"] = False


def restart(data, state):
    data["done6"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)