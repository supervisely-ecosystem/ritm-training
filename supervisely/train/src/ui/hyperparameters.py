import os
import supervisely as sly
import sly_globals as g


def init_general(state):
    state["epochs"] = 20
    state["gpusId"] = 0

    state["input_size"] = {
        "value": {
            "width": 512,
            "height": 512,
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
    state["batchSizePerGPU"] = 2
    state["workersPerGPU"] = 1
    state["checkpointInterval"] = 5
    state["visualizationInterval"] = 1
    state["maxNumPoints"] = 12

def init_optimizer(state):
    state["optimizer"] = "adam"
    state["stepLrGamma"] = 0.95
    state["lr"] = 1e-4
    state["beta1"] = 0.9
    state["beta2"] = 0.999
    state["weightDecay"] = 0
    state["amsgrad"] = False
    state["momentum"] = 0.9
    state["nesterov"] = False

def init_loss(data, state):
    data["availableLosses"] = ["SigmoidBinaryCrossEntropyLoss", "NormalizedFocalLossSigmoid", "FocalLoss", "SoftIoU"]
    state["instanceLoss"] = "SigmoidBinaryCrossEntropyLoss"
    state["instanceAuxLoss"] = "SigmoidBinaryCrossEntropyLoss"
    state["instanceLossWeight"] = 1.0
    state["instanceAuxLossWeight"] = 0.4

def init(data, state):
    init_general(state)
    init_optimizer(state)
    init_loss(data, state)

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