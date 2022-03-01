import supervisely as sly
import sly_globals as g
import os
import random
from collections import namedtuple

ItemInfo = namedtuple('ItemInfo', ['dataset_name', 'name', 'img_path', 'ann_path'])
train_set = None
val_set = None

items_to_ignore: dict = {}

train_set_path = os.path.join(g.my_app.data_dir, "train.json")
val_set_path = os.path.join(g.my_app.data_dir, "val.json")

def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    # RITM model requires to contain minimum 2 samples in batch
    if train_count < 2:
        train_count = 2
    elif project_info.items_count - train_count < 2:
        train_count = project_info.items_count - 2
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedImages"] = "train"
    state["splitInProgress"] = False
    state["trainImagesCount"] = None
    state["valImagesCount"] = None
    data["done3"] = False
    state["collapsed3"] = True
    state["disabled3"] = True


def refresh_table():
    global items_to_ignore
    ignored_items_count = sum([len(ds_items) for ds_items in items_to_ignore.values()])
    total_items_count = g.project_fs.total_items - ignored_items_count
    train_percent = 80
    train_count = int(total_items_count / 100 * train_percent)
    # RITM model requires to contain minimum 2 samples in batch
    if train_count < 2:
        train_count = 2
    elif g.project_info.items_count - train_count < 2:
        train_count = g.project_info.items_count - 2
    random_split_tab = {
        "count": {
            "total": total_items_count,
            "train": train_count,
            "val": total_items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    fields = [
        {'field': 'state.randomSplit', 'payload': random_split_tab},
        {'field': 'data.totalImagesCount', 'payload': total_items_count},
    ]
    g.api.app.set_fields(g.task_id, fields)



def get_train_val_splits_by_count(train_count, val_count):
    global items_to_ignore
    ignored_count = sum([len(ds_items) for ds_items in items_to_ignore.values()])
    
    if g.project_fs.total_items != train_count + val_count + ignored_count:
        raise ValueError("total_count != train_count + val_count + ignored_count")
    all_items = []
    for dataset in g.project_fs.datasets:
        for item_name in dataset:
            if item_name in items_to_ignore[dataset.name]:
                continue
            all_items.append(ItemInfo(dataset_name=dataset.name,
                                name=item_name,
                                img_path=dataset.get_img_path(item_name),
                                ann_path=dataset.get_ann_path(item_name)))
    random.shuffle(all_items)
    train_items = all_items[:train_count]
    val_items = all_items[train_count:]
    return train_items, val_items


def get_train_val_splits_by_tag(train_tag_name, val_tag_name, untagged="ignore"):
    global items_to_ignore
    untagged_actions = ["ignore", "train", "val"]
    if untagged not in untagged_actions:
        raise ValueError(f"Unknown untagged action {untagged}. Should be one of {untagged_actions}")

    train_items = []
    val_items = []
    for dataset in g.project_fs.datasets:
        for item_name in dataset:
            if item_name in items_to_ignore[dataset.name]:
                continue
            img_path, ann_path = dataset.get_item_paths(item_name)
            info = ItemInfo(dataset.name, item_name, img_path, ann_path)

            ann = sly.Annotation.load_json_file(ann_path, g.project_meta)
            if ann.img_tags.get(train_tag_name) is not None:
                train_items.append(info)
            if ann.img_tags.get(val_tag_name) is not None:
                val_items.append(info)
            if ann.img_tags.get(train_tag_name) is None and ann.img_tags.get(val_tag_name) is None:
                # untagged item
                if untagged == "ignore":
                    continue
                elif untagged == "train":
                    train_items.append(info)
                elif untagged == "val":
                    val_items.append(info)
    return train_items, val_items

def get_train_val_splits_by_dataset(train_datasets, val_datasets):
    def _add_items_to_list(datasets_names, items_list):
        global items_to_ignore
        for dataset_name in datasets_names:
            dataset = g.project_fs.datasets.get(dataset_name)
            if dataset is None:
                raise KeyError(f"Dataset '{dataset_name}' not found")
            for item_name in dataset:
                if item_name in items_to_ignore[dataset.name]:
                    continue
                img_path, ann_path = dataset.get_item_paths(item_name)
                info = ItemInfo(dataset.name, item_name, img_path, ann_path)
                items_list.append(info)

    train_items = []
    _add_items_to_list(train_datasets, train_items)
    val_items = []
    _add_items_to_list(val_datasets, val_items)
    return train_items, val_items


def get_train_val_sets(state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = get_train_val_splits_by_count(train_count, val_count)
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        train_set, val_set = get_train_val_splits_by_tag(train_tag_name, val_tag_name,
                                                                     add_untagged_to)
        return train_set, val_set
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = get_train_val_splits_by_dataset(train_datasets, val_datasets)
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    elif len(train_set) < 2:
        raise ValueError("Train set is not big enough, min size is 2.")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
    elif len(val_set) < 2:
        raise ValueError("Val set is not big enough, min size is 2.")


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        train_set, val_set = get_train_val_sets(state)

        verify_train_val_sets(train_set, val_set)
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        step_done = False
        raise e
    finally:
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": f"data.done3", "payload": step_done},
            {"field": f"state.trainImagesCount", "payload": None if train_set is None else len(train_set)},
            {"field": f"state.valImagesCount", "payload": None if val_set is None else len(val_set)},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsed4", "payload": False},
                {"field": "state.disabled4", "payload": False},
                {"field": "state.activeStep", "payload": 4},
            ])
        g.api.app.set_fields(g.task_id, fields)
    if train_set is not None:
        _save_set_to_json(train_set_path, train_set)
    if val_set is not None:
        _save_set_to_json(val_set_path, val_set)

def _save_set_to_json(save_path, items):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    res = []
    for item in items:
        res.append({
            "dataset_name": item.dataset_name,
            "item_name": item.name
        })
    sly.json.dump_json_file(res, save_path)