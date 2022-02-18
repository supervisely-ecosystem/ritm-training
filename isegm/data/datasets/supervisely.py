from pathlib import Path

import cv2
import numpy as np
import os
import json

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

import sly_globals as g
import splits


class SuperviselyDataset(ISDataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.0, **kwargs):
        super(SuperviselyDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.ds_ind_to_name = {}
        for ds_ind, dataset in enumerate(g.project_seg.datasets):
            self.ds_ind_to_name[ds_ind] = dataset.name
        self._images_path = 'img'
        self._insts_path = 'seg'
        # self._buggy_objects = dict()
        # self._buggy_mask_thresh = buggy_mask_thresh
        classes_json = g.seg_project_meta.obj_classes.to_json()
        classes_json = [obj for obj in classes_json if obj['title'] != '__bg__']
        self.palette = [obj["color"].lstrip('#') for obj in classes_json]
        self.palette = [[int(color[i:i + 2], 16) for i in (0, 2, 4)] for color in self.palette] # hex to rgb
        self.dataset_samples = self.get_items_by_set_path(os.path.join(g.my_app.data_dir, f'{split}.json'))

    def get_samples_number(self):
        return sum([len(items) for dataset, items in self.dataset_samples.items()])

    def get_items_by_set_path(self, set_path):
        files_by_datasets = {}
        with open(set_path, 'r') as set_file:
            set_list = json.load(set_file)

            for row in set_list:
                existing_items = files_by_datasets.get(row['dataset_name'], [])
                existing_items.append(row['item_name'])
                files_by_datasets[row['dataset_name']] = existing_items

        return files_by_datasets

    def index_to_ds_and_file(self, index):
        passed_files = 0
        for ds_name, items in self.dataset_samples.items():
            if index >= len(items) + passed_files:
                passed_files += len(items)
                continue
            ind_in_dataset = index - passed_files
            item_name = self.dataset_samples[ds_name][ind_in_dataset]
            return ds_name, item_name


    def get_sample(self, index):
        dataset_name, image_name = self.index_to_ds_and_file(index)

        image_path = str(self.dataset_path / dataset_name / self._images_path / image_name)
        inst_info_path = str(self.dataset_path / dataset_name / self._insts_path / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(inst_info_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2RGB)
        result_mask = np.zeros((instances_mask.shape[0], instances_mask.shape[1]), dtype=np.int32)
        # human masks to machine masks
        for color_idx, color in enumerate(self.palette):
            colormap = np.where(np.all(instances_mask == color, axis=-1))
            result_mask[colormap] = color_idx + 1

        # result_mask = self.remove_buggy_masks(index, result_mask)
        instances_ids, _ = get_labels_with_sizes(result_mask)

        return DSample(image, result_mask, objects_ids=instances_ids, sample_id=index)

    """
    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask
    """
