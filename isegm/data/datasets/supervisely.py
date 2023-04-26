from pathlib import Path

import cv2
import numpy as np
import os
import json

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
import supervisely as sly
from sly_sample import Sample

import sly_globals as g
import splits


class SuperviselyDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(SuperviselyDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = 'img'
        self._insts_path = 'seg'
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

        instances_ids, _ = get_labels_with_sizes(result_mask)

        return Sample(image, result_mask, objects_ids=instances_ids, sample_id=index)


class InstanceSegmentationDataset(ISDataset):
    def __init__(self, dataset_path, split='train', crop_to_object=True, **kwargs):
        super(InstanceSegmentationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.sly_project : sly.Project = g.project_seg
        split_json = sly.json.load_json_file(os.path.join(g.my_app.data_dir, f'{split}.json'))
        self.dataset_samples_dict = self.init_dataset_samples(split_json)
        self.dataset_samples = list(zip(*self.dataset_samples_dict.values()))

        self.crop_to_object = crop_to_object
        self.crop_scale_range = [1.0, 1.5]

    def init_dataset_samples(self, split_json):
        items = {"dataset_name":[], "item_name":[], "mask":[], "img_path":[]}
        for item in split_json:
            ds : sly.Dataset = self.sly_project.datasets.get(item["dataset_name"])
            ann = ds.get_ann(item["item_name"], g.seg_project_meta)
            for l in ann.labels:
                items['dataset_name'].append(item['dataset_name'])
                items['item_name'].append(item['item_name'])
                items['mask'].append(l.geometry)
                items['img_path'].append(ds.get_img_path(item['item_name']))
        return items

    def get_sample(self, index):
        dataset_name, item_name, bitmap, image_path = self.dataset_samples[index]
        bitmap : sly.Bitmap
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        if self.crop_to_object:
            rect = bitmap.to_bbox()
            c = np.array([rect.center.col, rect.center.row, rect.center.col, rect.center.row])
            xyxy = np.array([rect.left, rect.top, rect.right, rect.bottom])
            ox, oy = xyxy[:2]  # origin
            scales = np.random.uniform(*self.crop_scale_range, size=4)
            xyxy_scaled = (xyxy - c) * scales + c
            xyxy_scaled[::2] = np.clip(xyxy_scaled[::2], 0, img_w)
            xyxy_scaled[1::2] = np.clip(xyxy_scaled[1::2], 0, img_h)
            l,t,r,b = np.round(xyxy_scaled).astype(int)
            rect_scaled = sly.Rectangle(t,l,b,r)
            h,w = rect_scaled.height, rect_scaled.width
            # crop image to object
            image = image[t:b+1, l:r+1]
            # draw object in mask
            result_mask = np.zeros((h,w), dtype=np.int32)
            mask = bitmap.data
            result_mask[oy-t:rect.bottom-t+1, ox-l:rect.right-l+1] = mask
            assert image.shape[:2] == result_mask.shape
        else:
            # uncrop bitmap
            result_mask = np.zeros((img_h,img_w), dtype=np.int32)
            rect = bitmap.to_bbox()
            l,t,r,b = [rect.left, rect.top, rect.right, rect.bottom]
            result_mask[t:b+1,l:r+1] = bitmap.data

        instances_ids, _ = get_labels_with_sizes(result_mask)

        return Sample(image, result_mask, objects_ids=instances_ids, sample_id=index)

