import os
import shutil
import numpy as np
import supervisely as sly
from isegm.data.datasets.supervisely import SuperviselyDataset, InstanceSegmentationDataset
from isegm.data.points_sampler import MultiPointSampler
import sly_globals as g
import ui
import imgaug.augmenters as iaa

# if os.path.exists("app_dir"):
#     shutil.rmtree("app_dir")

# sly.download_project(g.api, g.project_id, g.project_dir, cache=g.my_app.cache, save_image_info=True)

# sly.Project.to_segmentation_task(
#             g.project_dir, g.project_dir_seg,
#             segmentation_type='instance'
#         )

g.project_seg = sly.Project(g.project_dir_seg, sly.OpenMode.READ)
g.seg_project_meta = g.project_seg.meta


tw, th = 400,200
train_augs = iaa.Sequential([
            # iaa.PadToFixedSize(width=512, height=512),
            # iaa.CropToFixedSize(width=512, height=512),
            # iaa.Resize({"longer-side":64, "shorter-side":"keep-aspect-ratio"})
            iaa.PadToAspectRatio(tw/th),
            iaa.Resize({"width":tw, "height":th})
        ], random_order=False)

img = sly.image.read("b.png")
import cv2
img = cv2.resize(img, (200,300))
sly.image.write("b2.png", img)
img2 = train_augs(image=img)
sly.image.write("b3.png", img2)

points_sampler = MultiPointSampler(20, prob_gamma=0.80,
                                    merge_objects_prob=0.15,
                                    max_num_merged_objects=2)

trainset = InstanceSegmentationDataset(
    g.project_dir_seg,
    split='train',
    crop_to_object=True,
    augmentator=train_augs,
    min_object_area=10,
    keep_background_prob=0.0,
    points_sampler=points_sampler
)

for i in range(len(trainset)):
    s = trainset.get_sample(i)

os.makedirs("test", exist_ok=True)
sly.image.write("test/image.png", s.image)
sly.image.write("test/_encoded_masks.png", s._encoded_masks*255)

s2 = trainset.augment_sample(s)
trainset.points_sampler.sample_object(s2)
mask = trainset.points_sampler.selected_mask

sly.image.write("test/image2.png", s2.image)
sly.image.write("test/_encoded_masks2.png", s2._encoded_masks*255)
sly.image.write("test/mask.png", mask[0].astype(bool)*255)
