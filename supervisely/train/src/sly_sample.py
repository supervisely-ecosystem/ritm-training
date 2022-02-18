from isegm.data.sample import DSample
import imgaug.augmenters as iaa
import supervisely as sly

class Sample(DSample):
    def __init__(self, image, encoded_masks, objects=None,
                 objects_ids=None, ignore_ids=None, sample_id=None):
        super(Sample, self).__init__(image, encoded_masks, objects=objects,
                 objects_ids=objects_ids, ignore_ids=ignore_ids, sample_id=sample_id)

    def augment(self, augs: iaa.Sequential):
        self.reset_augmentation()
        res_im, res_mask = sly.imgaug_utils.apply_to_image_and_mask(augs, self.image, self._encoded_masks)

        self.image = res_im.copy()
        self._encoded_masks = res_mask.copy()

        self._compute_objects_areas()
        self.remove_small_objects(min_area=1)

        self._augmented = True