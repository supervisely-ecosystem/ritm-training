from isegm.utils.exp_imports.default import *
import sly_globals as g
import supervisely as sly
import augs
import imgaug.augmenters as iaa

MODEL_NAME = 'hrnet18s'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = g.train_cfg["input_size"]
    model_cfg.num_max_points = g.train_cfg["max_num_points"]

    model = HRNetModel(width=18, ocr_width=48, small=True, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5,
                       with_prev_mask=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18_SMALL)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    loss_cfg = edict()
    loss_cfg.instance_loss = g.train_cfg["instance_loss"]()
    loss_cfg.instance_loss_weight = g.train_cfg["instance_loss_weight"]
    loss_cfg.instance_aux_loss = g.train_cfg["instance_aux_loss"]()
    loss_cfg.instance_aux_loss_weight = g.train_cfg["instance_aux_loss_weight"]
    
    width, height = g.train_cfg["input_size"][1], g.train_cfg["input_size"][0]
    aspect_ratio = width/height

    if augs.augs_config_path is not None:
        augs_config = sly.json.load_json_file(augs.augs_config_path)
        train_augs = sly.imgaug_utils.build_pipeline(augs_config["pipeline"], random_order=augs_config["random_order"])
        train_augs.append(iaa.PadToAspectRatio(aspect_ratio))
        train_augs.append(iaa.Resize({"width": width, "height": height}))
    else:
        iaa.Sequential()
        train_augs = iaa.Sequential([
            iaa.PadToAspectRatio(aspect_ratio),
            iaa.Resize({"width": width, "height": height}),
        ], random_order=False)
    val_augs = iaa.Sequential([
            iaa.PadToAspectRatio(aspect_ratio),
            iaa.Resize({"width": width, "height": height}),
    ], random_order=False)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)
    
    dataset_cls = InstanceSegmentationDataset if g.is_instance_segmentation else SuperviselyDataset
    
    trainset = dataset_cls(
        g.project_dir_seg,
        split='train',
        augmentator=train_augs,
        min_object_area=10,
        keep_background_prob=0.0,
        points_sampler=points_sampler
    )

    valset = dataset_cls(
        g.project_dir_seg,
        split='val',
        augmentator=val_augs,
        min_object_area=10,
        points_sampler=points_sampler
    )

    if g.is_instance_segmentation:
        trainset.crop_to_object = g.crop_objects
        valset.crop_to_object = g.crop_objects

    optimizer_params = g.train_cfg["optim_params"]

    lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, gamma=g.train_cfg["step_lr_gamma"], step_size=1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer=g.train_cfg["optimizer"],
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=g.train_cfg["checkpoint_interval"],
                        image_dump_interval=g.train_cfg["visualization_interval"],
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=5)
    trainer.run(num_epochs=g.train_cfg["num_epochs"])
