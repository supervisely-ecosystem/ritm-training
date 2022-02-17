from isegm.utils.exp_imports.default import *
import sly_globals as g
import os

MODEL_NAME = 'hrnet18'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (460, 460)
    model_cfg.num_max_points = 12

    model = HRNetModel(width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    # loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = SuperviselyDataset(
        g.project_dir_seg,
        split='train',
        augmentator=train_augmentator,
        min_object_area=10,
        keep_background_prob=0.0,
        points_sampler=points_sampler
    )

    valset = SuperviselyDataset(
        g.project_dir_seg,
        split='val',
        augmentator=val_augmentator,
        min_object_area=10,
        points_sampler=points_sampler
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[60, 100, 140], gamma=0.1)

    # lr_scheduler = partial(torch.optim.lr_scheduler.CyclicLR,
    #                        cycle_momentum=False, step_size_up=50,
    #                        base_lr=5e-6, max_lr=1e-4)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=10,
                        image_dump_interval=10,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=200)
