_base_ = [
    '../_base_/datasets/shiprs3.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]#../_base_/datasets/shiprs3.py
pretrained = '/home/lucid/lwt/code/PETDet/swin_tiny_patch4_window7_224.pth'
angle_version = 'le90'
contrast_size = 512
model = dict(
    type='PETDet',
    backbone=dict(
    type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        # type='SwinTransformer',   #SwinTransformer
        # embed_dims=96,#96 128
        # depths=[2, 2, 6, 2],#2 2 6 2      2 2 18 2
        # num_heads=[3, 6, 12, 24],#3 6 12 24  4 8 16 32
        # window_size=7,
        # mlp_ratio=4,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.2,#0.2 0.5
        # patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        # with_cp=False,
        # convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(
        type='CGAFPN',  ##CGAFPN   FPN
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[96,192,384,768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    rpn_head=dict(
        type='QualityOrientedRPNHead',
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        scale_angle=False,
        use_fpn_feature=True,
        enable_sa=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.25),
        bbox_coder=dict(
            type='RotatedDistancePointBBoxCoder', angle_version=angle_version),
        loss_bbox=dict(type='PolyGIoULoss', loss_weight=0.25)),
    roi_head=dict(
        type='OBBContrastRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor' ,
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),  # extend_factor=(1.4, 1.2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64]),#featmap_strides=[4, 8, 16, 32]
        bbox_head=dict(
            type='ContrastRotatedConvFCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            contrast_out_channels=contrast_size,
            roi_feat_size=7,
            num_classes=50,#50 21
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss',
                           beta=1.0,
                           loss_weight=1.0),
            loss_contrast=dict(
                type='SupConProxyAnchorLoss',#LeaSupConProxyAnchorLoss
                class_num=50,#50 21
                size_contrast=contrast_size,
                stage=2,#2
                mrg=0,
                alpha=32,
                loss_weight=0.02),#0.08
        )),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='RotatedATSSAssigner',
                          topk=9,
                          iou_calculator=dict(type='RBboxOverlaps2D'),
                          ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler', #RRandomSampler   OBBCateBalanceSampler
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)), #(1024, 1024) (1333, 800)
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000, #2000
    warmup_ratio=1.0 / 2000,#1.0 / 2000
    step=[24, 33])

# optimizer = dict(lr=0.02)
optimizer = dict(
    # _delete_=True,
    # type='Adam',
    # lr=1e-4,
    # weight_decay=0.00001,
    # betas=(0.9, 0.999),
    type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001
) #  lr=0.005 #  SGD momentum=0.9, lr=0.02, momentum=0.9, weight_decay=0.0001
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=4)
evaluation = dict(interval=1, save_best='auto',metric='mAP')#save_best='auto'

###################################################################################
