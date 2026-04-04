# dataset settings
# from configs.petdet.qopn_rcnn_r50_fpn_3x_shiprs3_le90 import ship_classes

dataset_type = 'HRSCDataset'
data_root = '/home/lucid/lwt/dataset/HRSC2016/'#DOSR  #HRSC2016
# level = 3
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 800)), #1333, 800
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classwise=True,      #False
        ann_file=data_root + 'HRSC2016/ImageSets/trainval.txt',    #HRSC2016/ImageSets/trainval.txt    ImageSets/trainval.txt
        ann_subdir=data_root + 'HRSC2016/FullDataSet/Annotations/', #HRSC2016/FullDataSet/Annotations/   Annotations_5_parameters_version/
        img_subdir=data_root + 'HRSC2016/FullDataSet/AllImages/', #HRSC2016/FullDataSet/AllImages/   Images/
        img_prefix=data_root + 'HRSC2016/FullDataSet/AllImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classwise=True,      #False
        ann_file=data_root + 'HRSC2016/ImageSets/test.txt',       #HRSC2016/ImageSets/test.txt   ImageSets/test.txt
        ann_subdir=data_root + 'HRSC2016/FullDataSet/Annotations/',
        img_subdir=data_root + 'HRSC2016/FullDataSet/AllImages/',
        img_prefix=data_root + 'HRSC2016/FullDataSet/AllImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classwise=True,      #False
        ann_file=data_root + 'HRSC2016/ImageSets/test.txt',
        ann_subdir=data_root + 'HRSC2016/FullDataSet/Annotations/',
        img_subdir=data_root + 'HRSC2016/FullDataSet/AllImages/',
        img_prefix=data_root + 'HRSC2016/FullDataSet/AllImages/',
        pipeline=test_pipeline))
