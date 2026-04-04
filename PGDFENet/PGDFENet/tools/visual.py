import argparse
import time

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
#from models.xxx import Model  # 加载自己的模型, 这里xxx是自己模型名字
# from mmdet.models.backbones.pvt import pvt_small
from mmrotate.models.detectors.petdet import PETDet
import os


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('--config', default='/home/lucid/lwt/code/PETDet/configs/petdet/pet_pcl.py',help='test config file path')
    parser.add_argument('--checkpoint',default='/home/lucid/lwt/code/PETDet/网络日志权重/shipsr3/best_mAP_epoch_35.pth', help='checkpoint file')
    args = parser.parse_args()
    return args


def main():
    device = torch.device('cuda:0')
    #img_path = '/root/autodl-tmp/visualize_image/vid_000086_frame0000049.jpg'
    img_path = '/home/lucid/lwt/dataset/ShipRSImageNet_V1/ShipRSImageNet_V1/VOC_Format/JPEGImages/000040.bmp'
    img = Image.open(img_path)
    imgarray = np.array(img) / 255.0

    # 将图片处理为可以预测的形式
    transform = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
    ])
    input_img = transform(img).unsqueeze(0)  # unsqueeze(0)用于升维
    print(input_img.shape)  # torch.Size([1, 3, 512, 512])

    # 定义钩子函数
    activation = {}  # 保存获取的输出

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook



    args = parse_args()

    cfg = Config.fromfile(args.config)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'))
    model = model.to(device)
    #model = PETDet().to(device)
    print(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    #model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # model.res.layer1[2].register_forward_hook(get_activation('bn3'))  #resnet50 layer1中第三个模块的bn3注册钩子
    # model.block1[2].mlp.se.register_forward_hook(get_activation('fc[3]'))
    model.backbone.layer1[2].register_forward_hook(get_activation('bn3'))
    input_img = input_img.to(device)  # cpu数据转一下gpu,这个看你会不会报错，我的不转会报错
    img_metas = input_img['img_metas'][0].input_img[0]
    out_total= model(input_img,img_metas)
#    B,H,W,C=out_total[0].shape
    bn3 = activation['bn3']  # 结果将保存在activation字典中  bn3输出<class 'torch.Tensor'>, tensor是无法用plt正常显示的
    print(bn3.shape)  # 调试到这里基本成功了

    # fc = fc.reshape(1, H, W, -1).permute(0, 3, 1, 2).contiguous()
    bn3 = bn3.cpu().numpy() # 转一下numpy,  shape:(1，256, 128, 128)
    plt.figure(figsize=(8,8))
    #plt.imshow(norm2[0][0], cmap='gray')  # bn3[0][0]  shape:(128, 128)
    plt.imshow(bn3[0][0])
    plt.axis('off')
    # # shape:(128, 128)
    #plt.savefig('D:/transformer/code/boosting_rcnn_ma/tools/visiualize_imagesresult/figure2.jpg')
    plt.show()
    #plt.savefig('D:/transformer/code/boosting_rcnn_ma/tools/visiualize_imagesresult/figure1.jpg')


if __name__ == '__main__':
    main()

