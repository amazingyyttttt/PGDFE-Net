import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mmrotate
import mmrotate.models.detectors.petdet
from PIL import Image
import torch
from torchvision import transforms
from mmdet.apis import init_detector, inference_detector

# 如果出现 OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 保存 hook activations 的字典
activation = {}

def get_activation(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook


def parse_args():
    parser = argparse.ArgumentParser(description='PETDet CAM visualization')
    parser.add_argument('--config',
                        default='/home/lucid/lwt/code/PETDet/configs/petdet/qopn_rcnn_r50_fpn_3x_shiprs3_le90.py', #/home/lucid/lwt/code/PETDet/configs/petdet/qopn_rcnn_r50_fpn_3x_shiprs3_le90.py  /home/lucid/lwt/code/PETDet/configs/petdet/pet_pcl.py
                        help='测试配置文件路径')
    parser.add_argument('--checkpoint',
                        default='/home/lucid/lwt/code/PETDet/baseline_dir/shipsr3/best_mAP_epoch_35.pth',#/home/lucid/lwt/code/PETDet/baseline_dir/shipsr3/best_mAP_epoch_35.pth   /home/lucid/lwt/code/PETDet/网络日志权重/shipsr3/best_mAP_epoch_35.pth
                        help='权重文件路径')
    parser.add_argument('--img',
                        default='/home/lucid/lwt/dataset/ShipRSImageNet_V1/ShipRSImageNet_V1/VOC_Format/JPEGImages/100001072.bmp',
                        help='输入图像路径')
    parser.add_argument('--device', default='cuda:0', help='运行设备，例如 cuda:0 或 cpu')
    parser.add_argument('--out-dir',
                        default='/home/lucid/lwt/code/PETDet/hotimages/petdet',
                        help='热图保存目录')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = args.device

    # 初始化检测模型
    model = init_detector(
        config=args.config,
        checkpoint=args.checkpoint,
        device=device
    )
    model.eval()
    model = model.to(device)
    print(model)
    # model.backbone.layer1[0].register_forward_hook(get_activation('bn3'))

    model.neck.lateral_convs[0].register_forward_hook(get_activation('conv'))

    # model.neck.wfr.convs3_list[0].register_forward_hook(get_activation('conv'))
    # model.neck.asm.fine_enhancement.register_forward_hook(get_activation('Conv2d'))
    # model.neck.fsbm[0].cat.register_forward_hook(get_activation('conv1'))
    # 读取并预处理图像
    img_path = args.img
    img = Image.open(img_path).convert('RGB')
    # inference_detector 内部会根据 test_pipeline 做预处理，这里直接传入路径

    # 用 MMDet 的 API 进行推理，会触发前向钩子
    _ = inference_detector(model, img_path)

    # 从 hook 中取出激活
    # bn3 = activation['bn3']  # Tensor, shape = [1, C, H, W]
    bn3 = activation['conv']  # Tensor, shape = [1, C, H, W]
    print(f"bn3 shape: {bn3.shape}")

    # 转为 numpy 并可视化第一个通道
    # bn3_np = bn3.cpu().numpy()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(bn3_np[0, 0], cmap='gray')
    # plt.axis('off')
    # plt.show()
    bn3_np = bn3.cpu().numpy()[0, 0]
    bn3_norm = (bn3_np - bn3_np.min()) / (bn3_np.max() - bn3_np.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(bn3_norm, cmap='jet')
    plt.axis('off')
    img_name = os.path.splitext(os.path.basename(args.img))[0]
    save_path = os.path.join(args.out_dir, f"{img_name}_cam.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()