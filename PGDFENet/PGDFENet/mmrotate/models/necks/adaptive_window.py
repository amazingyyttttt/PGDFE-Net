# import torch
# import torch.nn.functional as F
#
#
# def preprocess_input(x, H, W):
#     # 1. 计算需要填充的高度和宽度
#     pad_h = (8 - H % 8) % 8
#     pad_w = (32 - W % 32) % 32
#     # 对输入进行填充
#     x_padded = F.pad(x, (0, pad_w, 0, pad_h))  # 右下方填充
#     H_pad, W_pad = H + pad_h, W + pad_w
#
#     # 2. 自适应调整窗口大小
#     window_size = (H_pad // 8, W_pad // 32) if H_pad > W_pad else (H_pad // 32, W_pad // 8)
#
#     # 3. 计算滑动窗口偏移量
#     shift_size = (window_size[0] // 2, window_size[1] // 2)
#
#     return x_padded, window_size, shift_size