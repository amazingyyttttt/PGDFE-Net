# tools/print_cgafpn.py
import os
import sys

# --------------------------
# 动态加入项目根目录（假设本脚本位于 tools/ 下）
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------

from mmrotate.models.necks.cga_fpn import CGAFPN
from torchinfo import summary


def main():
    # 配置 CGAFPN 的构造参数，需与 cga_fpn.py 中 __init__ 签名保持一致
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256
    num_outs = 5
    start_level = 0
    end_level = -1
    add_extra_convs = 'on_output'
    relu_before_extra_convs = False
    no_norm_on_lateral = False
    conv_cfg = None
    norm_cfg = None
    act_cfg = None
    upsample_cfg = dict(mode='nearest')
    init_cfg = dict(type='Xavier', layer='Conv2d', distribution='uniform')

    # 实例化 CGAFPN
    cgafpn = CGAFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=num_outs,
        start_level=start_level,
        end_level=end_level,
        add_extra_convs=add_extra_convs,
        relu_before_extra_convs=relu_before_extra_convs,
        no_norm_on_lateral=no_norm_on_lateral,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        upsample_cfg=upsample_cfg,
        init_cfg=init_cfg,
    )

    # 打印完整模块列表（不被截断）
    print("=== CGAFPN 全模块列表 ===")
    for name, module in cgafpn.named_modules():
        print(f"{name or '[root]':30} -> {module.__class__.__name__}")

    # 如果需要查看各层输入/输出尺寸、参数量，也可使用 torchinfo.summary
    print("\n=== 使用 torchinfo.summary 打印结构 ===")
    feature_sizes = [
        (2, 256, 256, 256),
        (2, 512, 128, 128),
        (2, 1024, 64, 64),
        (2, 2048, 32, 32),
    ]
    summary(
        cgafpn,
        input_size=feature_sizes,
        depth=None,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        row_settings=("depth",),
    )


if __name__ == "__main__":
    main()