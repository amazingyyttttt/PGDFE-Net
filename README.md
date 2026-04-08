# PGDFE-Net
《PGDFE-Net: Patch Guided Discriminative Feature Enhancement Network for Fine-Grained Remote Sensing Ship Detection》

Fine-grained remote sensing ship detection plays an important role in maritime traffic management and security guarantee system. Despite considerable research advances have been achieved via elaborately designed deep neural networks, the small proportion of objects in complex remote sensing scenarios easily causes the detailed information loss. Meanwhile, the scales of different ships are various and the inter-class differences between different categories are relatively small, which further reduces the fine-grained remote sensing ship detection accuracy. To address the above issues, a patch guided discriminative feature enhancement network is proposed. First, a patch-wise feature boosting and suppression (PFBS) module is proposed. The multi-scale patch level feature representations are enhanced via attention mechanisms, and more detailed information in the most prominent area and other potential local part of various ships is extracted simultaneously via dual path feature boosting and suppression. In this way, the classification accuracy of the objects in different subcategories with high similarity can be improved. Furthermore, a patch enhanced cross-scale feature fusion (PE-CFF) module is devised to select discriminative local patches in a coarse-to-fine manner. Through cross scales feature interaction and compensation, more contextual information of objects with various scales is explored for multi-scale feature aggregation. Finally, the prototypical contrastive learning loss is introduced to further enlarge the inter-class divergence while reinforcing the intra-class compactness for different types of ships. The qualitative and quantitative evaluations on ShipRSImageNet and HRSC2016 datasets demonstrate the superior effectiveness of the proposed network over other state-of-the-art approaches. Our code and models will be released at https://github.com/amazingyyttttt/PGDFE-Net.

<img width="5525" height="2456" alt="PGDFENet" src="https://github.com/user-attachments/assets/dceaa9db-7e9d-448e-8cc0-33aa8d8ea73b" />


Installation：https://github.com/canoe-Z/PETDet

主函数路径：PGDFENet/PGDFENet/tools/train.py

ShipRSImageNet权重：https://pan.quark.cn/s/2f25b2a63fd4

HRSC2016权重：https://pan.quark.cn/s/96bb90061c09
