
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import torch

anchor_generator = AnchorGenerator(sizes=tuple([(16, 32, 64, 128, 256) for _ in range(5)]), # let num of tuple equal to num of feature maps
                                  aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)])) # ref: https://github.com/pytorch/vision/issues/978
rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])


RPNNet = RegionProposalNetwork(
    anchor_generator= anchor_generator, head= rpn_head,
    fg_iou_thresh= 0.7, bg_iou_thresh=0.3,
    batch_size_per_image=48, # use fewer proposals
    positive_fraction = 0.5,
    pre_nms_top_n=dict(training=200, testing=100),
    post_nms_top_n=dict(training=160, testing=80),
    nms_thresh = 0.7
)

inputs = torch.randn([10,3,64,64])
outputs = RPNNet(inputs)
print(outputs.shape)