import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import *
# AnchorGenerator,RPNHead,RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
import torchvision.models.detection._utils as det_utils

def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    """
    ONNX spec requires the k-value to be less than or equal to the number of inputs along
    provided dim. Certain models use the number of elements along a particular axis instead of K
    if K exceeds the number of elements along that axis. Previously, python's min() function was
    used to determine whether to use the provided k-value or the specified dim axis value.
    However in cases where the model is being exported in tracing mode, python min() is
    static causing the model to be traced incorrectly and eventually fail at the topk node.
    In order to avoid this situation, in tracing mode, torch.min() is used instead.
    Args:
        input (Tensor): The orignal input tensor.
        orig_kval (int): The provided k-value.
        axis(int): Axis along which we retreive the input size.
    Returns:
        min_kval (int): Appropriately selected k-value.
    """
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return min_kval

class GreedyNet(nn.Module):
    def __init__(self,im_size = 32,pre_nms_top_n = dict(training=200, testing=100),post_nms_top_n = dict(training=160, testing=80)):
        super().__init__()
        self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50',pretrained = False)
        self.anchor_generator = AnchorGenerator(
        sizes = tuple([16,32,64,128,256] for _ in range(5)),
        aspect_ratios=tuple([(0.75, 0.5, 1.25) for _ in range(5)]))
        self.head = RPNHead(256,self.anchor_generator.num_anchors_per_location()[0])

        self.box_coder = det_utils.BoxCoder(weights = (1.0,1.0,1.0,1.0))

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
    
        self.min_size = 1e-3
        self.score_thresh = 0.01
        self.nms_thresh = 0.7

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = _topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(self,ims):
        im = ims.tensors
        features = list(self.backbone(im).values())
        #for k in features:print(":",k.shape)
        objectness,pred_bbox_deltas = self.head(features)

        anchors = self.anchor_generator(ims,features) # a list of
        #for anchor in anchors:print(anchor.shape)
        num_images = len(anchors)

        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)    
        boxes, scores = self.filter_proposals(proposals, objectness, ims.image_sizes, num_anchors_per_level)
        return boxes,scores
    
im_size = 32
GN = GreedyNet(im_size)

inputs = torch.randn([3,3,im_size,im_size])
ims = ImageList(inputs,[(im_size,im_size) for _ in range(3)])

outputs,scores = GN(ims)

for b in outputs:print(b.shape)
for s in scores:print(s.shape)

