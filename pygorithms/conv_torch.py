"""
    Attempt on yolo with pytorch.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Dict, List, Tuple, Optional


def parse_yolo_cfg(cfg: str) -> List[Dict]:
    blocks = []
    b = {}
    with open(cfg, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().lstrip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                if b:
                    blocks.append(b)
                    b = {}
                b["type"] = line[1:-1]
            else:
                key, val = line.split("=")
                b[key.rstrip()] = val.lstrip()
        blocks.append(b)

    return blocks


def create_modules(blocks: List[Dict]) -> Tuple[Dict, nn.ModuleList]:
    network_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, b in enumerate(blocks[1:]):
        module = nn.Sequential()

        if b["type"] == "convolutional":
            activation = b["activation"]
            filters = int(b["filters"])
            kernel = int(b["size"])
            stride = int(b["stride"])
            bias = False if "batch_normalize" in b else True
            padding = (kernel - 1) // 2 if int(b.get("pad", 0)) else 0

            conv = nn.Conv2d(prev_filters, filters, kernel, stride, padding, bias=bias)
            module.add_module("conv_{}".format(idx), conv)

            if not bias:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(idx), bn)

            if activation == "leaky":
                a = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(idx), a)

        elif b["type"] == "upsample":
            stride = int(b["stride"])
            ups = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(idx), ups)

        elif (b["type"] == "route"):
            b["layers"] = b["layers"].split(',')
            start = int(b["layers"][0])
            end = int(b["layers"][1]) if len(b["layers"]) > 1 else 0

            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx

            route = nn.Module()
            module.add_module("route_{0}".format(idx), route)

            filters = output_filters[idx + start] + (output_filters[idx + end] if end < 0 else 0)

        elif b["type"] == "shortcut":
            shortcut = nn.Module()
            module.add_module("shortcut_{}".format(idx), shortcut)

        elif b["type"] == "yolo":
            mask = b["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = b["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (network_info, module_list)


class Darknet(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        self.blocks = parse_yolo_cfg(config_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, cuda):
        module_list = self.blocks[1:]
        outputs = {}
        write = False

        for (idx, module) in enumerate(module_list):
            module_type = module["type"]
            if module_type == "convolutional" or module_type == "upsample":
                print(self.module_list[idx])
                x = self.module_list[idx](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - idx
                if len(layers) == 1:
                    x = outputs[idx + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - idx

                    x = torch.cat((outputs[idx + layers[0]], outputs[idx + layers[1]]), 1)

            elif module_type == "shortcut":
                x = outputs[idx - 1] + outputs[idx + int(module["from"])]

            elif module_type == "yolo":
                anchors = self.module_list[idx][0].anchors
                n_inputs = int(self.net_info["height"])
                n_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, n_inputs, anchors, n_classes, cuda)
                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x

        return detections


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def predict_transform(prediction: torch.Tensor, inp_dim: int,
                      anchors: List[Tuple[int, int]],
                      num_classes: int,
                      cuda: Optional[bool] = True) -> torch.Tensor:
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors
    prediction[:, :, 5:(5 + num_classes)] = torch.sigmoid((prediction[:, :, 5:(5 + num_classes)]))
    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except Exception:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:, -1])
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except Exception:
        return 0


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = (torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) *
                  torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def get_test_input():
    img = cv2.imread("pygorithms/dog-cycle-car.png")
    img = cv2.resize(img, (608, 608))
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :] / 255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img


if __name__ == '__main__':
    model = Darknet("pygorithms/yolov3.cfg").cuda()
    print(model.module_list)

    inp = get_test_input().to(device='cuda' if torch.cuda.is_available() else 'cpu')
    pred = model(inp, torch.cuda.is_available())
    print(pred)
