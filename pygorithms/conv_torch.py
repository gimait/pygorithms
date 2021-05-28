"""
    Attempt on yolo with pytorch.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Dict, List, Tuple


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
            padding = kernel - 1 if "pad" in b else 0

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
                n_inputs = self.net_info["height"]
                n_classes = module["classes"]

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


def predict_transform(prediction, inp_dim, anchors, num_classes, cuda=True):
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


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :] / 255.0
    img = torch.from_numpy(img).float()
    img = Variable(img)
    return img


if __name__ == '__main__':
    model = Darknet("yolov3.cfg")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred)
