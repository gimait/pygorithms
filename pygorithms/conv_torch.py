"""
    Attempt on yolo with pytorch.
"""

import torch.nn as nn
from typing import Dict, List


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


def create_modules(blocks: List[Dict]) -> nn.ModuleList:
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

            conv = nn.Conv2d(prev_filters, filters, kernel, stride, padding, bias)
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


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


if __name__ == '__main__':
    blocks = parse_yolo_cfg("yolov3.cfg")
    print(create_modules(blocks))
