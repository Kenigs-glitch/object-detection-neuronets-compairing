#import logging
from warnings import simplefilter

import cv2
import numpy as np
import torch

#from logger.logger import Logger
#from modules.consts import ModuleName
from models.experimental import attempt_load
from utils.general import non_max_suppression

simplefilter(action='ignore', category=DeprecationWarning)


#logger: Logger = logging.getLogger(f'Module.{ModuleName.object_detector}')


def resize_and_pad(img, size):
    h, w = img.shape[:2]
    size_h, size_w = new_h, new_w = size
    pad_h, pad_w = 0, 0
    if w > size_w or h > size_h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w / h
    wr = w / size_w
    hr = h / size_h
    if wr > hr:
        new_h = int(new_w / aspect)
        pad_h = size_h - new_h
    else:
        new_w = int(new_h * aspect)
        pad_w = size_w - new_w
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT)
    return scaled_img, max(wr, hr)


class ClassDisplayAttributes:
    @staticmethod
    def read_from_file(path):
        result = []
        with open(path, 'r') as strings:
            for s in strings:
                items = s.strip('\n').split(',')
                name = items[0]
                color = (int(items[1]), int(items[2]), int(items[3]))
                result.append(ClassDisplayAttributes(name, color))
        return result

    def __init__(self, name=None, color=None):
        self.name = name
        self.color = color


class NnStaff:
    input_size = (640, 640)
    input_shape = (3,) + input_size
    classes_disp = None
    device = None
    model = None
    is_inited = False

    @staticmethod
    def init():
        if NnStaff.is_inited:
            return

        model_path = 'yolov5.pt'
        disp_attrs_path = 'mymix_class_deps.txt'
        NnStaff.classes_disp =\
            ClassDisplayAttributes.read_from_file(disp_attrs_path)

        NnStaff.device = torch.device('cuda:0')  # TODO: device id?
        NnStaff.model = attempt_load(model_path, map_location=NnStaff.device).half()

        NnStaff.is_inited = True

    @staticmethod
    def run(image):
        h, w = image.shape[:2]
        image, r = resize_and_pad(image, NnStaff.input_size)
        image = np.array(cv2.split(image))
        tensor = torch.from_numpy(image).to(NnStaff.device).half() / 255.0
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            pred = NnStaff.model(tensor)
        pred = non_max_suppression(pred[0], 0.25, 0.45, classes=None)
        pred = pred[0]

        wr = NnStaff.input_size[0]
        hr = NnStaff.input_size[0] / w * h
        if pred.is_cuda:
            boxes = pred[:, :4] / torch.Tensor([wr, hr, wr, hr]).cuda(NnStaff.device)
            boxes = boxes.cpu().numpy()
            scores = pred[:, 4].cpu().numpy()
            classes_ids = pred[:, 5].round().cpu().numpy().astype(np.int32)
        else:
            boxes = pred[:, :4] / torch.Tensor([wr, hr, wr, hr]).cpu()
            boxes = boxes.numpy()
            scores = pred[:, 4].numpy()
            classes_ids = pred[:, 5].round().numpy().astype(np.int32)
        return boxes, scores, classes_ids

    @staticmethod
    def close():
        raise NotImplementedError()

    def __init__(self):
        raise NotImplementedError('Static class need no constructor')


NnStaff.init()
