from progress.bar import IncrementalBar
from facenet_pytorch import MTCNN
from collections import namedtuple
from math import sin, cos, pi, sqrt
from time import time
from scipy.interpolate import interp1d
import torch
import face_detector
import nn_staff
import cv2
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np

IOU_threshold = 0.15  # Порог intersection-oer-union для интерпретации результата как TP или FP
magic_number_of_images = 2854
trust_threshold = 0  # Уверенность детектора
txt_folder = 'FDDB-folds/'  # Директория с аннотациями к изображениям
img_folder = ''  # Директория с датасетом
progress_x = 0
detectors = ['Mobilenet', 'Yolov 5', 'MTCNN', 'CNN']

metrics = {}

for detector in detectors:
    metrics[detector] = {'TP': 0, 'FP': 0, 'FN': 0, 'average_IOU': 0, 'speed': 0, 'precision': [], 'recall': [],
                         'accuracy': []}

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
detector4 = face_detector.FaceDetector()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


def ellipse_to_bb(major_radius, minor_radius, xcenter, ycenter, angle):  # Перевод отметки лица эллипсом в bounding box
    ux = minor_radius * cos(angle)
    uy = minor_radius * sin(angle)
    vx = major_radius * cos(angle + pi / 2)
    vy = major_radius * cos(angle + pi / 2)
    bbox_halfwidth = sqrt(ux * ux + vx * vx)
    bbox_halfheight = sqrt(uy * uy + vy * vy)
    bbox_ul_corner = int(xcenter - bbox_halfwidth), int(ycenter - bbox_halfheight)
    bbox_br_corner = int(xcenter + bbox_halfwidth), int(ycenter + bbox_halfheight)
    return bbox_ul_corner, bbox_br_corner


def find_closest_bb(bb, marked_data):  # Вычисляем ближайший размеченный бокс
    distances = {}
    for item in marked_data:
        distances[item] = sqrt((abs(bb[0] - bb[2]) - abs(item[0][0] - item[1][0])) ** 2 + (
                abs(bb[1] - bb[3]) - (abs(item[0][1] - item[1][1]))) ** 2)
    closest = min(distances.values())
    for box, distance in distances.items():
        if distance == closest:
            return [box[0][0], box[0][1], box[1][0], box[1][1]]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] + boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except ZeroDivisionError:
        return None
    # return the intersection over union value
    return iou


def update_metrics(detector, boxes, faces, iou_threshold):
    FN = len(faces)
    faces_copy = faces
    for box in boxes:
        closest_face = find_closest_bb(box, faces)
        IOU = bb_intersection_over_union(closest_face, box)
        if IOU:
            metrics[detector]['average_IOU'] += IOU
            if IOU > iou_threshold:
                metrics[detector]['TP'] += 1
                if FN > 0:
                    FN -= 1
            else:
                metrics[detector]['FP'] += 1
    metrics[detector]['FN'] += FN


def test_detectors(detection_trust_threshold, box_intersection_threshold):
    image_counter = 0
    face_counter = 0
    bar = IncrementalBar('Detection score thresold ' + str(detection_trust_threshold) + ', IoU thresold ' + str(
        box_intersection_threshold) + ', image', max=magic_number_of_images)  # Magic number (of images)
    for item in os.listdir(txt_folder):
        if 'ellipseList' in item:
            with open(txt_folder + item) as file:
                lines = file.readlines()
                for line in range(len(lines)):
                    if '/' in lines[line][:-1]:
                        image = img_folder + lines[line][:-1] + '.jpg'
                        bar.next()
                        image_counter += 1
                        faces_number = int(lines[line + 1][:-1])
                        face_counter += faces_number
                        faces = []
                        img = cv2.imread(image)
                        height, width, channels = img.shape
                        for i in range(faces_number):
                            faces.append(ellipse_to_bb(float(lines[line + 2 + i].split(' ')[0]),
                                                       float(lines[line + 2 + i].split(' ')[1]),
                                                       float(lines[line + 2 + i].split(' ')[3]),
                                                       float(lines[line + 2 + i].split(' ')[4]),
                                                       float(lines[line + 2 + i].split(' ')[2])))
                            # cv2.rectangle(img, face[0], face[1], (255, 255, 255), 2)

                        yolov_start = time()
                        boxes5, scores5, classes = nn_staff.NnStaff.run(img)
                        metrics['Yolov 5']['speed'] += time() - yolov_start
                        yolov5_results = []
                        for result in zip(boxes5, scores5, classes):
                            if result[1] > detection_trust_threshold and result[2] == 1:
                                result[0][0] *= width
                                result[0][2] *= width
                                result[0][1] *= height
                                result[0][3] *= height
                                # cv2.rectangle(img, (int(result[0][0]), int(result[0][1])), (int(result[0][2]), int(result[0][3])), (255, 0, 0), 2)
                                yolov5_results.append(result[0])
                        update_metrics('Yolov 5', yolov5_results, faces, box_intersection_threshold)
                        mobilenet_start = time()
                        boxes4, scores4 = detector4.run(img)
                        metrics['Mobilenet']['speed'] += time() - mobilenet_start
                        mobilenet_results = []
                        for result in zip(boxes4, scores4):
                            if result[1] > detection_trust_threshold:
                                result[0][0] *= width
                                result[0][2] *= width
                                result[0][1] *= height
                                result[0][3] *= height
                                # cv2.rectangle(img, (int(result[0][0]), int(result[0][1])), (int(result[0][2]), int(result[0][3])), (255, 255, 0), 2)
                                mobilenet_results.append(result[0])
                        update_metrics('Mobilenet', mobilenet_results, faces, box_intersection_threshold)

                        mtcnn_boxes = []
                        mtcnn_start = time()
                        boxes, scores = mtcnn.detect(img)
                        metrics['MTCNN']['speed'] += time() - mtcnn_start
                        for i in range(len(scores)):
                            if scores[i] and scores[i] > detection_trust_threshold:
                                mtcnn_boxes.append(boxes[i])
                                # cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                        update_metrics('MTCNN', mtcnn_boxes, faces, box_intersection_threshold)

                        dlib_results = []
                        cnn_start = time()
                        faces_cnn = cnn_face_detector(img, 1)
                        metrics['CNN']['speed'] += time() - cnn_start
                        for face in faces_cnn:
                            if face.confidence > detection_trust_threshold:
                                cnn_box = [face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()]
                                dlib_results.append(cnn_box)

                                # cv2.rectangle(img, (cnn_box[0], cnn_box[1]), (cnn_box[2], cnn_box[3]), (255, 255, 0), 2)
                        update_metrics('CNN', dlib_results, faces, box_intersection_threshold)

                    # cv2.imshow('window', img)
                    # cv2.waitKey(0)
    bar.finish()

    if face_counter > 0:
        for detector in metrics:
            metrics[detector]['average_IOU'] = round(metrics[detector]['average_IOU'] / image_counter, 3)
            metrics[detector]['speed'] = str(round(metrics[detector]['speed'] * 1000 / image_counter)) + ' ms'
            if metrics[detector]['TP'] + metrics[detector]['FP'] != 0:
                metrics[detector]['precision'].append(
                    metrics[detector]['TP'] / (metrics[detector]['TP'] + metrics[detector]['FP']))
            else:
                metrics[detector]['precision'].append(0)
            metrics[detector]['recall'].append(metrics[detector]['TP'] / face_counter)
            metrics[detector]['accuracy'].append(metrics[detector]['TP'] / (face_counter + metrics[detector]['FP']))
            print(detector + ' result')
            print('TP:', metrics[detector]['TP'], 'FP:', metrics[detector]['FP'], 'FN:', metrics[detector]['FN'],
                  'average_IOU:', metrics[detector]['average_IOU'], 'speed:',
                  metrics[detector]['speed'])
            print('Precision, recall:')
            for pr, rc in zip(metrics[detector]['precision'], metrics[detector]['recall']):
                print(round(pr, 4), round(rc, 4))
            print('average accuracy:', sum(metrics[detector]['accuracy']) / len(metrics[detector]['accuracy']))

    cv2.destroyAllWindows()


for IOU_threshold in np.linspace(0.03, 0.33, num=11, endpoint=True):
    test_detectors(trust_threshold, IOU_threshold)
    for detector in detectors:
        metrics[detector]['TP'] = 0
        metrics[detector]['FP'] = 0
        metrics[detector]['FN'] = 0
        metrics[detector]['average_IOU'] = 0
        metrics[detector]['speed'] = 0

for detector in detectors:
    plt.scatter(metrics[detector]['recall'], metrics[detector]['precision'], label=detector)

print(metrics)

plt.legend()
plt.ylabel('precision')
plt.xlabel('recall')
plt.show()
