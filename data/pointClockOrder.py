import scipy.io as scio
import os
# import torch
# import torch.utils.data as data
import cv2
import numpy as np
import re
import itertools
import math

def distance(p1, p2, p):
    return abs(((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0]) /
               math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2))


def antipodal_pairs(convex_polygon):
    l = []
    n = len(convex_polygon)
    p1, p2 = convex_polygon[0], convex_polygon[1]

    t, d_max = None, 0
    for p in range(1, n):
        d = distance(p1, p2, convex_polygon[p])
        if d > d_max:
            t, d_max = p, d
    l.append(t)

    for p in range(1, n):
        p1, p2 = convex_polygon[p % n], convex_polygon[(p + 1) % n]
        _p, _pp = convex_polygon[t % n], convex_polygon[(t + 1) % n]
        while distance(p1, p2, _pp) > distance(p1, p2, _p):
            t = (t + 1) % n
            _p, _pp = convex_polygon[t % n], convex_polygon[(t + 1) % n]
        l.append(t)

    return l


# returns score, area, points from top-left, clockwise , favouring low area
def mep(convex_polygon):
    def compute_parallelogram(convex_polygon, l, z1, z2):
        def parallel_vector(a, b, c):
            v0 = [c[0] - a[0], c[1] - a[1]]
            v1 = [b[0] - c[0], b[1] - c[1]]
            return [c[0] - v0[0] - v1[0], c[1] - v0[1] - v1[1]]

        # finds intersection between lines, given 2 points on each line.
        # (x1, y1), (x2, y2) on 1st line, (x3, y3), (x4, y4) on 2nd line.
        def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            return px, py

        # from each antipodal point, draw a parallel vector,
        # so ap1->ap2 is parallel to p1->p2
        #    aq1->aq2 is parallel to q1->q2
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1 + 1) % n]
        q1, q2 = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n]
        ap1, aq1 = convex_polygon[l[z1 % n]], convex_polygon[l[z2 % n]]
        ap2, aq2 = parallel_vector(p1, p2, ap1), parallel_vector(q1, q2, aq1)

        a = line_intersection(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1], q2[0], q2[1])
        b = line_intersection(p1[0], p1[1], p2[0], p2[1], aq1[0], aq1[1], aq2[0], aq2[1])
        d = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], q1[0], q1[1], q2[0], q2[1])
        c = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], aq1[0], aq1[1], aq2[0], aq2[1])

        s = distance(a, b, c) * math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        return s, a, b, c, d

    z1, z2 = 0, 0
    n = len(convex_polygon)

    # for each edge, find antipodal vertice for it (step 1 in paper).
    l = antipodal_pairs(convex_polygon)

    so, ao, bo, co, do, z1o, z2o = 100000000000, None, None, None, None, None, None

    # step 2 in paper.
    for z1 in range(0, n):
        if z1 >= z2:
            z2 = z1 + 1
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1 + 1) % n]
        a, b, c = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n], convex_polygon[l[z2 % n]]
        if distance(p1, p2, a) >= distance(p1, p2, b):
            continue

        while distance(p1, p2, c) > distance(p1, p2, b):
            z2 += 1
            a, b, c = convex_polygon[z2 % n], convex_polygon[(z2 + 1) % n], convex_polygon[l[z2 % n]]

        st, at, bt, ct, dt = compute_parallelogram(convex_polygon, l, z1, z2)

        if st < so:
            so, ao, bo, co, do, z1o, z2o = st, at, bt, ct, dt, z1, z2

    return so, ao, bo, co, do, z1o, z2o

def load_icdar2015_gt(dataFolder):
    gt_folder_path = os.listdir(os.path.join(dataFolder, "ch4_training_localization_transcription_gt"))
    whole_bboxes = []
    for gt_path in gt_folder_path:
        gt_path = os.path.join(dataFolder, "ch4_training_localization_transcription_gt/" + gt_path)
        img_path = gt_path.replace("ch4_training_localization_transcription_gt", "icdar_c4_train_imgs").replace(".txt",".jpg").replace("gt_", "")
        image = cv2.imread(img_path)
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxesInfo = {'img_path':img_path, "bboxes":[], "words":[]}
        bboxesInfo_ignore = {'img_path':img_path, "bboxes":[], "words":[]}

        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                bboxesInfo_ignore["bboxes"].append(box)
                bboxesInfo_ignore["words"].append("###")
                # words.append('###')
                # bboxes.append(box)
                # area, p0, p3, p2, p1, _, _ = mep(box)
                #
                # bbox1 = np.array([p0, p1, p2, p3])
                # distance1 = 10000000
                # index = 0
                # for i in range(4):
                #     d = np.linalg.norm(box[0] - bbox1[i])
                #     if distance1 > d:
                #         index = i
                #         distance1 = d
                # new_box1 = []
                # for i in range(index, index + 4):
                #     new_box1.append(bbox1[i % 4])
                # cv2.polylines(image, [np.array(new_box1).astype(np.int)], True, (0, 255, 255), 1)
                continue
            area, p0, p3, p2, p1, _, _ = mep(box)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            cv2.polylines(image, [np.array(new_box).astype(np.int)], True, (0, 0, 255), 1)
            bboxesInfo["bboxes"].append(np.array(new_box))
            bboxesInfo["words"].append(word)

            # bboxes.append(new_box)
            # words.append(word)
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        whole_bboxes.append(bboxesInfo)
    return whole_bboxes, words

if __name__ == "__main__":
    gt_path = "/home/yanhai/OCR/Text-Detection-Data/icdar2015/text_localization"

    _, __ = load_icdar2015_gt(gt_path)