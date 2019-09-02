import cv2
import numpy as np
import math


def watershed1(image, viz=False):
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    ret, binary = cv2.threshold(gray, 0.6 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("binary", binary)
        cv2.waitKey()

    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  
    if viz:
        cv2.imshow("sure_bg", sure_bg)
        cv2.waitKey()


    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    if viz:
        cv2.imshow("dist", dist)
        cv2.waitKey()
    ret, sure_fg = cv2.threshold(dist, 0.2 * np.max(dist), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)

    ret, markers = cv2.connectedComponents(surface_fg)


    markers = markers + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()

    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    if viz:
        cv2.imshow("image", image)
        cv2.waitKey()
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        # print(np_contours.shape)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)


def getDetCharBoxes_core(textmap, text_threshold=0.7, low_text=0.4):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        # segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def watershed(image, viz=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    boxes, _, _ = getDetCharBoxes_core(gray)
    return np.array(boxes)

def area_ratio(area, box,mask, max_number):
    #print(box.transpose(1,0).shape)
    cv2.fillPoly(mask, [np.int32(box)], (255))
    box_area = np.sum(np.greater(mask, 0))
    ratio = box_area/area
    #ratio_max = (max_number)/2
    return ratio, box_area

def watershed2(image, viz=False):
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("binary", binary)
        cv2.waitKey()

    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  
    if viz:
        cv2.imshow("sure_bg", sure_bg)
        cv2.waitKey()

    dist = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    if viz:
        cv2.imshow("dist", dist)
        cv2.waitKey()
    #print(dist.max())
    ret, sure_fg = cv2.threshold(gray, 0.4 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)

    ret, markers = cv2.connectedComponents(surface_fg)


    markers = markers + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()

    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    if viz:
        cv2.imshow("image", image)
        cv2.waitKey()
    area = gray.shape[0] * gray.shape[1]
    max_number = np.max(markers)

    for i in range(0, np.max(markers) + 1):
        mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        # print(np_contours.shape)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        boxes.append(box)

        # ratio, box_area = area_ratio(area, box, mask, max_number)
        # if ratio < 0.3 and box_area > 4:
        #     boxes.append(box)

    return np.array(boxes)


if __name__ == '__main__':
    image = cv2.imread('images/standard.jpg', cv2.IMREAD_COLOR)
    boxes = watershed(image, False)
    print(boxes)
