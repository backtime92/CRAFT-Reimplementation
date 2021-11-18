import cv2
import numpy as np
import random

def random_scale(img, bboxes, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        bboxes *= scale

    h, w = img.shape[0:2]
    random_scale = np.array([1.0, 1.5, 2.0])
    random_scale = [1.0, 1.5, 2.0]
    # scale = np.random.choice(random_scale)
    scale = random.sample(random_scale, 1)[0]
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    bboxes *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def padding_image(image,imgsize):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs

# random crop cite from PaddleOCR
def is_poly_in_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
        return False
    if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
        return False
    return True


def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax

# cite from PaddleOCR
def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    h, w = im.shape[:2]
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in text_polys:
        for point in points:
            point = np.round(point, decimals=0).astype(np.int32)
            minx = np.min(point[:, 0])
            maxx = np.max(point[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(point[:, 1])
            maxy = np.max(point[:, 1])
            h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            ymin, ymax = random_select(h_axis, h)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        num_poly_in_rect = 0
        for polys in text_polys:
            for poly in polys:
                if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                            ymax - ymin):
                    num_poly_in_rect += 1
                    break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h

# cite from PaddleOCR
class EastRandomCropData(object):
    def __init__(self,
                 size=(640, 640),
                 max_tries=10,
                 min_crop_side_ratio=0.1,
                 keep_ratio=True,
                 **kwargs):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, region_scores, affinity_scores, text_polys):
        region_scores = region_scores
        affinity_scores = affinity_scores
        text_polys = text_polys

        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = crop_area(
            region_scores, text_polys, self.min_crop_side_ratio, self.max_tries)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg_region = np.zeros((self.size[1], self.size[0]),
                                 region_scores.dtype)
            padimg_region[:h, :w] = cv2.resize(
                region_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            region_scores = padimg_region

            padimg_affinity = np.zeros((self.size[1], self.size[0]),
                              affinity_scores.dtype)
            padimg_affinity[:h, :w] = cv2.resize(
                affinity_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            affinity_scores = padimg_affinity

        else:
            region_scores = cv2.resize(
                region_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
            affinity_scores = cv2.resize(
                affinity_scores[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        for poly in zip(text_polys):
            poly = np.array([poly[i]-(crop_x,crop_y) for i in range(len(poly))])[0]
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
        return region_scores, affinity_scores, text_polys_crop