# coding=utf-8
from math import exp
import numpy as np
import cv2
import os
import math
from gaussianMap import imgproc
from data.boxEnlarge import enlargebox


class GaussianTransformer(object):

    def __init__(self, imgSize=200, enlargeSize=1.50):
        self.imgSize = imgSize
        isotropicGrayscaleImage, isotropicGrayscaleImageColor = self.gen_gaussian_heatmap()
        self.standardGaussianHeat = isotropicGrayscaleImage
        self.enlargeSize = enlargeSize
        # color_gaussian = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        # cv2.imshow("test", color_gaussian)
        # cv2.waitKey(0)
        # self._test()

    def gen_gaussian_heatmap(self):
        circle_mask = self.gen_circle_mask()
        imgSize = self.imgSize
        isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

        # scaledGaussian = lambda x: exp(-((x ** 2)/(2 * (40 **2)))) * (1 / sqrt(2 * pi * (40 **2)))
        # scaledGaussian = lambda x: exp(-((x ** 2) / (2 * (40 ** 2))))

        # 生成高斯图
        for i in range(imgSize):
            for j in range(imgSize):
                isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                    -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))
        # 如果要可视化对比正方形和最大内切圆高斯图的区别，注释下面这行即可
        isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)

        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage) * 255).astype(np.uint8)
        isotropicGrayscaleImageColor = cv2.applyColorMap(isotropicGrayscaleImage, cv2.COLORMAP_JET)
        return isotropicGrayscaleImage, isotropicGrayscaleImageColor

    # 生成高斯图的mask，对于正方形的高斯图来说，只将最大内切圆作为字符的高斯图区域去学习
    # 在初版开源的高斯图生成中，是将正方形完整区域作为高斯图的
    # 新的方法可视化后与作者论文中展示的可视化效果是完全一致的
    def gen_circle_mask(self):
        imgSize = self.imgSize
        circle_img = np.zeros((imgSize, imgSize), np.float32)
        circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

        # circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 255, -1)
        # circle_mask = cv2.applyColorMap(circle_mask, cv2.COLORMAP_JET)
        # cv2.imshow("circle", circle_mask)
        # cv2.waitKey(0)
        return circle_mask

    # 将原始的box扩大1.5倍
    # 仅仅作用于正矩形，不规则四边形无效， 请参考data/boEnlarge文件
    def enlargeBox(self, box, imgh, imgw):
        boxw = box[1][0] - box[0][0]
        boxh = box[2][1] - box[1][1]

        if imgh <= boxh or imgw <= boxw:
            return box

        enlargew = boxw * 0.5
        enlargeh = boxh * 0.5

        # box扩大这部分为了清晰，code写的比较冗余
        # 左上角顶点扩充后坐标， 剩下点顺时针以此类推
        box[0][0], box[0][1] = max(0, box[0][0] - int(enlargew*0.5)), max(0, box[0][1] - int(enlargeh*0.5))
        box[1][0], box[1][1] = min(imgw, box[1][0] + int(enlargew*0.5)), max(0, box[1][1] - int(enlargeh*0.5))
        box[2][0], box[2][1] = min(imgw, box[2][0] + int(enlargew*0.5)), min(imgh, box[2][1] + int(enlargeh*0.5))
        box[3][0], box[3][1] = max(0, box[3][0] - int(enlargew*0.5)), min(imgh, box[3][1] + int(enlargeh*0.5))

        return box

    def four_point_transform(self, target_bbox, save_dir=None):
        '''

        :param target_bbox:目标bbox
        :param save_dir:如果不是None，则保存图片到save_dir中
        :return:
        '''
        width, height = np.max(target_bbox[:, 0]).astype(np.int32), np.max(target_bbox[:, 1]).astype(np.int32)
        right = self.standardGaussianHeat.shape[1] - 1
        bottom = self.standardGaussianHeat.shape[0] - 1
        ori = np.array([[0, 0], [right, 0],
                        [right, bottom],
                        [0, bottom]], dtype="float32")
        M = cv2.getPerspectiveTransform(ori, target_bbox)
        warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (int(width), int(height)))
        warped = np.array(warped, np.uint8)
        if save_dir:
            warped_color = cv2.applyColorMap(warped, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, 'warped.jpg'), warped_color)

        return warped, width, height

    def add_character(self, image, bbox, singal = None):
        # bbox = self.enlargeBox(bbox, image.shape[0], image.shape[1])
        bbxo_copy = bbox.copy()
        bbox = enlargebox(bbox, image.shape[0], image.shape[1])
        if singal == "affinity":
            bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0] = bbxo_copy[0][0], bbxo_copy[1][0], bbxo_copy[2][0], bbxo_copy[3][0]

        # bbox = np.array([[45, 135], [135, 135], [135, 295], [45, 295]], dtype=np.float32)

        if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
            return image
        ori_box = bbox
        top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        point = top_left
        bbox -= top_left[None, :]
        transformed, width, height = self.four_point_transform(bbox.astype(np.float32))
        # if width * height < 10:
        #     return image
        try:
            score_map = image[top_left[1]:top_left[1] + transformed.shape[0],
                        top_left[0]:top_left[0] + transformed.shape[1]]
            score_map = np.where(transformed > score_map, transformed, score_map)
            image[top_left[1]:top_left[1] + transformed.shape[0],
            top_left[0]:top_left[0] + transformed.shape[1]] = score_map
        except Exception as e:
            # print('tansformed shape:{}\n image top_left shape:{}\n top transformed shape:{}\n width and hright:{}\n ori box:{}\n top left:{}\n point:{}\n min width height:{}\n bbox:{}\n'
            #       .format(transformed.shape, image[top_left[1]:top_left[1],
            # top_left[0]:top_left[0]].shape, image[top_left[1]:top_left[1] + transformed.shape[0],
            # top_left[0]:top_left[0] + transformed.shape[1]].shape, (width, height), ori_box, top_left,point,
            #       np.array([np.min(ori_box[:, 0]), np.min(ori_box[:, 1])]).astype(np.int32),
            #       ori_box-np.array([np.min(ori_box[:, 0]), np.min(ori_box[:, 1])]).astype(np.int32)))
            print('second filter {} {} {}'.format(width,height,singal))
        return image

    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = (bbox_1[0:2].sum(0) + center_1) / 3
        bl = (bbox_2[0:2].sum(0) + center_2) / 3
        tr = (bbox_2[2:4].sum(0) + center_2) / 3
        br = (bbox_1[2:4].sum(0) + center_1) / 3

        affinity = np.array([tl, bl, tr, br]).astype(np.float32)
        # tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        # bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        # tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        # br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
        #
        # affinity = np.array([tl, tr, br, bl])

        return self.add_character(image, affinity.copy(), singal='affinity'), np.expand_dims(affinity, axis=0)

    def generate_region(self, image_size, bboxes):
        height, width, channel = image_size
        target = np.zeros([height, width], dtype=np.float32)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i])
            for j in range(bboxes[i].shape[0]):
                target = self.add_character(target, character_bbox[j], singal='region')

        return target

    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        standardGaussianHeat1[standardGaussianHeat1 < (0.4 * 255)] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)

    def generate_affinity(self, image_size, bboxes, words):
        height, width, channel = image_size
        target = np.zeros([height, width], dtype=np.float32)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities


if __name__ == '__main__':
    # gaussian = GaussianTransformer(200, 1.5)
    # gaussian.saveGaussianHeat()
    image = np.zeros((500,500,3),dtype=np.uint8)

    gen = GaussianTransformer(200, 1.5)
    gen.gen_circle_mask()
    bbox = np.array([[[60, 140], [110, 160], [110, 260], [60, 230]], [[110, 165], [180, 165], [180, 255], [110, 255]]])
    bbox = bbox[np.newaxis, :, :,:]
    region_image = gen.generate_region(image.shape, bbox)
    region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
    affinity_image, affinities = gen.generate_affinity(image.shape, bbox, [[1,2]])
    affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)
    target_bbox = np.array([[45, 135], [135, 135], [135, 295], [45, 295]], dtype=np.int8)

    for boxes in bbox:
        for box in boxes:
            # cv2.rectangle(image, tuple(box[0]), tuple(box[2]), (0, 255, 255), 2)
            enlarge = enlargebox(box, image.shape[1], image.shape[0])
            cv2.polylines(region_image, [box], True, (0, 255, 255), 2)
            cv2.polylines(region_image, [enlarge], True, (0, 0, 255), 2)
            cv2.polylines(affinity_image, [box], True, (0, 255, 255), 2)
            # cv2.polylines(affinity_image, [enlarge], True, (0, 0, 255), 2)

    cv2.polylines(affinity_image, [affinities[0].astype(np.int)], True, (255, 0, 255), 2)

    cv2.imshow('test', np.hstack((region_image, affinity_image)))
    cv2.waitKey(0)

    # weight, target = gaussian.generate_target((1024, 1024, 3), bbox.copy())
    # target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(weight.copy() / 255)
    # cv2.imshow('test', target_gaussian_heatmap_color)
    # cv2.waitKey()
    # cv2.imwrite("test.jpg", target_gaussian_heatmap_color)
