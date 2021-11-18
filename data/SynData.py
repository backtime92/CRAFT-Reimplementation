import scipy.io as scio
import os
# import torch
# import torch.utils.data as data
import cv2
import numpy as np
import re
import itertools

from gaussianMap.gaussian import GaussianTransformer
from data.boxEnlarge import enlargebox
from data.pointClockOrder import mep


class craftDataset(object):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, vis=False):
        assert 'synthtext' in data_dir_list.keys()
        assert 'icdar2015' in data_dir_list.keys()

        self.target_size = target_size
        self.data_dir_list = data_dir_list
        self.vis = vis

        self.charbox, self.image, self.imgtxt = self.load_synthtext()

    def load_synthtext(self):
        gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]
        return charbox, image, imgtxt

    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        # image = random_scale(image, _charbox, self.target_size)

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), img_path

    def inference_pursedo_bboxes(self, craft_model, image, word_bboxes, words):
        # TODO
        return None, None, None

    def load_icdar2015_gt(self, dataFolder):
        gt_folder_path = os.listdir(os.path.join(dataFolder, "ch4_training_localization_transcription_gt"))
        whole_bboxes = []
        for gt_path in gt_folder_path:
            gt_path = os.path.join(dataFolder, "ch4_training_localization_transcription_gt/" + gt_path)
            img_path = gt_path.replace("ch4_training_localization_transcription_gt", "icdar_c4_train_imgs").replace(
                ".txt", ".jpg").replace("gt_", "")
            image = cv2.imread(img_path)
            lines = open(gt_path, encoding='utf-8').readlines()
            bboxesInfo = {'img_path': img_path, "bboxes": [], "words": []}
            bboxesInfo_ignore = {'img_path': img_path, "bboxes": [], "words": []}

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


    def load_icdar2015_image_gt(self, index):
        '''
                根据索引加载ground truth
                :param index:索引
                :return:bboxes 字符的框，
                '''
        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)

        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        confidences = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    continue
                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=self.viz)
                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)
        return image, character_bboxes, new_words, confidence_mask, confidences



if __name__ == "__main__":
    data_dir_list = {"synthtext":"/media/yanhai/disk21/SynthTextData/SynthText", "icdar2015": "/home/yanhai/OCR/Text-Detection-Data/icdar2015/text_localization/icdar_c4_train_imgs"}
    craft_data = craftDataset(768, data_dir_list)
    for index in range(10000):
        image, character_bboxes, words, shapes, img_path = craft_data.load_synthtext_image_gt(index)
        # # 测试
        # image = cv2.imread("/media/yanhai/disk21/SynthTextData/SynthText/8/ballet_106_0.jpg")
        # character_bboxes = np.array([[[[423.16126397,  22.26958901],
        #                              [438.2997574,   22.46075572],
        #                              [435.54895424,  40.15739982],
        #                              [420.17946701,  39.82138755]],
        #                             [[439.60847343,  21.60559248],
        #                              [452.61288403,  21.76391911],
        #                              [449.95797159,  40.47241401],
        #                              [436.74150236,  40.18347166]],
        #                             [[450.66887979,  27.0241972 ],
        #                              [466.31976402,  27.25747678],
        #                              [464.5848793,   40.79219178],
        #                              [448.74896556,  40.44598236]],
        #                             [[466.31976402,  27.25747678],
        #                              [482.22585715,  27.49456029],
        #                              [480.68235876,  41.14411963],
        #                              [464.5848793,   40.79219178]],
        #                             [[479.76190495,  27.45783459],
        #                              [498.3934528,   27.73554156],
        #                              [497.04793842,  41.50190876],
        #                              [478.18853922,  41.08959901]],
        #                             [[504.59927448,  28.73896576],
        #                              [512.20555863,  28.85582217],
        #                              [511.1101386,   41.80934074],
        #                              [503.4152019,   41.64111176]]]])
        # character_bboxes = character_bboxes.astype(np.int)
        gaussian_map = np.zeros(image.shape, dtype=np.uint8)
        gen = GaussianTransformer(200, 1.5)
        gen.gen_circle_mask()

        region_image = gen.generate_region(image.shape, character_bboxes)
        region_image = (region_image).astype(np.uint8)
        # region_image = (((region_image/255)>0.1).astype(np.float32) * 255).astype(np.uint8)
        region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
        affinity_image, affinities = gen.generate_affinity(image.shape, character_bboxes, words)
        affinity_image = (affinity_image).astype(np.uint8)
        affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)
        # cv2.imshow("gaussion", gaussion)
        # cv2.waitKey(0)

        for boxes in character_bboxes:
            for box in boxes:
                # image = cv2.imread(img_path)
                enlarge = enlargebox(box.astype(np.int), image.shape[0], image.shape[1])
                # print("enlarge:", enlarge)
                # gaussion = gen.generate_region(image.shape, np.array([[box]]))
                # gaussion = cv2.applyColorMap(gaussion, cv2.COLORMAP_JET)
        #         cv2.polylines(image, [enlarge], True, (255, 0, 255), 1)
        #         cv2.polylines(image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(region_image, [enlarge], True, (255, 0, 255), 1)
        #         cv2.polylines(region_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        # for box in affinities:
        #     cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 0, 255), 1)
        stack_image = np.hstack((image, region_image, affinity_image))
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", stack_image)
        cv2.waitKey(0)




1 / 2 / np.pi / (40 ** 2) * np.exp(-1 / 2 * ((100 - 200 / 2) ** 2 / (40 ** 2) + (100 - 200 / 2) ** 2 / (40 ** 2)))

