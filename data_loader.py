import torch
import torch.utils.data as data
import scipy.io as scio
from gaussian import GaussianTransformer
from watershed import watershed2
import re
import itertools
from file_utils import *
from coordinates import *
from mep import mep
import random
from PIL import Image
import torchvision.transforms as transforms


def random_scale(img, bboxes, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        bboxes *= scale

    h, w = img.shape[0:2]
    random_scale = np.array([1.0, 1.3, 1.5])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    bboxes *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


class craft_base_dataset(data.Dataset):
    def __init__(self, target_size=768, viz=False, debug=False):
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        self.gaussianTransformer = GaussianTransformer(imgSize=512, distanceRatio=1.7)
    def load_image_gt_and_confidencemask(self, index):
        return None, None, None, None

    def crop_image_by_bbox(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    def inference_pursedo_bboxes(self, net, img, word, viz=False):
        net.eval()
        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = img.copy()
        scale = 64.0 / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)

        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).cuda()
        scores, _ = net(img_torch)
        region_scores = scores[0, :, :, 0].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
        bgr_region_scores = cv2.cvtColor(region_scores, cv2.COLOR_GRAY2BGR)
        pursedo_bboxes = watershed2(bgr_region_scores)

        if pursedo_bboxes.shape[0] > 1:
            index = np.argsort(pursedo_bboxes[:, 0, 0])
            pursedo_bboxes = pursedo_bboxes[index]

        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))
        bboxes = []
        if confidence <= 0.5:
            width = input.shape[1]
            height = input.shape[0]

            width_per_char = width / len(word)
            for i, char in enumerate(word):
                if char == ' ':
                    continue
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        else:
            bboxes = pursedo_bboxes
            bboxes *= 2

        if viz:
            _tmp_bboxes = np.int32(bboxes.copy())
            for bbox in _tmp_bboxes:
                cv2.polylines(np.uint8(input), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))
            region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
            region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))
            viz_image = np.hstack([input[:, :, ::-1], region_scores_color])
            # cv2.imshow("crop_image", viz_image)
            # cv2.waitKey()
        bboxes /= scale
        return bboxes, region_scores, confidence

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_imagename(self, index):
        return None

    def saveInput(self, imagename, image, region_scores, affinity_scores, confidence_mask):

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray],
                                axis=0)
        output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), "%s_input.jpg" % imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def saveImage(self, imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            # affinity_bboxes = affinity_bboxes * 2
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        output = np.concatenate(
            [output_image, target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color,
             confidence_mask_gray],
            axis=1)
        # output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask = self.load_image_gt_and_confidencemask(index)
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []
        if len(character_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(image.shape, character_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(image.shape,
                                                                                          character_bboxes,
                                                                                          words)
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores,
                           confidence_mask)

        random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        random_transforms = random_horizontal_flip(random_transforms)
        random_transforms = random_rotate(random_transforms)
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size))

        cvimage, region_scores, affinity_scores, confidence_mask = random_transforms
        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
        # image = cvimage
        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image.double(), region_scores_torch.double(), affinity_scores_torch.double(), confidence_mask_torch.double()










class Synth80k(craft_base_dataset):

    def __init__(self, synthtext_folder, target_size=768, viz=False, debug=False):
        super(Synth80k, self).__init__(target_size, viz, debug)
        self.synthtext_folder = synthtext_folder
        gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))
        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''
        img_path = os.path.join(self.synthtext_folder, self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale(image, _charbox, self.target_size)

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

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32)
class ICDAR2013(craft_base_dataset):
    def __init__(self, net, icdar2015_folder, target_size=768, viz=False, debug=False):
        super(ICDAR2013, self).__init__(target_size, viz, debug)
        self.net = net
        self.img_folder = os.path.join(icdar2015_folder, 'ch4_training_images')
        self.gt_folder = os.path.join(icdar2015_folder, 'ch4_training_localization_transcription_gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def convert2013(box):
        str = box[-1][1:-1]
        bboxes = [box[0], box[1], box[2], box[1],
                  box[2], box[3], box[0], box[3],
                  str]
        return bboxes


    def load_image_gt_and_confidencemask(self, index):
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

        image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0],image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###':
                    # skip ### bbox,so fill the mask with zero
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                else:
                    word_image, MM = self.crop_image_by_bbox(image, word_bboxes[i])
                    pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net,
                                                                                                   word_image,
                                                                                                   words[i],
                                                                                                   viz=self.viz)
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                    new_words.append(words[i])
                    for j in range(len(pursedo_bboxes)):
                        ones = np.ones((4, 1))
                        tmp = np.concatenate([pursedo_bboxes[j], ones], axis=-1)
                        I = np.matrix(MM).I
                        ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                        pursedo_bboxes[j] = ori[:, :2]
                    character_bboxes.append(pursedo_bboxes)

        return image, character_bboxes, new_words, confidence_mask

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            ori_box = self.convert2013(ori_box)
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
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
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
            words.append(word)
        return bboxes, words

class ICDAR2015(craft_base_dataset):
    def __init__(self, net, icdar2015_folder, target_size=768, viz=False, debug=False):
        super(ICDAR2015, self).__init__(target_size, viz, debug)
        self.net = net
        self.img_folder = os.path.join(icdar2015_folder, 'ch4_training_images')
        self.gt_folder = os.path.join(icdar2015_folder, 'ch4_training_localization_transcription_gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def load_image_gt_and_confidencemask(self, index):
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

        image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0],image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###':
                    # skip ### bbox,so fill the mask with zero
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                else:
                    word_image, MM = self.crop_image_by_bbox(image, word_bboxes[i])
                    pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net,
                                                                                                   word_image,
                                                                                                   words[i],
                                                                                                   viz=self.viz)
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                    new_words.append(words[i])
                    for j in range(len(pursedo_bboxes)):
                        ones = np.ones((4, 1))
                        tmp = np.concatenate([pursedo_bboxes[j], ones], axis=-1)
                        I = np.matrix(MM).I
                        ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                        pursedo_bboxes[j] = ori[:, :2]
                    character_bboxes.append(pursedo_bboxes)

        return image, character_bboxes, new_words, confidence_mask

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
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
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
            words.append(word)
        return bboxes, words



