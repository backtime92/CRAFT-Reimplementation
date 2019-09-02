from test import test_net
import time
import file_utils
import os
import imgproc
import cv2
from eval.icdar2015.script import eval_2015
from eval.icdar2013.script import eval_2013


def eval2013(craft, test_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    image_list, _, _ = file_utils.get_files(test_folder)
    t = time.time()
    res_gt_folder = os.path.join(result_folder, 'gt')
    res_mask_folder = os.path.join(result_folder, 'mask')
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(craft, image, text_threshold, link_threshold, low_text, True, False, 980,
                                             1.5, False)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult13(image_path, polys, dirname=res_gt_folder)

    eval_2013(res_gt_folder)
    print("elapsed time : {}s".format(time.time() - t))


def eval2015(craft, test_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    image_list, _, _ = file_utils.get_files(test_folder)
    t = time.time()
    res_gt_folder = os.path.join(result_folder, 'gt')
    res_mask_folder = os.path.join(result_folder, 'mask')
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(craft, image, text_threshold, link_threshold, low_text, True, False, 2240,
                                             1.5, False)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult15(image_path, polys, dirname=res_gt_folder)

    eval_2015(os.path.join(result_folder, 'gt'))
    print("elapsed time : {}s".format(time.time() - t))
