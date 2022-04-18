import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

from skimage.measure import regionprops, label

eps = 1e-8
threshold = 0.5


def get_similiarity(output, target, ch=None):
    """
    Call to calculate similiarity metrics for train, validation and test
    @ inputs:
        output : predicted model result (N x C x H x W)
        target : one-hot formatted labels (N x C x H x W)
    @ outputs:
        dice similiarity = 2 * inter / (inter + union + e)
        jaccard similiarity = inter / (union + e)
    """
    if ch is not None:
        output1 = output[:, ch, :, :]
        target1 = target[:, ch, :, :]
    else:
        output1 = output
        target1 = target

    intersection = torch.sum(output1 * target1.float())
    union = torch.sum(output1) + torch.sum(target1.float()) - intersection
    dice = 2 * intersection / (union + intersection + eps)
    # jaccard = intersection / (union + eps)
    return dice  # , jaccard


def show_contour(label_path):

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(label_path.replace('label', 'image'))

    contours, hierarchy = cv2.findContours(image=label, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the original image
    cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    for i, item in enumerate(contours):
        coor = item.squeeze().mean(axis=0)
        x, y = round(coor[0]), round(coor[1])
        cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=2)

    fig = plt.figure(figsize=(12,12))
    plt.imshow(image)

    return None


class Stent:
    def __init__(self, x_list, upper_y, lower_y):
        self.pixels = set()
        assert len(x_list) == len(upper_y) == len(lower_y)
        for i, x in enumerate(x_list):
            self.pixels.update([(x,y) for y in np.arange(upper_y[i], lower_y[i]+1)])
        self.centroid = round(np.array(list(self.pixels)).T.mean(axis=1)[0]), round(np.array(list(self.pixels)).T.mean(axis=1)[1])


def find_stent_list(label):
    upper = []
    lower = []
    col_list = []
    stent_list = []

    for col_index in range(len(label)):

        col = label[:, col_index]

        target_index = np.where(col > 0)[0]

        if len(target_index) == 0:
            if len(col_list) > 0:
                stent_list += [Stent(col_list, upper, lower)]
                upper = []
                lower = []
                col_list = []
            continue

        upper += [target_index[0]]
        lower += [target_index[-1]]
        col_list += [col_index]

    # Check for last column
    if len(col_list) > 0:
        stent_list += [Stent(col_list, upper, lower)]

    for s in stent_list:
        if len(s.pixels) < 10:
            stent_list.remove(s)

    return stent_list


def calculate_precision_recall_f1(polar_pred, polar_label, threshold_distance=10):

    cart_pred = cv2.linearPolar(polar_pred.T, (512, 512), 484, cv2.WARP_INVERSE_MAP)
    cart_label = cv2.linearPolar(polar_label.T, (512, 512), 484, cv2.WARP_INVERSE_MAP)

    # Clean outer background
    mask = np.zeros_like(cart_pred)
    mask = cv2.circle(mask, (512, 512), 483, (255, 255, 255), -1)
    cart_pred = cv2.bitwise_and(mask, cart_pred)
    cart_label = cv2.bitwise_and(mask, cart_label)

    # cv2.imwrite('img_test11.bmp', cart_pred)
    # cv2.imwrite('img_test22.bmp', cart_label)

    pred_regions = regionprops(label(cart_pred, connectivity=None))
    label_regions = regionprops(label(cart_label, connectivity=None))

    total_pred = len(pred_regions)
    total_label = len(label_regions)

    if total_pred>30:
        print("Warning: Pixel Error Causing Overflow Regions [Prediction]")
    if total_label>30:
        print("Warning: Pixel Error Causing Overflow Regions [Label]")
    TP = 0

    for props in pred_regions:
        y0, x0 = props.centroid
        candidate = None
        min_distance = threshold_distance
        for label_props in label_regions:
            yl, xl = label_props.centroid

            distance = np.sqrt(np.power(x0 - xl, 2) + np.power(y0 - yl, 2))

            if distance < min_distance:
                min_distance = distance
                candidate = label_props

        if candidate is not None:
            TP += 1
            label_regions.remove(candidate)

    try:
        precision = TP / total_pred
    except:
        precision = 0
        print(f"Warning : Zero Division [Precision]")

    try:
        recall = TP / total_label
    except:
        recall = 0
        print(f"Warning : Zero Division [Precision]")

    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except:
        f1 = 0
        print(f'Warning: Zero Division [F1 Score]')

    return precision, recall, f1


# def calculate_precision_recall_f1(pred_stent_list, label_stent_list, pixel_padding=2):
#     TP, FP, TN, FN = 0, 0, 0, 0
#     p = pixel_padding
#
#     # Check TP and FP
#     if len(label_stent_list) > len(pred_stent_list):
#         FN = len(label_stent_list) - len(pred_stent_list)
#
#     for pred_stent in pred_stent_list:
#
#         candidate = None
#         min_distance = 30
#         for label_stent in label_stent_list:
#             x_diff = pred_stent.centroid[0] - label_stent.centroid[0]
#             y_diff = pred_stent.centroid[1] - label_stent.centroid[1]
#             distance = np.sqrt(x_diff**2 + y_diff**2)
#
#             if distance < min_distance:
#                 candidate = label_stent
#
#         if candidate is None:
#             FP += 1
#         else:
#             UNION = pred_stent.pixels.union(candidate.pixels)
#             INTERSECT = pred_stent.pixels.intersection(candidate.pixels)
#             IOU = len(INTERSECT) / len(UNION)
#
#             if IOU > 0.5:
#                 TP += 1
#             else:
#                 FP += 1
#
#     try:
#         precision = TP / (TP + FP)
#     except:
#         print(f'Warning: Zero Division during Precision Calculation')
#         precision = 0
#
#     try:
#         recall = TP / (TP + FN)
#     except:
#         print(f'Warning: Zero Division during Recall Calculation')
#         recall = 0
#
#     try:
#         f1 = (2 * precision * recall) / (precision + recall)
#     except:
#         print(f'Warning: Zero Division during F1-Score Calculation')
#         f1 = 0
#
#     return precision, recall, f1


# def calculate_precision_recall_f1(pred, label, pixel_padding=2):
#     TP, FP, TN, FN = 0, 0, 0, 0
#     p = pixel_padding
#
#     pred_contours, _ = cv2.findContours(image=pred, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#     label_contours, _ = cv2.findContours(image=label, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#
#     pred_box = list(map(lambda x: dict(centroid=x.squeeze().mean(axis=0), pixels=x.squeeze()), pred_contours))
#     label_box = list(map(lambda x: dict(centroid=x.squeeze().mean(axis=0), pixels=x.squeeze()), label_contours))
#
#     pairs_list = []
#     for box in pred_box:
#
#         min_distance = 50
#         candidate = None
#
#         for l_box in label_box:
#             distance = np.sum((box['centroid'] - l_box['centroid']) ** 2)
#
#             if distance < min_distance:
#                 min_distance = np.sum((box['centroid'] - l_box['centroid']) ** 2)
#                 candidate = l_box
#
#         if candidate is None:
#             FP += 1
#         else:
#             pairs_list += [(box, candidate)]
#
#     # Check for False Negative
#     if len(pairs_list) < len(label_box):
#         FN = len(label_box) - len(pairs_list)
#
#     for pair in pairs_list:
#         pred_pixels = set([(pixel[0], pixel[1]) for pixel in pair[0]['pixels']])
#         label_pixels = set([(pixel[0], pixel[1]) for pixel in pair[1]['pixels']])
#
#         for x, y in pred_pixels.copy():
#             for padding_x, padding_y in set(
#                     permutations(np.concatenate([np.arange(-p, p + 1), np.arange(-p, p + 1)]), 2)):
#                 pred_pixels.add((x + padding_x, y + padding_y))
#         for x, y in label_pixels.copy():
#             for padding_x, padding_y in set(
#                     permutations(np.concatenate([np.arange(-p, p + 1), np.arange(-p, p + 1)]), 2)):
#                 label_pixels.add((x + padding_x, y + padding_y))
#
#         UNION = len(pred_pixels.union(label_pixels))
#         INTERCEPT = len(pred_pixels.intersection(label_pixels))
#
#         IOU = INTERCEPT / UNION
#
#         if IOU > 0.5:
#             TP += 1
#         else:
#             FP += 1
#
#     try:
#         precision = TP / (TP + FP)
#     except:
#         print(f'Warning: Zero Division during Precision Calculation')
#         precision = 0
#
#     try:
#         recall = TP / (TP + FN)
#     except:
#         print(f'Warning: Zero Division during Recall Calculation')
#         recall = 0
#
#     try:
#         f1 = (2 * precision * recall) / (precision + recall)
#     except:
#         print(f'Warning: Zero Division during F1-Score Calculation')
#         f1 = 0
#
#     return precision, recall, f1


# def calculate_precision_recall_f1(pred, label, pixel_padding=2):
#     TP, FP, TN, FN = 0, 0, 0, 0
#     p = pixel_padding
#
#     pred_contours, _ = cv2.findContours(image=pred, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#     label_contours, _ = cv2.findContours(image=label, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#
#     pred_box = list(map(lambda x: dict(centroid=x.squeeze().mean(axis=0), pixels=x.squeeze()), pred_contours))
#     label_box = list(map(lambda x: dict(centroid=x.squeeze().mean(axis=0), pixels=x.squeeze()), label_contours))
#
#     pairs_list = []
#     for box in pred_box:
#
#         min_distance = 50
#         candidate = None
#
#         for l_box in label_box:
#             distance = np.sum((box['centroid'] - l_box['centroid']) ** 2)
#
#             if distance < min_distance:
#                 min_distance = np.sum((box['centroid'] - l_box['centroid']) ** 2)
#                 candidate = l_box
#
#         if candidate is None:
#             FP += 1
#         else:
#             pairs_list += [(box, candidate)]
#
#     # Check for False Negative
#     if len(pairs_list) < len(label_box):
#         FN = len(label_box) - len(pairs_list)
#
#     for pair in pairs_list:
#         pred_pixels = set([(pixel[0], pixel[1]) for pixel in pair[0]['pixels']])
#         label_pixels = set([(pixel[0], pixel[1]) for pixel in pair[1]['pixels']])
#
#         for x, y in pred_pixels.copy():
#             for padding_x, padding_y in set(
#                     permutations(np.concatenate([np.arange(-p, p + 1), np.arange(-p, p + 1)]), 2)):
#                 pred_pixels.add((x + padding_x, y + padding_y))
#         for x, y in label_pixels.copy():
#             for padding_x, padding_y in set(
#                     permutations(np.concatenate([np.arange(-p, p + 1), np.arange(-p, p + 1)]), 2)):
#                 label_pixels.add((x + padding_x, y + padding_y))
#
#         UNION = len(pred_pixels.union(label_pixels))
#         INTERCEPT = len(pred_pixels.intersection(label_pixels))
#
#         IOU = INTERCEPT / UNION
#
#         if IOU > 0.5:
#             TP += 1
#         else:
#             FP += 1
#
#     try:
#         precision = TP / (TP + FP)
#     except:
#         print(f'Warning: Zero Division during Precision Calculation')
#         precision = 0
#
#     try:
#         recall = TP / (TP + FN)
#     except:
#         print(f'Warning: Zero Division during Recall Calculation')
#         recall = 0
#
#     try:
#         f1 = (2 * precision * recall) / (precision + recall)
#     except:
#         print(f'Warning: Zero Division during F1-Score Calculation')
#         f1 = 0
#
#     return precision, recall, f1