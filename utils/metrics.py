import numpy as np

from utils.misc import Result


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    # TODO:不一定要有
    # mask_p = (label_pred >= 0) & (label_pred < n_class)

    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.epsilon = np.finfo(np.float32).eps  # 防止÷0变成nan

    def precision_i(self, hist):
        precision = (hist.diagonal() + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        return precision

    def recall_i(self, hist):
        recall = (hist.diagonal() + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        return recall

    def pixel_accuracy(self, hist):
        pa = np.diag(hist).sum() / hist.sum()
        return pa

    def mean_pixel_accuracy(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        mpa = np.nanmean(cpa)
        return mpa

    def precision(self, hist):
        precision = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        precision = np.nanmean(precision)
        return precision

    def recall(self, hist):
        recall = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        recall = np.nanmean(recall)
        return recall

    def f1_score(self, hist):
        f1 = (np.diag(hist) + self.epsilon) * 2 / (
                hist.sum(axis=1) * 2 + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        f1 = np.nanmean(f1)
        return f1

    def mean_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self, hist):
        freq = hist.sum(axis=1) / hist.sum()
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def class_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        # print("iou", iou)
        return iou


def evaluate(output, label, num_class, n_ignore=0):
    evaluator = Evaluator(num_class)
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(label.flatten(), output.flatten(), num_class)
    hist = hist[n_ignore:, n_ignore:]

    precision_i = evaluator.precision_i(hist)
    recall_i = evaluator.recall_i(hist)
    pixel_accuracy = evaluator.pixel_accuracy(hist)
    mean_pixel_accuracy = evaluator.mean_pixel_accuracy(hist)
    precision = evaluator.precision(hist)
    recall = evaluator.recall(hist)
    f1_score = evaluator.f1_score(hist)
    mean_iou = evaluator.mean_intersection_over_union(hist)
    fwiou = evaluator.frequency_weighted_intersection_over_union(hist)
    class_iou = evaluator.class_intersection_over_union(hist)

    result = Result(as_dict=True)
    result.append(hist, 'hist')
    result.append(mean_iou, 'miou')
    # 计算miou的时候，算不算background
    # result.append(class_iou[1:].sum() / len(class_iou[1:]), 'miou')

    result.append(precision, 'precision')
    result.append(f1_score, 'f1_score')
    result.append(fwiou, 'fwiou')
    result.append(class_iou, 'class_iou')
    result.append(precision_i, 'precision_i')
    result.append(recall_i, 'recall_i')
    result.append(pixel_accuracy, 'pixel_accuracy')
    result.append(mean_pixel_accuracy, 'mean_pixel_accuracy')
    result.append(recall, 'recall')

    return result.as_return()
