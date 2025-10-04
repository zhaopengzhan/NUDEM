import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self, num_masks: float):
        super(DiceLoss, self).__init__()
        self.num_masks = num_masks

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / self.num_masks


def main1():
    # Usage
    dice_loss_fn = DiceLoss(num_masks=10)
    inputs = torch.randn(3, 1, 256, 256)  # Example input tensor
    targets = torch.randint(0, 2, (3, 1, 256, 256)).float()  # Example target tensor
    loss = dice_loss_fn(inputs, targets)
    print(loss)


class FocalLoss2(nn.Module):
    def __init__(self, class_num=3, alpha=None, gamma=2, size_average=True, ignore_index=None):
        super(FocalLoss2, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        # 1. flatten
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        targets = targets.view(-1)  # (N*H*W)

        # 2. 处理 ignore_index
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            inputs = inputs[valid_mask]
            targets = targets[valid_mask]
            if targets.numel() == 0:
                return torch.tensor(0.0, requires_grad=True, device=inputs.device)

        class_mask = inputs.new_zeros(inputs.size(0), C)
        ids = targets.view(-1, 1).long()
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (F.softmax(inputs, dim=1) * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * ((1 - probs) ** self.gamma) * log_p

        return batch_loss.mean() if self.size_average else batch_loss.sum()


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=5, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def dice_loss(inputs, targets, batch_size, epsilon=1e-7):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # num_masks=5
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / batch_size

    # 计算input和label的乘积，并在指定维度上求和
    # reduce_dim = list(range(1, len(inputs.shape)))
    # inse = torch.sum(inputs * targets, dim=reduce_dim)
    #
    # # 计算input和label的和，再在指定维度上求和
    # dice_denominator = torch.sum(inputs, dim=reduce_dim) + torch.sum(targets, dim=reduce_dim)
    #
    # # 计算Dice分数，添加一个小常数epsilon以防止除零
    # dice_score = 1 - 2 * inse / (dice_denominator + epsilon)
    #
    # # 返回Dice分数的均值
    # return torch.mean(dice_score)


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 输入: y_pred (B, C, H, W), y_true (B, C, H, W) (one-hot)
        y_pred = y_pred.softmax(dim=1)
        batch_size, num_classes = y_pred.shape[:2]

        # 展开为 (B, C, H*W)
        y_pred = y_pred.view(batch_size, num_classes, -1)
        y_true = y_true.view(batch_size, num_classes, -1)

        # 计算每个类别的 intersection 和 union
        intersection = torch.einsum("bwh,bwh->b", y_pred, y_true)  # 结果形状 (B,)
        union = y_pred.sum(dim=(0, 2)) + y_true.sum(dim=(0, 2))  # (C,)

        # Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # 返回平均 Dice Loss


import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple, List


class JointLoss(nn.Module):
    """
    支持任意数量的 loss，并按给定权重做加权和（或平均）。

    参数
    ----
    losses : Sequence[nn.Module] | Sequence[Tuple[nn.Module, float]]
        - 如果元素是 `nn.Module`，则需另外传 `weights`。
        - 如果元素是 `(loss_fn, weight)` 元组，则忽略外部 `weights`。
    weights : Sequence[float] | None
        每个 loss 的权重；若为 None，则全部权重为 1。
    reduction : str
        "sum"（默认）或 "mean"。对加权值做求和或取平均。
    """

    def __init__(self,
                 losses: Sequence[Union[nn.Module, Tuple[nn.Module, float]]],
                 weights: Sequence[float] | None = None,
                 reduction: str = "sum"):
        super().__init__()
        # 把 (loss, w) 或 loss 整理成统一格式
        modules: List[nn.Module] = []
        wts: List[float] = []

        if isinstance(losses[0], tuple):
            for loss_fn, w in losses:  # type: ignore
                modules.append(loss_fn)
                wts.append(float(w))
            if weights is not None:
                raise ValueError("若 losses 已含权重，请不要再传 weights")
        else:
            modules.extend(losses)  # type: ignore
            wts = [float(w) for w in (weights or [1.0] * len(losses))]
            if len(wts) != len(modules):
                raise ValueError("weights 数量必须和 losses 一致")

        self.losses = nn.ModuleList(modules)
        self.weights = wts
        assert reduction in ("sum", "mean")
        self.reduction = reduction

    def forward(self, *inputs, **kwargs):
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(*inputs, **kwargs)
        if self.reduction == "mean":
            total = total / len(self.losses)
        return total


if __name__ == "__main__":
    # masks_pred表示模型的预测结果，true_masks表示真实标签   此处在多分类情况下，如果标签维度仅为(b，h,w），则需要onehot编码，增加channel维度，保持标签与预测结果的size一致
    # dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
    #           multiclass=True)
    # main1()
    # nn.CrossEntropyLoss()
    from pytorch_toolbelt import losses as L

    # Creates a loss function that is a weighted sum of focal loss
    # and lovasz loss with weigths 1.0 and 0.5 accordingly.
    loss_fn = JointLoss([
        (L.CrossEntropyFocalLoss(ignore_index=0), 1.0),
        (L.DiceLoss(mode="multiclass", ignore_index=0), 0.5),
        (L.JaccardLoss(mode="multiclass"), 0.1)
    ])
    loss = loss_fn(torch.randn((1, 25, 224, 224)), torch.randint(0, 25, (1, 224, 224)))
    pass
