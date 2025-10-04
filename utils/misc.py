'''
“misc”是“miscellaneous”的缩写，意为“杂项”或“其他”
放点工具直接导入就行

'''
import functools
import inspect
import time

import numpy as np
import torch
from matplotlib import pyplot as plt


def calRunTime(func):
    """ 计算函数运行时间的装饰器 """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"函数 `{func.__name__}` 运行时间: {end_time - start_time:.6f} 秒")
        return result

    return wrapper


class calRunTimer():
    def __init__(self, block_name='', func=None, disable=False):
        self.func = func
        self.block_name = block_name
        self.disable = disable
        if func is not None:
            # 如果作为装饰器使用
            # 保留函数的属性（如__name__, __doc__等）
            functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        if not self.disable: print(f"函数 `{self.func.__name__}` 运行时间: {end_time - start_time:.4f} 秒")
        return result

    def __enter__(self):
        """ 上下文管理器计算运行时间 """
        self.start_time = time.time()  # 记录开始时间
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 上下文管理器计算运行时间 """
        self.end_time = time.time()  # 记录结束时间
        elapsed_time = self.end_time - self.start_time
        if not self.disable: print(f"代码块`{self.block_name}`运行时间 运行时间: {elapsed_time:.4f} 秒")


class Wrapper:
    '''
    自己写的封装类，封装Loss\AverageMeter
    '''

    def __init__(self):
        self._container = {}

    def register(self, name, value):
        """通过名字添加或更新损失函数"""
        self._container[name] = value

    def __getattr__(self, name):
        """尝试访问不存在的属性时返回 None"""
        return self._container.get(name, None)

    def __getitem__(self, item):
        """尝试访问不存在的属性时返回 None"""
        return self._container.get(item, None)

    def keys(self):
        return self._container.keys()


class Result:
    '''
    GroupVIT里面搞得结果封装类
    '''

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = None
        self.avg = None
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = np.array(val)
        self.count += batch_size

        if self.sum is None:
            self.sum = self.val * batch_size
        else:
            self.sum += self.val * batch_size

        self.avg = self.sum / self.count


def truncate_tensor(tensor, target_shape):
    """截断 tensor 以匹配 target_shape"""
    slices = tuple(slice(0, min(tensor.size(i), target_shape[i])) for i in range(len(target_shape)))
    return tensor[slices]


def pad_tensor(tensor, target_shape):
    """补充 tensor 以匹配 target_shape
    现在是0填充
    """
    pad_width = []
    for i in range(len(target_shape)):
        diff = target_shape[i] - tensor.size(i)
        pad_width.extend([0, max(0, diff)])  # 在每一维的末尾补充
    result = torch.nn.functional.pad(tensor, pad=pad_width[::-1])
    return result


def adjust_checkpoint(checkpoint, model_state):
    updated_checkpoint = {}

    for key, pretrained_weight in checkpoint.items():
        if key in model_state:
            model_weight = model_state[key]
            if pretrained_weight.shape != model_weight.shape:
                print(f"Resizing {key}: {pretrained_weight.shape} -> {model_weight.shape}")
                pretrained_weight = truncate_tensor(pretrained_weight, model_weight.shape)
                # 先搞截断，再搞填充
                if pretrained_weight.shape != model_weight.shape:
                    pretrained_weight = pad_tensor(pretrained_weight, model_weight.shape)
            updated_checkpoint[key] = pretrained_weight
        else:
            print(f"Skipping {key}, not found in model.")

    return updated_checkpoint


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    # images = torch.stack(images, dim=0)
    return images, labels


def change_conv_channel(conv, in_ch=None, out_ch=None, bias=True):
    # 1. 拿到 Conv2d 构造函数的参数列表
    sig = inspect.signature(conv.__class__.__init__)
    valid_keys = sig.parameters.keys()

    # 2. old_conv.__dict__ 里挑出构造函数用得上的参数
    kwargs = {k: v for k, v in conv.__dict__.items() if k in valid_keys}

    # 3. 替换 in/out_channels
    if in_ch is not None:
        kwargs["in_channels"] = in_ch
    if out_ch is not None:
        kwargs["out_channels"] = out_ch

    # 4. bias 要特殊处理：bias 是个 Parameter，不是 bool
    if bias:
        kwargs["bias"] = (conv.bias is not None)
    else:
        kwargs["bias"] = False

    return conv.__class__(**kwargs)

def compare_model_weights(model_a: torch.nn.Module, model_b: torch.nn.Module, rtol=1e-5, atol=1e-8):
    """
    对比两个模型的 state_dict，逐个 key 检查权重是否一致。

    Args:
        model_a, model_b: 待比较的模型
        rtol, atol: torch.allclose 的容差设置

    Returns:
        results: dict[str, bool]，key 是参数名，value=True 表示完全一致，False 表示不一致或缺失
    """
    # 临时拷贝到 CPU
    # state_a = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
    # state_b = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    results = {}
    all_keys = set(state_a.keys()) | set(state_b.keys())
    for key in sorted(all_keys):
        if key not in state_a:
            results[key] = False
            print(f"[缺失] {key} 只在 model_b 中")
        elif key not in state_b:
            results[key] = False
            print(f"[缺失] {key} 只在 model_a 中")
        else:
            same = torch.allclose(state_a[key].cpu(), state_b[key].cpu(),
                                  rtol=rtol, atol=atol)
            results[key] = same
            if not same:
                diff = (state_a[key] - state_b[key]).abs().max().item()
                print(f"[不同] {key}  最大差异 {diff:.6f}")
    return results


def find_first_input_conv(model: torch.nn.Module, expected_in_ch: int):
    """
    在模型中查找第一个 nn.Conv2d 且 in_channels == expected_in_ch 的层。
    返回 (layer_name, layer_module)；若未找到则抛出异常。
    """
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.in_channels == expected_in_ch:
            return name, m
    # 若像 SegFormer 这类模型首层是 patch embedding，可能 in_channels 匹配在某个 proj 上
    # 可放宽条件（例如前若干层里找 kernel_size>1 的 Conv2d），但此处优先严格匹配：
    raise RuntimeError(f"找不到首层输入卷积：未发现 in_channels == {expected_in_ch} 的 nn.Conv2d")


def compare_model_first_input_conv(args, model_a: torch.nn.Module, model_b: torch.nn.Module):

    nameA, convA = find_first_input_conv(model_a, args.in_channels)
    nameB, convB = find_first_input_conv(model_b, args.in_channels_val)

    wA = convA.weight.detach().cpu().numpy()
    wB = convB.weight.detach().cpu().numpy()

    # ---------- 绘制 A ----------
    for c in range(wA.shape[1]):
        fig, axes = plt.subplots(6, 6, figsize=(16, 16))
        axes = axes.ravel()
        for i in range(32):  # out_channels
            # 这里只画第0个输入通道的卷积核，可以改成 np.mean(wA[i], axis=0) 来画7个通道的平均
            kernel = wA[i, c, :, :]
            axes[i].imshow(kernel, cmap="RdBu_r")
            axes[i].set_title(f"A-{i}", fontsize=8)
            axes[i].axis("off")
        plt.suptitle("Model A Conv Kernels (ch0)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"inc/A_{c}.png")
        # plt.show()

    # ---------- 绘制 B ----------
    for c in range(wB.shape[1]):
        fig, axes = plt.subplots(6, 6, figsize=(16, 16))
        axes = axes.ravel()
        for i in range(32):
            kernel = wB[i, c, :, :]
            axes[i].imshow(kernel, cmap="RdBu_r")
            axes[i].set_title(f"B-{i}", fontsize=8)
            axes[i].axis("off")
        plt.suptitle("Model B Conv Kernels (ch0)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"inc/B_{c}.png")
        # plt.show()
    pass


if __name__ == '__main__':
    loss_wrapper = Wrapper()
    loss_wrapper.register('loss_hr', AverageMeter())
    loss_wrapper.register('loss_lr', AverageMeter())
    loss_wrapper.register('loss_sum', AverageMeter())

    loss_wrapper.loss_hr.register(1, 1)
    loss_wrapper.loss_hr.register(2, 3)

    print(loss_wrapper.loss_mr)
