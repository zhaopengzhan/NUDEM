# NUDEM

**NUDEM** 是一个基于深度学习的数字高程模型（DEM）重建方法，通过等高线数据来重建高质量的地形表面。

## 项目概述

NUDEM 使用神经网络和有限差分卷积来重建DEM，特别适用于从等高线数据中恢复连续的地形表面。该方法结合了：

- **绝对等高线约束**：利用已知高程值的等高线
- **相对等高线约束**：处理未知高程值的等高线
- **薄板样条平滑**：确保重建表面的平滑性和连续性

## 核心特性

### 🧠 神经网络架构
- 基于PyTorch的深度学习框架
- 使用有限差分卷积进行二阶导数计算
- 支持GPU加速训练

### 📐 数学基础
- **薄板样条能量最小化**：∫ (f_xx² + 2f_xy² + f_yy²)
- **有限差分方法**：计算二阶偏导数（∂²f/∂x², ∂²f/∂y², ∂²f/∂x∂y）
- **等高线约束**：绝对值和相对值约束

### 🎯 优化策略
- Adam优化器配合余弦学习率调度
- 交替优化绝对和相对等高线约束
- 自适应权重衰减

## 项目结构

```
NUDEM/
├── models/
│   └── nudem/
│       ├── modeling_nudem.py    # 主模型实现
│       └── finite_diff_conv.py  # 有限差分卷积层
├── test/
│   ├── 测试NUDEM.py            # 基础测试脚本
│   └── 测试NUDEM在真实.py      # 真实数据测试
└── README.md
```

## 核心算法

### 1. 薄板样条损失 (Thin Plate Spline Loss)
```python
def thin_plate_loss(self):
    # 计算二阶导数
    dxx = self.fd_xx(x)
    dyy = self.fd_yy(x) 
    dxy = self.fd_xy(x)
    
    # 薄板能量：∫ (f_xx^2 + 2 f_xy^2 + f_yy^2)
    e = dxx.pow(2) + 2.0 * dxy.pow(2) + dyy.pow(2)
    return e[self.valid_mask].mean()
```

### 2. 等高线约束损失
- **绝对等高线损失**：约束已知高程值的等高线
- **相对等高线损失**：优化未知高程值的等高线

### 3. 有限差分卷积
支持多种导数类型：
- `x`: 对x方向的导数
- `y`: 对y方向的导数  
- `xy`: 混合二阶导数
- `lap`: 拉普拉斯算子

## 使用方法

### 基本用法

```python
from models.nudem.modeling_nudem import NUDEM

# 初始化模型
model = NUDEM(
    dem=init_tensor,           # 初始DEM张量
    valid_mask=valid_mask,     # 有效区域掩膜
    abs_val=abs_vals,         # 绝对等高线值
    abs_mask=abs_mask,        # 绝对等高线掩膜
    rel_mask=rel_masks,       # 相对等高线掩膜
    kernel_size=5             # 卷积核大小
)

# 训练模型
model.fit(
    max_epoch=2000,           # 最大训练轮数
    lr=5e-3,                 # 学习率
    epoch_print_result=100    # 结果打印间隔
)
```

### 数据准备

1. **初始DEM**：作为重建的起点
2. **绝对等高线**：包含已知高程值的等高线数据
3. **相对等高线**：仅包含位置信息，高程值待优化的等高线

## 技术特点

- ✅ **GPU加速**：支持CUDA训练
- ✅ **内存优化**：自动垃圾回收和缓存清理
- ✅ **可视化**：训练过程实时可视化
- ✅ **灵活配置**：可调节的卷积核大小和训练参数
- ✅ **鲁棒性**：处理缺失数据和边界情况

## 应用场景

- 地形重建和插值
- 等高线数据后处理
- 数字地形分析
- 地理信息系统（GIS）应用
- 遥感数据处理

## 依赖要求

- Python 3.7+
- PyTorch 1.8+
- Rasterio
- NumPy
- Matplotlib
- SymPy

## 许可证

本项目采用开源许可证，详见LICENSE文件。
