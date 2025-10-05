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

## 开发计划

### 🚀 即将推出的功能

#### 1. 模型评价系统
**目标**：开发完整的模型结果评价API，支持多种评价指标和可视化分析。

**核心功能**：
- **水位面积曲线对比**：比较重建DEM与真值DEM的水位面积关系
- **蓄变量分析**：评估不同水位下的蓄水体积变化
- **多指标评价**：支持RMSE、MAE、相关系数等统计指标

**API设计**：
```python
# 基础评价接口
evaluate_dem(truth_dem_path, pred_dem_path)

# 水位面积曲线对比
compare_water_area_curve(truth_dem, area_values)

# 蓄变量分析
analyze_volume_change(truth_dem, pred_dem, area_values)
```

#### 2. 数据后处理模块 (PostProcessor)
**目标**：确保重建DEM与真值DEM的元数据完全一致，保证评价结果的准确性。

**关键检查项**：
- ✅ **坐标系一致性**：确保投影坐标系完全匹配
- ✅ **空间分辨率**：保证像素大小和地理范围一致
- ✅ **Nodata掩膜**：精确保持湖泊/陆地边界信息

**处理流程**：
```python
postprocessor = DEMPostProcessor(reference_dem_path)
aligned_dem = postprocessor.align_dem(predicted_dem_path)
```

#### 3. 评价指标模块

**3.1 水位面积曲线分析**
- **输入**：真值DEM + 重建DEM
- **输出**：水位-面积关系曲线数据
- **应用**：直接拟合结果对比

**3.2 蓄变量变化分析**  
- **输入**：真值DEM + 重建DEM + 面积值数组
- **输出**：不同水位下的蓄水体积变化
- **应用**：重建精度评估

**3.3 统计指标计算**
- **空间统计**：RMSE、MAE、相关系数
- **分布分析**：高程分布直方图对比
- **误差分析**：误差空间分布可视化

#### 4. 数据接口设计

**底层接口**：
```python
# 核心评价函数
def evaluate_water_area_curve(dem_array, area_values):
    """计算水位面积曲线"""
    pass

def calculate_volume_change(truth_dem, pred_dem, area_values):
    """计算蓄变量变化"""
    pass
```

**顶层接口**：
```python
# 文件输入接口
def load_area_values_from_csv(csv_path):
    """从CSV文件读取面积值"""
    pass

def load_dem_from_tif(tif_path):
    """加载DEM数据并验证格式"""
    pass
```

#### 5. 质量保证

**元数据验证**：
- 自动检查坐标系、分辨率、边界框
- 验证Nodata值设置和掩膜区域
- 确保数据类型和精度一致

**错误处理**：
- 提供详细的错误信息和修复建议
- 支持自动修复常见的不一致问题
- 生成验证报告

### 📋 开发优先级

1. **Phase 1**：PostProcessor模块 - 确保数据一致性
2. **Phase 2**：基础评价API - 核心评价功能
3. **Phase 3**：可视化模块 - 结果展示和分析
4. **Phase 4**：高级分析 - 蓄变量和曲线对比

### 🎯 预期成果

- 完整的模型评价工具链
- 标准化的数据格式处理
- 丰富的可视化分析功能
- 简单易用的API接口

## 许可证

本项目采用开源许可证，详见LICENSE文件。
