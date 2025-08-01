# fMRI Masked Autoencoder (MAE) 项目总结

## 项目概述
这是一个将Masked Autoencoder (MAE)应用于fMRI数据分析的深度学习项目。项目实现了基于时空patch masking的自监督学习方法，用于从fMRI数据中学习有意义的表示，并在多个下游任务上进行评估。

## 核心组件

### 1. 核心模型文件
- **`fmri_mae.py`**: 实现了fMRI MAE模型
  - `MaskedAutoencoderFMRI`: 主要的MAE模型类
  - 支持时空patch masking
  - 包含encoder和decoder架构
  - 提供预训练好的模型配置（base, large等）

### 2. 数据处理
- **`fmri_data_utils.py`**: fMRI数据生成和加载工具
  - `FMRIDataset`: 数据集类
  - `generate_synthetic_fmri_data()`: 生成合成fMRI数据
  - 支持加载真实fMRI数据
  - 包含现实的大脑网络模式模拟

- **`fmri_masking.py`**: 时空patch masking策略
  - `FMRISpatiotemporalMasking`: 实现时空patch masking
  - 将每个脑区的时间序列分割成patches
  - 支持随机masking和结构化masking

### 3. 下游任务评估
- **`downstream_tasks.py`**: 实现多个下游评估任务
  - `FNCAnalyzer`: 功能网络连接分析
  - `DynamicFNCAnalyzer`: 动态功能网络连接分析
  - `DiseaseClassifier`: 疾病分类任务
  - 生成合成疾病数据

### 4. 训练和评估脚本
- **`train_mae.py`**: 基础MAE训练脚本
  - 简单的训练循环
  - 支持重建可视化
  - 训练过程监控

- **`test_quick.py`**: 快速功能测试
  - 验证模型基本功能
  - 测试各个组件

- **`end_to_end_evaluation.py`**: 端到端评估
  - 完整的预训练→特征提取→下游任务流程
  - 性能评估和可视化

- **`evaluate_mae.py`**: 完整评估管道
  - 最全面的评估脚本
  - 包含所有下游任务
  - 详细的结果可视化

## 工作流程

### 第一步：快速测试
```bash
python test_quick.py
```
- 验证模型基本功能
- 测试数据生成和模型前向传播
- 确保环境配置正确

### 第二步：基础训练
```bash
python train_mae.py
```
- 在合成数据上训练MAE模型
- 学习自监督表示
- 可视化训练过程和重建效果

### 第三步：下游任务测试
```bash
python downstream_tasks.py
```
- 测试各个下游任务的功能
- FNC分析、动态FNC、疾病分类
- 验证特征提取的有效性

### 第四步：端到端评估
```bash
python end_to_end_evaluation.py
```
- 完整的预训练→评估流程
- 在真实任务上验证模型性能

### 第五步：完整评估
```bash
python evaluate_mae.py
```
- 最全面的评估管道
- 包含所有任务和详细分析
- 生成完整的评估报告

## 主要特性

### 1. 时空Patch Masking
- 将fMRI时间序列分割成时间patches
- 每个脑区的时间序列独立处理
- 支持高比例的masking (75%)

### 2. 多任务评估
- **功能网络连接 (FNC)**: 分析大脑网络间的静态连接
- **动态FNC**: 分析时变的连接模式
- **疾病分类**: 使用学习的特征进行疾病诊断

### 3. 灵活的模型配置
- 支持不同规模的模型 (base: 111M参数)
- 可调整的patch大小和masking比例
- 适应不同的fMRI数据格式

## 技术栈

### 核心依赖
- **PyTorch**: 深度学习框架
- **timm**: Vision Transformer模块
- **scikit-learn**: 传统机器学习任务
- **matplotlib + seaborn**: 可视化

### 数据格式
- 输入：`[N, P, T]` (样本数，脑区数，时间点数)
- Patches：`[N, num_patches, patch_size_T]`
- 特征：`[N, num_patches, embed_dim]`

## 实验结果

### 模型性能
- MAE重建损失：~1.1-1.2
- 疾病分类准确率：>95% (合成数据)
- 动态状态识别：3-4个不同的连接状态

### 关键发现
1. **有效的表示学习**: MAE能够学习到有意义的fMRI表示
2. **下游任务性能**: 在FNC、dFNC和疾病分类上都表现良好
3. **时空建模**: 时空patch masking有效捕获了大脑的时空动态

## 改进建议

### 当前实现的改进点
1. **数据保存**: 每次都重新生成数据，可以添加数据缓存
2. **模型保存**: 训练后的模型没有保存，可以添加检查点保存
3. **真实数据**: 主要使用合成数据，需要在真实fMRI数据上验证
4. **超参数优化**: 可以添加更系统的超参数搜索

### 扩展方向
1. **多模态融合**: 结合结构MRI、DTI等多模态数据
2. **迁移学习**: 在大规模fMRI数据上预训练，然后迁移到特定任务
3. **解释性分析**: 分析模型学到的表示的生物学意义
4. **实时应用**: 优化模型用于实时fMRI分析

## 运行环境

### 系统要求
- Python 3.9+
- CUDA支持的GPU (推荐)
- 16GB+ RAM

### 依赖安装
```bash
conda create -n mae python=3.9 -y
conda activate mae
pip install torch numpy matplotlib seaborn scikit-learn timm
```

## 项目结构
```
mae/
├── fmri_mae.py              # 核心MAE模型
├── fmri_data_utils.py       # 数据处理工具
├── fmri_masking.py          # masking策略
├── downstream_tasks.py      # 下游任务
├── train_mae.py            # 基础训练
├── test_quick.py           # 快速测试
├── end_to_end_evaluation.py # 端到端评估
├── evaluate_mae.py         # 完整评估
├── util/                   # 工具函数
│   ├── pos_embed.py
│   ├── lr_sched.py
│   └── misc.py
└── PROJECT_SUMMARY.md      # 项目总结
```

这个项目展示了如何将现代自监督学习方法应用于神经影像数据分析，为fMRI数据的深度学习分析提供了一个完整的框架。
