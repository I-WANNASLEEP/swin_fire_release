# 遥感图像火灾检测项目

基于深度学习的遥感图像时空火灾检测系统，使用Swin Transformer和ConvLSTM进行时空特征建模。

## 项目简介

本项目实现了一个基于深度学习的遥感图像火灾检测模型，能够从多时相卫星图像中检测火灾区域。

SwinConvLSTM为本项目实验得到的最佳模型。

主要特点包括：

- 时空建模：结合Swin Transformer的空间特征提取能力和ConvLSTM的时序建模能力
- 多模型支持：提供UNet、AttentionUNet、SwinUNETR、UNETR和SwinConvLSTM等多种模型架构
- 混合损失函数：采用Tversky Loss、Focal Loss和Cross Entropy Loss的组合损失函数
- 实验跟踪：集成WandB进行训练过程监控和实验管理

## 主要功能

### 数据处理
- 支持多时相卫星图像序列输入
- 自动数据增强和归一化处理
- 灵活的训练/验证/测试数据集划分

### 模型架构
- SwinConvLSTM：Swin Transformer + ConvLSTM的时空融合模型
- SwinUNETR：3D Swin Transformer用于时空分割
- UNet/AttentionUNet：经典的语义分割模型
- UNETR：基于Transformer的编码器-解码器架构

### 训练策略
- 混合精度训练（AMP）
- 余弦退火学习率调度
- 梯度裁剪和早停机制
- 多阈值评估和最优阈值搜索

### 评估指标
- Precision、Recall、F1 Score
- IoU（交并比）
- Specificity、Sensitivity
- 混淆矩阵可视化

## 项目结构

    swin_fire_released/
    ├── dataset_generate.py          # 数据集生成脚本
    ├── train_models_spatial_temp.py # 模型训练主脚本
    ├── visualize_model.py           # 模型架构可视化
    ├── visualize_prediction.py      # 预测结果可视化
    ├── train_monitor.sh             # 训练监控脚本
    ├── roi/                         # ROI数据
    ├── satimg_dataset_processor/    # 卫星图像处理模块
    ├── spatial_models/              # 模型定义目录
    │   ├── swin_convlstm.py         # SwinConvLSTM模型
    │   ├── unet.py                  # UNet模型
    │   ├── attentionunet.py         # AttentionUNet
    │   └── swinunetr/               # SwinUNETR实现
    ├── saved_models/                # 保存的模型文件
    ├──environment.yaml              # 带构建号的环境文件
    ├──environment_less.yaml         # 简化后的环境文件
    └── README.md                    # 项目说明文档
    

## 环境要求

参考environment.yaml和environment_less.yaml

## 使用方法

### 数据准备

准备卫星图像数据，按以下结构组织：

    data/
    ├── dataset_train/      # 训练数据集
    ├── dataset_val/        # 验证数据集
    └── dataset_test/       # 测试数据集

### 生成数据集

    python dataset_generate.py -mode train -ts 10 -it 3 -uc af
    python dataset_generate.py -mode val -ts 10 -it 3 -uc af
    python dataset_generate.py -mode test -ts 10 -it 3 -uc af

参数说明：
- -mode：数据集模式（train/val/test）
- -ts：时间序列长度
- -it：时间间隔
- -uc：使用场景（af）

### 训练模型

    python train_models_spatial_temp.py \
        -m swin_convlstm \
        -mode af \
        -b 1 \
        -r 1 \
        -lr 0.0001 \
        -nh 4 \
        -ed 96 \
        -nc 8 \
        -ts 10 \
        -it 3 \
        -epoch 100 \
        -patience 15 \
        -grad_clip 1.0 \
        -scheduler cosine

主要训练参数：
- -m：模型名称（swin_convlstm/swinunetr3d/unet3d/attunet3d/unetr3d）
- -mode：模式（使用af）
- -b：批次大小
- -lr：学习率
- -nh：注意力头数
- -ed：嵌入维度
- -nc：输入通道数
- -ts：时间序列长度
- -it：时间间隔
- -epoch：训练轮数
- -patience：早停耐心值
- -scheduler：学习率调度器类型

### 模型可视化

    python visualize_model.py --pth_path saved_models/model.pth --n_channel 8 --ts_length 5

### 预测可视化

    python visualize_prediction.py --model_path saved_models/model.pth --data_path test_data

## 实验配置

损失函数配置：

    criterion = HybridLoss(
        tversky_weight=0.4,
        focal_weight=0.3,
        ce_weight=0.3
    )

优化器配置：

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.00001,
        betas=(0.9, 0.999)
    )

学习率调度器：
- CosineAnnealingWarmRestartsWithDecay：带衰减的余弦退火重启
- 支持周期性学习率调整和峰值衰减

## 自定义配置

修改数据路径，在train_models_spatial_temp.py中修改：

    root_path = '/your/data/path'
    pretrained_path = '/your/pretrained/model/path'

修改模型参数，在create_model函数中调整：

    model = SwinConvLSTM(
        image_size=(256, 256),
        in_channels=8,
        out_channels=2,
        feature_size=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        hidden_dim=64,
        dropout=0.1,
    )

## 结果分析

评估指标：
- Precision：精确率
- Recall：召回率
- F1 Score：F1分数
- IoU：交并比
- Specificity：特异性
- Sensitivity：敏感性

可视化输出：
- 混淆矩阵
- 训练曲线
- 预测结果对比图
- 模型架构图

## 注意事项

1. WandB配置：首次使用需要配置WandB账号和项目名称
2. 预训练权重：建议使用ImageNet预训练的Swin Transformer权重初始化
3. 为了适配 2D 预训练权重的维度，Swin 编码器的多头自注意力模块在四个阶段分别严格遵循了 (3, 6, 12, 24) 的配置，即超参数的多头注意力机制输入实际不生效。
