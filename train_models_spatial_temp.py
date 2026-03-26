import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F

# ==================== 自定义调度器 ====================
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    """
    在每一次 restart（余弦周期结束）时，把学习率的峰值乘以 decay_factor。
    decay_factor < 1 时会使每轮的最高学习率逐渐下降（如 0.99）。
    """
    def __init__(self, optimizer, T_0, T_mult=1, decay_factor=0.99, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay_factor = decay_factor
        self._restart_cnt = 0

    def _decay_base_lrs(self):
        self._restart_cnt += 1
        self.base_lrs = [lr * (self.decay_factor ** self._restart_cnt) for lr in self.base_lrs]

    def step(self, epoch=None):
        super().step(epoch)
        # 当检测到一次完整的周期结束后（即 restart），_last_restart 为 True
        if getattr(self, "_last_restart", False):
            self._decay_base_lrs()

# ------------------- 原始导入 -------------------
from monai.losses import FocalLoss
from monai.metrics import MeanIoU, DiceMetric
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete, Compose

from spatial_models.unet import UNet
from spatial_models.attentionunet import AttentionUnet
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unetr.unetr import UNETR
from spatial_models.swin_convlstm import SwinConvLSTM, ConvLSTMCell
from satimg_dataset_processor.data_generator_torch import Normalize, FireDataset

from sklearn.metrics import (
    f1_score, jaccard_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 新增组件 ====================
# 删除了 InputAdapter 和 SwinConvLSTM_Wrapper 类，避免与 swin_convlstm.py 中的设计理念冲突


def get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr=5e-6):
    """Warmup + Cosine Annealing 调度器"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Parameters:")
    print(f"   Total: {total / 1e6:.2f}M")
    print(f"   Trainable: {trainable / 1e6:.2f}M")
    print(f"   Frozen: {(total - trainable) / 1e6:.2f}M")

    return total, trainable


def visualize_sample(image, label, pred=None, save_path='sample_check.png'):
    """可视化样本"""
    img_t0 = image[3, 0].cpu().numpy()
    label_t0 = label[0, 0].cpu().numpy()

    img_t0 = (img_t0 - img_t0.min()) / (img_t0.max() - img_t0.min() + 1e-8)

    fig, axes = plt.subplots(1, 3 if pred is not None else 2, figsize=(15, 5))

    axes[0].imshow(img_t0, cmap='gray')
    axes[0].set_title('Input (Channel 3, Time 0)')
    axes[0].axis('off')

    axes[1].imshow(label_t0, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    if pred is not None:
        pred_t0 = pred[1, 0].cpu().numpy()
        axes[2].imshow(pred_t0, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Sample visualization saved to {save_path}")


# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description='Fire Prediction Model Training')
parser.add_argument('-m', type=str, help='Model to be executed')
parser.add_argument('-mode', type=str, help='BA or af')
parser.add_argument('-b', type=int, help='batch size')
parser.add_argument('-r', type=int, help='run')
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-av', type=str, help='attention version')
parser.add_argument('-nh', type=int, help='number-of-head')
parser.add_argument('-ed', type=int, help='embedding dimension')
parser.add_argument('-nc', type=int, help='n_channel')
parser.add_argument('-ts', type=int, help='ts_length')
parser.add_argument('-it', type=int, help='interval')
parser.add_argument('-epoch', type=int, help='Load Epoch', default=0, nargs='?')
parser.add_argument('-patience', type=int, default=15, help='Early stopping patience')
parser.add_argument('-grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('-scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'step'], help='Learning rate scheduler')
parser.add_argument('-decay_factor', type=float, default=0.99, help='Decay factor for LR peak after each restart (default 0.99)')

args = parser.parse_args()

# ==================== 超参数设置 ====================
model_name = args.m
batch_size = args.b
num_heads = args.nh
hidden_size = args.ed
ts_length = args.ts
attn_version = args.av
run = args.r
lr = args.lr
MAX_EPOCHS = 100
weight_decay = lr / 10
num_classes = 2
n_channel = args.nc
interval = args.it
mode = args.mode
patience = args.patience
grad_clip_max_norm = args.grad_clip
scheduler_type = args.scheduler

# ==================== 设置随机种子 ====================
SEED = run + 41
print(f"Random Seed: {SEED}")
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

root_path = '/home/congwei/ts-satfire-tran'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

# ==================== HybridLoss (回滚到 2.19 策略 + CE) ====================
class HybridLoss(nn.Module):
    """
    回归版：结合 2.19 的稳定性。
    1. 忽略 -1 区域 (Masking) 而不是强行惩罚。
    2. 加回 CrossEntropy Loss 辅助收敛。
    """
    def __init__(self, tversky_weight=0.4, focal_weight=0.3, ce_weight=0.3):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
        # Tversky 参数 (均衡模式)
        self.alpha = 0.5
        self.beta = 0.5
        
        self.focal_loss = FocalLoss(
            include_background=False, to_onehot_y=True, gamma=3.0, reduction='none'
        )
        # CE Loss 忽略 -1
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def tversky_loss_manual(self, pred, target, alpha, beta):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()
        return 1 - (TP + 1e-6) / (TP + alpha * FN + beta * FP + 1e-6)

    def forward(self, preds, target):
        # 兼容 Deep Supervision (如果是列表，取第一个)
        if isinstance(preds, list):
            pred = preds[0]
        else:
            pred = preds

        # 1. 预处理 Target
        if target.dim() == 5 and target.shape[1] > 1:
            target = target[:, 0:1, ...]
        elif target.dim() == 4:
            target = target.unsqueeze(1)
        
        # 2. 生成 Mask (关键回归：忽略 -1 区域)
        valid_mask = (target != -1).float()
        
        # 3. 准备二值标签 (0 或 1)
        target_binary = ((target > 0) & (target != -1)).long()
        target_float = target_binary.float()
        
        # === 计算 Tversky (只在有效区域) ===
        pred_fire_logit = pred[:, 1:2, ...]
        loss_tversky = self.tversky_loss_manual(
            pred_fire_logit * valid_mask, 
            target_float * valid_mask, 
            self.alpha, self.beta
        )
        
        # === 计算 Focal (加权平均) ===
        loss_focal_map = self.focal_loss(pred, target_binary)
        loss_focal = (loss_focal_map * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        
        # === 计算 CE (加权平均) ===
        target_squeeze = target_binary.squeeze(1)  # [B, T, H, W]
        valid_mask_squeeze = valid_mask.squeeze(1)
        loss_ce_map = self.ce_loss(pred, target_squeeze)
        loss_ce = (loss_ce_map * valid_mask_squeeze).sum() / (valid_mask_squeeze.sum() + 1e-6)
        
        # 总损失
        total_loss = (self.tversky_weight * loss_tversky + 
                      self.focal_weight * loss_focal + 
                      self.ce_weight * loss_ce)
        
        return total_loss, {
            'tversky_loss': loss_tversky.item(),
            'focal_loss': loss_focal.item(),
            'ce_loss': loss_ce.item(),
            'fire_weight': 0.0,
            'fire_ratio': 0.0
        }



# ==================== 早停机制 ====================
class EarlyStopping:
    """早停机制，只监控 F1 得分"""
    def __init__(self, patience=50, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = None
        self.early_stop = False
        
    def __call__(self, val_f1):
        if self.best_f1 is None:
            self.best_f1 = val_f1
            return False
        
        f1_improved = val_f1 > (self.best_f1 + self.min_delta)
        
        if f1_improved:
            self.best_f1 = val_f1
            self.counter = 0
            print(f"  - EarlyStopping improved (f1: {val_f1:.4f})")
        else:
            print(f"  - EarlyStopping counter: {self.counter}/{self.patience} "
                  f"[f1: {val_f1:.4f} vs best {self.best_f1:.4f}]")
            
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"\n- Early stopping triggered! No F1 improvement for {self.patience} epochs.")
            print(f"   Best F1: {self.best_f1:.4f}")

        return self.early_stop


# ==================== 评估指标 ====================
class MetricsCalculator:
    """计算并记录各种评估指标"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, pred, target):
        """更新预测和真实标签"""
        pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        target_np = target.cpu().numpy() if torch.is_tensor(target) else target
        
        self.predictions.extend(pred_np.flatten())
        self.targets.extend(target_np.flatten())
        
    def compute(self):
        """计算所有指标"""
        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)
        
        # 基本指标
        # 将多类标签转换为二进制标签（0表示无火灾，>0表示有火灾）
        target_binary = (target_array > 0).astype(int)
        pred_binary = (pred_array > 0).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary', zero_division=1.0
        )
        
        iou = jaccard_score(target_binary, pred_binary, zero_division=1.0)
        
        # 混淆矩阵
        cm = confusion_matrix(target_binary, pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 特异性和敏感性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'confusion_matrix': cm
        }
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """打印指标"""
        print(f"\n{prefix} Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  TP: {metrics['true_positive']}, TN: {metrics['true_negative']}")
        print(f"  FP: {metrics['false_positive']}, FN: {metrics['false_negative']}")
        
    def plot_confusion_matrix(self, cm, save_path, title="Confusion Matrix"):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ==================== WandB配置 ====================
def wandb_config(model_name, num_heads, hidden_size, batch_size, attn_version):
    wandb.login()
    wandb.init(project=f"afba_{model_name}_hardneg", entity="15145202826-1")
    wandb.run.name = (f'nh_{num_heads}_hs_{hidden_size}_bs_{batch_size}_'
                      f'attn_{attn_version}_seed_{SEED}_scheduler_{scheduler_type}')
    wandb.config = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "patience": patience,
        "grad_clip": grad_clip_max_norm,
        "scheduler": scheduler_type,
        "attention_version": attn_version,
    }
    wandb.define_metric("epoch")
    wandb.define_metric("val_f1_score", step_metric="epoch", summary="max")

# ==================== 数据加载 ====================
if mode != 'af':
    transform = Normalize(
        mean=[17.952442, 26.94709, 19.82838, 317.80234, 308.47693, 13.87255, 291.0257, 288.9398],
        std=[15.359564, 14.336508, 10.64194, 12.505946, 11.571564, 9.666024, 11.495529, 7.9788895]
    )
else:
    transform = Normalize(
        mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
        std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657]
    )


# ==================== 模型初始化 ====================
def create_model(model_name, n_channel, num_classes, hidden_size, num_heads,
                 ts_length, attn_version):
    """创建模型"""

    if model_name == 'swin_convlstm_improved' or model_name == 'swin_convlstm':
        # 统一逻辑，不再区分 improved，都使用 swin_convlstm.py 中的类
        print(f"Initializing SwinConvLSTM with {n_channel}-channel input + {attn_version.upper()}")
        model = SwinConvLSTM(
            image_size=(256, 256),
            in_channels=n_channel, # 确保传入 8
            out_channels=num_classes,
            feature_size=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            hidden_dim=64,
            dropout=0.1,
            attn_version=attn_version,
        )
        # 权重加载逻辑移到主循环中调用 model.smart_load_weights

    elif model_name == 'swinunetr3d':
        image_size = (ts_length, 256, 256)
        patch_size = (1, 2, 2)
        window_size = (1, 4, 4)

        model = SwinUNETR(
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            in_channels=n_channel,
            out_channels=num_classes,
            depths=(1, 1, 2, 1),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name='batch',
            drop_rate=0.1,
            attn_drop_rate=0.1,
            spatial_dims=3,
            use_checkpoint=True,
        )

    elif model_name == 'unet3d':
        model = UNet(
            spatial_dims=3, 
            in_channels=n_channel, 
            out_channels=num_classes, 
            channels=(64, 128, 256, 512, 1024), 
            strides=(1, 2, 2)
        )
    elif model_name == 'attunet3d':
        model = AttentionUnet(
            spatial_dims=3, 
            in_channels=n_channel, 
            out_channels=num_classes, 
            channels=(64, 128, 256, 512, 1024), 
            strides=(1, 2, 2)
        )
    elif model_name == 'unetr3d':
        image_size = (ts_length, 256, 256)
        model = UNETR(
            in_channels=n_channel,
            out_channels=num_classes,
            img_size=image_size,
            spatial_dims=3,
            norm_name='batch',
            feature_size=hidden_size,
            patch_size=(1, 16, 16)
        )
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    return model


import timm
from collections import OrderedDict

# 删除了 load_pretrained_weights_to_improved_model 和 load_timm_swin_to_swinconvlstm 函数，
# 避免与 swin_convlstm.py 中的 smart_load_weights 冲突


# ==================== 统一学习率参数组 ====================
def get_unified_lr_groups(model, base_lr):
    """
    所有参数使用统一学习率
    
    Args:
        model: 模型
        base_lr: 基础学习率

    Returns:
        list: 参数组列表（所有参数统一学习率）
    """
    all_params = [param for param in model.parameters() if param.requires_grad]
    
    param_groups = [
        {'params': all_params, 'lr': base_lr, 'name': 'unified'}
    ]
    
    return param_groups


def update_freeze_status_all_trainable(model, epoch):
    """
    所有权重一起训练，无需解冻策略
    
    Args:
        model: 模型
        epoch: 当前 epoch

    Returns:
        bool: 是否执行了操作（始终为False）
    """
    if isinstance(model, nn.DataParallel):
        model_unwrapped = model.module
    else:
        model_unwrapped = model
    
    # 确保所有参数都是可训练的
    if epoch == 1:
        for param in model_unwrapped.parameters():
            param.requires_grad = True
            
        count_parameters(model)
    
    return False


# ==================== 训练函数 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    grad_clip_max_norm, epoch, max_epochs, scheduler=None):
    """
    训练回合
    
    改进点:
    - 支持按 batch step 更新的 scheduler (Cosine Annealing)
    - 使用 Copy-Paste 数据增强
    - 支持新的改进损失函数和Deep Supervision

    Args:
        model: 训练模型
        dataloader: 数据加载器
        criterion: 损失函数（ImprovedLoss）
        optimizer: 优化器
        scaler: GradScaler（混合精度训练）
        device: 设备
        grad_clip_max_norm: 梯度裁剪阈值
        epoch: 当前 epoch
        max_epochs: 总 epoch 数
        scheduler: 学习率调度器（可选，按 batch step）

    Returns:
        total_loss: 平均总损失
        loss_components: 包含各损失组件的字典
    """
    model.train()

    # 初始化损失组件记录
    loss_components = {
        'tversky_loss': 0.0,
        'focal_loss': 0.0,
        'ce_loss': 0.0,
        'fire_weight': 0.0,
        'fire_ratio': 0.0
    }
    total_loss = 0.0

    train_bar = tqdm(dataloader, total=len(dataloader),
                     desc=f"Training Epoch {epoch}/{max_epochs}")

    for i, batch in enumerate(train_bar):
        # 1. 准备数据
        data_batch = batch['data'].to(device)
        labels_batch = batch['labels'].to(torch.long).to(device)

        optimizer.zero_grad()

        with autocast():
            # 前向传播
            outputs = model(data_batch)
            loss, loss_dict = criterion(outputs, labels_batch)

            # 转换 tensor 为 float
            for k in loss_dict:
                if torch.is_tensor(loss_dict[k]):
                    loss_dict[k] = loss_dict[k].item()

        # 3. 反向传播
        scaler.scale(loss).backward()

        # 4. 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        # 5. 优化器步进
        scaler.step(optimizer)
        scaler.update()
        
        # 6. Step Scheduler (Per Batch) - 用于 Cosine Annealing
        if scheduler is not None:
            scheduler.step()

        # 7. 累加指标
        batch_size = data_batch.size(0)
        total_loss += loss.item() * batch_size

        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k] * batch_size

        # 8. 更新进度条
        current_loss = loss.item()
        lr_current = optimizer.param_groups[0]["lr"]
        
        # Copy-Paste 渐进式概率
        if epoch <= 20:
            cp_prob_str = "0%"
        elif epoch <= 40:
            cp_prob_str = "20%"
        elif epoch <= 60:
            cp_prob_str = "40%"
        elif epoch <= 80:
            cp_prob_str = "50%"
        else:
            cp_prob_str = "60%"
        
        train_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{lr_current:.6f}',
            'copy_paste': cp_prob_str
        })

    # 9. 计算平均值
    num_samples = len(dataloader.dataset)
    total_loss /= num_samples

    for k in loss_components:
        loss_components[k] /= num_samples

    return total_loss, loss_components


# ==================== 验证函数 ====================
def validate(model, dataloader, criterion, device, post_trans, epoch, max_epochs):
    model.eval()
    val_loss = 0.0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    loss_components = {'tversky_loss': 0.0, 'focal_loss': 0.0, 'ce_loss': 0.0, 'fire_weight': 0.0, 'fire_ratio': 0.0}
    
    mean_iou = MeanIoU(include_background=False, reduction='mean')
    mean_dice = DiceMetric(include_background=False, reduction='mean')

    val_bar = tqdm(dataloader, total=len(dataloader), desc=f"Validation Epoch {epoch}/{max_epochs}")

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            data_batch = batch['data'].to(device)
            labels_batch = batch['labels'].to(device)

            with autocast():
                outputs = model(data_batch)
                
                if isinstance(outputs, (list, tuple)):
                    val_outputs = outputs[0]
                    loss, loss_dict = criterion(outputs, labels_batch)
                else:
                    val_outputs = outputs
                    loss, loss_dict = criterion(outputs, labels_batch)

            val_loss += loss.item() * data_batch.size(0)
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key] * data_batch.size(0)

            probs = torch.softmax(val_outputs, dim=1)[:, 1, ...]
            preds = (probs > 0.5).long()
            
            if labels_batch.shape[1] == 1:
                targets = labels_batch.squeeze(1)
            else:
                targets = labels_batch[:, 0, ...]
            
            # 收集所有概率和标签用于多阈值计算
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            
            valid_mask = (targets != -1)
            preds_masked = preds[valid_mask]
            targets_masked = targets[valid_mask]
            
            if targets_masked.numel() > 0:
                targets_binary = (targets_masked > 0).long()
                tp = (preds_masked * targets_binary).sum().item()
                fp = (preds_masked * (1 - targets_binary)).sum().item()
                fn = ((1 - preds_masked) * targets_binary).sum().item()
                tn = ((1 - preds_masked) * (1 - targets_binary)).sum().item()
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

            outputs_list = decollate_batch(val_outputs)
            labels_binary = torch.clamp(labels_batch, 0, 1).long()
            labels_binary_list = decollate_batch(labels_binary)
            label_post_trans = Compose([AsDiscrete(to_onehot=2)])
            labels_post = [label_post_trans(l) for l in labels_binary_list]
            outputs_post = [post_trans(o) for o in outputs_list]
            mean_iou(outputs_post, labels_post)
            mean_dice(outputs_post, labels_post)

            val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    val_loss /= len(dataloader.dataset)
    for key in loss_components:
        loss_components[key] /= len(dataloader.dataset)

    mean_iou_val = mean_iou.aggregate().item()
    mean_dice_val = mean_dice.aggregate().item()
    mean_iou.reset()
    mean_dice.reset()

    epsilon = 1e-6
    precision = float(total_tp / (total_tp + total_fp + epsilon))
    recall = float(total_tp / (total_tp + total_fn + epsilon))
    f1 = float(2 * (precision * recall) / (precision + recall + epsilon))
    iou = float(total_tp / (total_tp + total_fp + total_fn + epsilon))
    
    # 多阈值计算
    thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    best_f1 = f1
    best_threshold = 0.5
    best_precision_val = precision
    best_recall_val = recall
    best_iou_val = iou
    
    # 合并所有概率和标签
    if all_probs and all_targets:
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        # 展平并应用有效掩码
        probs_flat = all_probs_tensor.flatten()
        targets_flat = all_targets_tensor.flatten()
        valid_mask = (targets_flat != -1)
        probs_valid = probs_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        
        if targets_valid.numel() > 0:
            targets_binary = (targets_valid > 0).long()
            
            for threshold in thresholds:
                preds_thresh = (probs_valid > threshold).long()
                tp = (preds_thresh * targets_binary).sum().item()
                fp = (preds_thresh * (1 - targets_binary)).sum().item()
                fn = ((1 - preds_thresh) * targets_binary).sum().item()
                
                precision_thresh = tp / (tp + fp + epsilon)
                recall_thresh = tp / (tp + fn + epsilon)
                f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh + epsilon)
                iou_thresh = tp / (tp + fp + fn + epsilon)
                
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = threshold
                    best_precision_val = precision_thresh
                    best_recall_val = recall_thresh
                    best_iou_val = iou_thresh
    
    custom_metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
        'specificity': float(total_tn / (total_tn + total_fp + epsilon)),
        'sensitivity': float(recall),
        'true_positive': int(total_tp),
        'true_negative': int(total_tn),
        'false_positive': int(total_fp),
        'false_negative': int(total_fn),
        'confusion_matrix': np.array([[total_tn, total_fp], [total_fn, total_tp]]),
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'best_precision': float(best_precision_val),
        'best_recall': float(best_recall_val),
        'best_iou': float(best_iou_val),
    }

    return val_loss, loss_components, mean_iou_val, mean_dice_val, custom_metrics


def find_optimal_threshold(model, dataloader, device):
    """Compute optimal threshold by evaluating F1 over thresholds 0.2-0.8 (step 0.05) using all valid pixels."""
    print(f"  - Calculating optimal threshold on {device} using all valid pixels...")
    model.eval()
    thresholds = np.arange(0.2, 0.801, 0.05)
    tp_counts = np.zeros_like(thresholds, dtype=np.int64)
    fp_counts = np.zeros_like(thresholds, dtype=np.int64)
    fn_counts = np.zeros_like(thresholds, dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Threshold Search", leave=False):
            data = batch['data'].to(device)
            labels = batch['labels'].to(device)
            with autocast():
                outputs = model(data)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)[:, 1, ...]
            # Determine target tensor shape
            targets = labels.squeeze(1) if labels.shape[1] == 1 else labels[:, 0, ...]
            # Flatten and mask invalid pixels
            probs_flat = probs.flatten()
            targets_flat = targets.flatten()
            valid_mask = (targets_flat != -1)
            probs_valid = probs_flat[valid_mask]
            targets_valid = (targets_flat[valid_mask] > 0).long()
            if targets_valid.numel() == 0:
                continue
            probs_np = probs_valid.cpu().numpy()
            targets_np = targets_valid.cpu().numpy()
            for i, thr in enumerate(thresholds):
                preds = (probs_np > thr).astype(np.int8)
                tp_counts[i] += np.sum((preds == 1) & (targets_np == 1))
                fp_counts[i] += np.sum((preds == 1) & (targets_np == 0))
                fn_counts[i] += np.sum((preds == 0) & (targets_np == 1))
    epsilon = 1e-6
    f1_scores = 2 * tp_counts / (2 * tp_counts + fp_counts + fn_counts + epsilon)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  - Found best threshold: {best_thr:.2f} (F1: {best_f1:.4f})")
    return best_thr, best_f1






# ==================== 主训练循环 ====================
# 初始化WandB
wandb_config(model_name, num_heads, hidden_size, batch_size, attn_version)

# 加载数据
print("Loading training data...")
image_path = os.path.join(root_path, f'dataset_train/{mode}_train_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
label_path = os.path.join(root_path, f'dataset_train/{mode}_train_label_seqtoseq_alll_{ts_length}i_{interval}.npy')
val_image_path = os.path.join(root_path, f'dataset_val/{mode}_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy')
val_label_path = os.path.join(root_path, f'dataset_val/{mode}_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy')

label_sel = 2 if mode == 'af' else 0

train_dataset = FireDataset(
    image_path=image_path,
    label_path=label_path,
    ts_length=ts_length,
    transform=transform,
    n_channel=n_channel,
    label_sel=label_sel,
    is_train=True,
    crop_size=224,
)
print(f"训练集样本数: {len(train_dataset)}")
print(f"原始图像文件总样本数: {np.load(image_path, mmap_mode='r').shape[0]}")
print(f"原始标签文件总样本数: {np.load(label_path, mmap_mode='r').shape[0]}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_dataset = FireDataset(
    image_path=val_image_path,
    label_path=val_label_path,
    ts_length=ts_length,
    transform=transform,
    n_channel=n_channel,
    label_sel=label_sel,
    is_train=False,
    crop_size=256,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

print("Creating model...")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

model = create_model(
    model_name, n_channel, num_classes, hidden_size,
    num_heads, ts_length, attn_version
)

count_parameters(model)
pretrained_path = '/home/congwei/swin_fire/swin_pretrained/swin_tiny_patch4_window7_224.pth'
# [必须] 调用智能权重加载
if os.path.exists(pretrained_path) and hasattr(model, 'smart_load_weights'):
    print(f"- Loading pretrained weights from: {pretrained_path}")
    model.smart_load_weights(pretrained_path)
    print(f"- Successfully loaded pretrained weights for {model_name}")
else:
    print(f"- Warning: Pretrained weights not loaded. Path: {pretrained_path}")

count_parameters(model)

# model = nn.DataParallel(model)
model.to(device)

print(f'Number of Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

# Using HybridLoss (Reverted to 2.19 logic + CE)
criterion = HybridLoss(
    tversky_weight=0.4,
    focal_weight=0.3,
    ce_weight=0.3
)

# Use unified learning rate parameter groups
param_groups = get_unified_lr_groups(model, lr)
optimizer = optim.AdamW(
    param_groups,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)
print("Created optimizer with unified learning rate:")
print(f"   Unified LR: {lr:.2e} (managed by CosineAnnealingWarmRestartsWithDecay)")

# Use CosineAnnealingWarmRestartsWithDecay
scheduler = CosineAnnealingWarmRestartsWithDecay(
        optimizer,
        T_0=MAX_EPOCHS,
        T_mult=1,
        decay_factor=args.decay_factor,
        eta_min=1e-7
)
print(f"Using CosineAnnealingWarmRestartsWithDecay with decay_factor={args.decay_factor} (no warmup).")

scaler = GradScaler()

post_trans = Compose([
    Activations(softmax=True),
    AsDiscrete(argmax=True, to_onehot=2)
])

early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

# 保存最佳模型：基于 F1 score，只保存一个最佳模型
best_f1 = 0.0
best_model_path = None

# 创建保存目录
os.makedirs('saved_models', exist_ok=True)
os.makedirs('confusion_matrices', exist_ok=True)

print("\n" + "="*50)
print("Starting Training...")
print("="*50 + "\n")

# 训练循环
for epoch in range(MAX_EPOCHS):
    # Update Copy-Paste probability based on epoch
    train_dataset.set_epoch(epoch + 1)
    val_dataset.set_epoch(epoch + 1)

    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    print("-" * 50)

    stage_changed = update_freeze_status_all_trainable(model, epoch + 1)
    
    if stage_changed and epoch > 0:
        print(f"- Stage Changed! Reducing LR to 0.1x to protect newly unfrozen layers.")
        
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            
            if 'initial_lr' in param_group:
                param_group['initial_lr'] = param_group['initial_lr'] * 0.1
            
            print(f"   Group {i} ({param_group.get('name', 'unknown')}): {old_lr:.2e} -> {new_lr:.2e}")
    
    # Since we now use a unified learning rate group, retrieve the single LR value
    unified_lr = optimizer.param_groups[0]["lr"]
    print(f"  - Learning Rate: {unified_lr:.2e}")
    
    # 训练
    train_loss, train_loss_components = train_one_epoch(
        model, train_dataloader, criterion, optimizer, scaler, device,
        grad_clip_max_norm, epoch + 1, MAX_EPOCHS, 
        scheduler=scheduler
    )
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"  - Tversky: {train_loss_components['tversky_loss']:.4f}")
    print(f"  - Focal: {train_loss_components['focal_loss']:.4f}")
    print(f"  - CE: {train_loss_components['ce_loss']:.4f}")
    print(f"  - Fire Weight: {train_loss_components['fire_weight']:.4f}")
    print(f"  - Fire Ratio: {train_loss_components['fire_ratio']:.4f}")
    
    # 验证
    val_loss, val_loss_components, mean_iou_val, mean_dice_val, custom_metrics = validate(
        model, val_dataloader, criterion, device, post_trans, epoch + 1, MAX_EPOCHS
    )
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"  - Tversky: {val_loss_components['tversky_loss']:.4f}")
    print(f"  - Focal: {val_loss_components['focal_loss']:.4f}")
    print(f"  - CE: {val_loss_components['ce_loss']:.4f}")
    print(f"  - Fire Weight: {val_loss_components['fire_weight']:.4f}")
    print(f"  - Fire Ratio: {val_loss_components['fire_ratio']:.4f}")
    print(f"Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}")
    
    # 打印自定义指标
    print(f"\nValidation Metrics:")
    print(f"  Precision: {custom_metrics['precision']:.4f}")
    print(f"  Recall: {custom_metrics['recall']:.4f}")
    print(f"  F1 Score: {custom_metrics['f1_score']:.4f}")
    print(f"  IoU: {custom_metrics['iou']:.4f}")
    print(f"  Specificity: {custom_metrics['specificity']:.4f}")
    print(f"  Sensitivity: {custom_metrics['sensitivity']:.4f}")
    print(f"  TP: {custom_metrics['true_positive']}, TN: {custom_metrics['true_negative']}")
    print(f"  FP: {custom_metrics['false_positive']}, FN: {custom_metrics['false_negative']}")
    
    # 打印最佳阈值结果
    print(f"\nBest Threshold Results:")
    print(f"  Best Threshold: {custom_metrics['best_threshold']:.2f}")
    print(f"  Best Precision: {custom_metrics['best_precision']:.4f}")
    print(f"  Best Recall: {custom_metrics['best_recall']:.4f}")
    print(f"  Best F1 Score: {custom_metrics['best_f1']:.4f}")
    print(f"  Best IoU: {custom_metrics['best_iou']:.4f}")

    # 获取当前学习率
    encoder_lr = decoder_lr = None
    for param_group in optimizer.param_groups:
        if 'encoder' in param_group.get('name', ''):
            encoder_lr = param_group['lr']
        elif 'decoder' in param_group.get('name', ''):
            decoder_lr = param_group['lr']

    # 记录到WandB
    log_dict = {
        'epoch': epoch + 1,
        'encoder_lr': encoder_lr,
        'decoder_lr': decoder_lr,
        'train_loss': train_loss,
        'train_tversky_loss': train_loss_components['tversky_loss'],
        'train_focal_loss': train_loss_components['focal_loss'],
        'train_ce_loss': train_loss_components['ce_loss'],
        'train_fire_weight': train_loss_components['fire_weight'],
        'train_fire_ratio': train_loss_components['fire_ratio'],
        'val_loss': val_loss,
        'val_tversky_loss': val_loss_components['tversky_loss'],
        'val_focal_loss': val_loss_components['focal_loss'],
        'val_ce_loss': val_loss_components['ce_loss'],
        'val_fire_weight': val_loss_components['fire_weight'],
        'val_fire_ratio': val_loss_components['fire_ratio'],
        'val_miou': mean_iou_val,
        'val_mdice': mean_dice_val,
        'val_precision': custom_metrics['precision'],
        'val_recall': custom_metrics['recall'],
        'val_f1_score': custom_metrics['f1_score'],
        'val_iou': custom_metrics['iou'],
        'val_specificity': custom_metrics['specificity'],
        'val_sensitivity': custom_metrics['sensitivity'],
        'val_best_threshold': custom_metrics['best_threshold'],
        'val_best_f1': custom_metrics['best_f1'],
        'val_best_precision': custom_metrics['best_precision'],
        'val_best_recall': custom_metrics['best_recall'],
        'val_best_iou': custom_metrics['best_iou'],
        'global_best_f1': early_stopping.best_f1,
    }
    
    try:
        wandb.log(log_dict)
    except Exception as e:
        print(f"- WandB logging failed: {e}")
    
    # Cosine Scheduler 已在 train_one_epoch 中按 batch 更新
    # 打印当前学习率
    # 使用统一的学习率组，直接获取当前学习率
    unified_lr = optimizer.param_groups[0]["lr"]
    print(f"  - End of Epoch LR: {unified_lr:.2e}")
    
    # 保存混淆矩阵
    if (epoch + 1) % 10 == 0:
        metrics_calc = MetricsCalculator()
        cm_path = f'confusion_matrices/cm_epoch_{epoch + 1}_seed_{SEED}.png'
        metrics_calc.plot_confusion_matrix(
            custom_metrics['confusion_matrix'], 
            cm_path, 
            title=f'Confusion Matrix - Epoch {epoch + 1}'
        )
        wandb.log({f'confusion_matrix_epoch_{epoch + 1}': wandb.Image(cm_path)})
        cm_path = f'confusion_matrices/cm_epoch_{epoch + 1}_seed_{SEED}.png'
        metrics_calc.plot_confusion_matrix(
            custom_metrics['confusion_matrix'], 
            cm_path, 
            title=f'Confusion Matrix - Epoch {epoch + 1}'
        )
        wandb.log({f'confusion_matrix_epoch_{epoch + 1}': wandb.Image(cm_path)})
    
    # 保存最佳检查点（基于 F1 score，只保存一个最佳模型）
    current_f1 = custom_metrics['f1_score']

    if current_f1 > best_f1 and epoch >= 30:
        print(f"  - New best performance detected (F1: {current_f1:.4f}). Preparing to save...")

        # Delete old best model if exists
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
            print(f"  - Removed old best model: {best_model_path}")

        # Move model to CPU for threshold search to avoid GPU memory issues
        model_cpu = model.module.cpu() if isinstance(model, nn.DataParallel) else model.cpu()

        best_threshold, best_f1_optimized = find_optimal_threshold(
            model_cpu, val_dataloader, torch.device('cpu')
        )

        # Move model back to GPU
        model_cpu.to(device)
        if isinstance(model, nn.DataParallel):
            model.module.to(device)
        else:
            model.to(device)

        custom_metrics['best_threshold'] = best_threshold
        custom_metrics['best_f1'] = best_f1_optimized

        best_f1 = current_f1
        best_model_path = (f"saved_models/model_{model_name}_run_{run}_seed_{SEED}_mode_{mode}_"
                    f"nh_{num_heads}_hs_{hidden_size}_bs_{batch_size}_av_{attn_version}_"
                    f"epoch_{epoch + 1}_f1_{current_f1:.4f}_thr_{best_threshold:.2f}.pth")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': custom_metrics,
            'best_threshold': best_threshold,
            'f1_score': current_f1,
        }, best_model_path)

        print(f"  - Saved best model: {best_model_path}")
        print(f"  - F1 Score: {current_f1:.4f}, Best Threshold: {best_threshold:.2f}")
    
    # 早停检查（只监控 F1）
    if early_stopping(custom_metrics['f1_score']):
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print("\n" + "="*50)
print("Training Completed!")
print("="*50)
if best_model_path:
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Best F1 Score: {best_f1:.4f}")

wandb.finish()

