"""
SwinConvLSTM: 2D SwinUNETR + ConvLSTM 时序火灾检测模型 (Pro版)
针对 ts-satfire 数据集优化：
1. 保留 8 通道完整输入 (不使用 InputAdapter 降维)
2. 支持智能加载 3 通道预训练权重 (Smart Weight Loading)
3. 集成 CBAM 注意力增强空间特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
import logging

# 配置日志
logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力 (v2)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ARBlock(nn.Module):
    """Attention Refinement (轻量级通道注意力)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class CBAM(nn.Module):
    """
    通道-空间双重注意力机制 (Convolutional Block Attention Module)
    作用：
    1. Channel Attention: 自动加权重要的光谱波段 (如 SWIR)
    2. Spatial Attention: 聚焦火点纹理，抑制背景噪声
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力 (7x7 卷积)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 1. Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        mc = self.sigmoid(avg_out + max_out)
        x = x * mc
        # 2. Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ms = torch.cat([avg_pool, max_pool], dim=1)
        ms = self.sigmoid(self.conv_spatial(ms))
        return x * ms


class DCBAM(nn.Module):
    """
    空洞卷积版本的 CBAM (DCBAM)
    将空间注意力的标准卷积替换为空洞卷积，扩大感受野，参数量不变。
    """
    def __init__(self, channels, reduction=16, dilation=2, kernel_size=7):
        super().__init__()
        # 通道注意力 (与 CBAM 相同)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力 (使用空洞卷积)
        # 为了保持输出尺寸不变，padding 需根据 dilation 调整：
        # padding = dilation * (kernel_size - 1) // 2
        self.conv_spatial = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False
        )

    def forward(self, x):
        # 1. 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        mc = self.sigmoid(avg_out + max_out)
        x = x * mc

        # 2. 空间注意力 (空洞卷积)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ms = torch.cat([avg_pool, max_pool], dim=1)
        ms = self.sigmoid(self.conv_spatial(ms))
        return x * ms


class DCBAM_ASPP(nn.Module):
    """
    多尺度空洞卷积版本的 CBAM (DCBAM_ASPP)
    类似 ASPP (Atrous Spatial Pyramid Pooling)，使用不同空洞率的多分支结构
    捕捉更丰富的上下文信息。
    """
    def __init__(self, channels, reduction=16, dilations=[1, 2, 3, 4]):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_convs = nn.ModuleList()
        for d in dilations:
            pad = d * (7 - 1) // 2
            self.spatial_convs.append(
                nn.Conv2d(2, 1, kernel_size=7, padding=pad, dilation=d, bias=False)
            )
        self.fusion = nn.Conv2d(len(dilations), 1, kernel_size=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        mc = self.sigmoid(avg_out + max_out)
        x = x * mc

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ms = torch.cat([avg_pool, max_pool], dim=1)

        spa_outs = [conv(ms) for conv in self.spatial_convs]
        spa_out = torch.cat(spa_outs, dim=1)
        spa_out = self.fusion(spa_out)
        spa_out = self.sigmoid(spa_out)
        return x * spa_out


class DualAttention(nn.Module):
    """
    双重注意力模块 (Dual Attention)
    包含：
    1. 位置注意力（Position Attention）：建模任意两点之间的空间依赖
    2. 通道注意力（Channel Attention）：建模任意两通道之间的通道依赖
    最后通过元素和融合两个注意力输出，并经过残差连接得到最终输出。
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # 位置注意力分支
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # 通道注意力分支
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # ---------- 位置注意力 ----------
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C/r]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C/r, HW]
        energy = torch.bmm(proj_query, proj_key)  # [B, HW, HW]
        attention = F.softmax(energy, dim=-1)  # 空间注意力图
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out_pos = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, HW]
        out_pos = out_pos.view(B, C, H, W)
        out_pos = self.gamma * out_pos + x  # 残差连接

        # ---------- 通道注意力 ----------
        proj_query = x.view(B, C, H * W)  # [B, C, HW]
        proj_key = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        energy = torch.bmm(proj_query, proj_key)  # [B, C, C]
        attention = F.softmax(energy, dim=-1)  # 通道注意力图
        proj_value = x.view(B, C, H * W)  # [B, C, HW]
        out_cha = torch.bmm(attention.permute(0, 2, 1), proj_value)  # [B, C, HW]
        out_cha = out_cha.view(B, C, H, W)
        out_cha = self.beta * out_cha + x  # 残差连接

        # 融合两个分支（元素相加）
        out = out_pos + out_cha
        return out


def get_attention_module(attn_version, channels, reduction=16):
    """工厂函数：获取注意力模块"""
    if attn_version == 'v1' or attn_version == 'none':
        return nn.Identity()
    elif attn_version == 'v2' or attn_version == 'se':
        return SEBlock(channels, reduction)
    elif attn_version == 'ar':
        return ARBlock(channels, reduction)
    elif attn_version == 'cbam':
        return CBAM(channels, reduction)
    elif attn_version == 'dual':
        return DualAttention(channels, reduction)
    elif attn_version == 'dcbam':
        return DCBAM(channels, reduction)
    elif attn_version == 'dcbam_aspp':
        return DCBAM_ASPP(channels, reduction)
    else:
        print(f"Warning: Unknown attention version '{attn_version}', using CBAM")
        return CBAM(channels, reduction)


class ConvLSTMCell(nn.Module):
    """ConvLSTM 单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SwinConvLSTM(nn.Module):
    """
    2D SwinUNETR + ConvLSTM 时序火灾检测模型
    """
    def __init__(
        self,
        image_size=(256, 256),
        in_channels=8,  # 默认为 8 通道，不要改回 3
        out_channels=2,
        feature_size=48,  # Swin 输出特征维度
        depths=(2, 2, 6, 2),  # Swin 深度配置
        num_heads=(3, 6, 12, 24),
        hidden_dim=64,  # LSTM 隐藏层维度
        dropout=0.1,
        attn_version='cbam',
    ):
        super().__init__()
        print(f" [Model Init] SwinConvLSTM (Input: {in_channels}ch)")
        # 1. 编码器：使用 MONAI SwinUNETR 提取空间特征
        # 注意：这里我们只用它做特征提取，所以 out_channels 设置为 feature_size
        self.swin_encoder = MonaiSwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=feature_size,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name='instance',
            drop_rate=dropout,
            attn_drop_rate=dropout,
            spatial_dims=2,
            use_checkpoint=True,
        )
        # 2. 注意力增强 (在进入 LSTM 前净化特征)
        self.attention = get_attention_module(attn_version, feature_size, reduction=8)
        # 3. 时序聚合
        self.convlstm = ConvLSTMCell(
            input_dim=feature_size,
            hidden_dim=hidden_dim,
            kernel_size=3
        )
        # 4. 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, out_channels, 1)
        )
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        # 初始化状态
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :, :]  # [B, 8, H, W]
            # 提取特征
            feat_t = self.swin_encoder(x_t)  # [B, feature_size, H, W]
            # 注意力增强
            feat_t = self.attention(feat_t)
            # 时序更新
            h, c = self.convlstm(feat_t, h, c)
            # 预测
            out_t = self.seg_head(h)
            outputs.append(out_t)
        outputs = torch.stack(outputs, dim=2)  # [B, 2, T, H, W]
        return outputs

    def smart_load_weights(self, checkpoint_path):
        import torch
        import re
        print(f" [Smart Load] Loading weights from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except FileNotFoundError:
            print(f" Error: Pretrained path not found: {checkpoint_path}")
            return

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model_dict = self.state_dict()
        new_state_dict = {}
        loaded_count = 0
        skipped_count = 0

        for k, v in state_dict.items():
            # 1. 移除可能的前缀（如 module.）
            if k.startswith('module.'):
                k = k[7:]

            # 2. 特殊处理：patch_embed.norm.* 直接跳过（MONAI 没有）
            if 'patch_embed.norm' in k:
                skipped_count += 1
                continue

            # 3. 转换 MLP 层命名
            k = k.replace('mlp.fc1', 'mlp.linear1')
            k = k.replace('mlp.fc2', 'mlp.linear2')

            # 4. 处理 stage 索引偏移（layers.X -> layers{X+1}.0）
            pattern = r'layers\.(\d+)\.'
            match = re.search(pattern, k)
            if match:
                stage_idx = int(match.group(1))
                new_stage = f'layers{stage_idx + 1}.0'   # MONAI 中格式为 layers1.0.blocks...
                k = re.sub(pattern, f'{new_stage}.', k)

            # 5. 添加公共前缀
            model_k = 'swin_encoder.swinViT.' + k

            # 6. 检查模型字典中是否存在
            if model_k not in model_dict:
                skipped_count += 1
                continue

            # 7. 特殊处理 patch_embed.proj.weight（通道扩展 + 空间插值）
            if 'patch_embed.proj.weight' in model_k:
                target_shape = model_dict[model_k].shape  # [96, 8, 2, 2]
                if v.shape != target_shape:
                    new_weight = torch.zeros(target_shape, dtype=v.dtype)

                    # 空间尺寸调整
                    if v.shape[2:] != target_shape[2:]:
                        v_resized = torch.nn.functional.interpolate(
                            v, size=target_shape[2:], mode='bilinear', align_corners=False
                        )
                    else:
                        v_resized = v

                    # 复制前3通道，其余用均值填充
                    new_weight[:, :3, :, :] = v_resized
                    mean_weight = torch.mean(v_resized, dim=1, keepdim=True)
                    new_weight[:, 3:, :, :] = mean_weight.repeat(1, target_shape[1] - 3, 1, 1)

                    new_state_dict[model_k] = new_weight
                    loaded_count += 1
                    continue

            # 8. 普通层：检查形状是否一致
            if model_dict[model_k].shape == v.shape:
                new_state_dict[model_k] = v
                loaded_count += 1
            else:
                # 对于相对位置编码表等特殊张量，如果形状不同可以尝试插值，但这里先跳过
                skipped_count += 1

        # 9. 更新模型
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)
        print(f" - Successfully loaded {loaded_count} layers, skipped {skipped_count} layers.")
