import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Normalize:
    """
    归一化变换类
    对输入图像按通道进行标准化
    """
    def __init__(self, mean, std):
        """
        Args:
            mean: list/array, 每个通道的均值
            std: list/array, 每个通道的标准差
        """
        self.mean = np.array(mean).reshape(-1, 1, 1, 1)  # [C, 1, 1, 1]
        self.std = np.array(std).reshape(-1, 1, 1, 1)
        
    def __call__(self, image):
        """
        Args:
            image: numpy array, shape [C, T, H, W]
        Returns:
            normalized image
        """
        # Z-score 归一化
        return (image - self.mean) / (self.std + 1e-8)



class FireDataset(Dataset):
   
    def __init__(
        self, 
        image_path, 
        label_path, 
        ts_length, 
        transform=None, 
        n_channel=8, 
        label_sel=0,
        is_train=True,  
        crop_size=224,
        dilation_range=(3, 9),
        dilation_grow_epochs=80,
        use_random_dilation=False,
    ):
        self.images = np.load(image_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.ts_length = ts_length
        self.transform = transform
        self.n_channel = n_channel
        self.label_sel = label_sel
        self.is_train = is_train
        self.crop_size = crop_size
        self.dilation_range = dilation_range
        self.dilation_grow_epochs = dilation_grow_epochs
        self.use_random_dilation = use_random_dilation
        
       
        self.aug_probs = {
            'flip': 0.5,
            'rotate': 0.5,
            'color': 0.3,
            'noise': 0.2,
            'channel_drop': 0.1,
            'time_mask': 0.1,
            'fire_crop': 0.8,
            'copy_paste': 0.0
        }
        # self.aug_probs = {
        #     'flip': 0.0,
        #     'rotate': 0.0,
        #     'color': 0.0,
        #     'noise': 0.0,
        #     'channel_drop': 0.0,
        #     'time_mask': 0.0,
        #     'fire_crop': 0.0,
        #     'copy_paste': 0.0
        # }
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """设置当前epoch，更新Copy-Paste概率"""
        self.current_epoch = epoch
        if epoch <= 40:
            self.aug_probs['copy_paste'] = 0.0
        elif epoch <= 60:
            self.aug_probs['copy_paste'] = 0.2
        elif epoch <= 80:
            self.aug_probs['copy_paste'] = 0.3
        elif epoch <= 100:
            self.aug_probs['copy_paste'] = 0.4
        else:
            self.aug_probs['copy_paste'] = 0.5
          
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        # 1. 加载数据 (Copy 确保内存连续且可修改)
        # image: [C, T, H, W], label: [1, T, H, W]
        image = self.images[idx][:self.n_channel, :self.ts_length, :, :].copy()
        label = self.labels[idx][self.label_sel:self.label_sel+1, :self.ts_length, :, :].copy()
        
        # 2. 归一化 (建议在增强前做，或者增强后做，取决于 transform 的实现)
        # 这里保持你的逻辑：先归一化
        if self.transform:
            image = self.transform(image)
            
        # 转 Tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        # 3. 数据增强
        if self.is_train:
            image, label = self._apply_augmentation(image, label)
        else:
            # 验证集：中心裁剪或不裁剪
            image, label = self._center_crop(image, label, self.crop_size)
            
        return {'data': image, 'labels': label}

    def _apply_augmentation(self, image, label):
        """应用组合增强"""
        # A. 几何增强 (Geometric)
        # 1. 随机翻转 (Flip)
        if torch.rand(1) < self.aug_probs['flip']:
            dim = -1 if torch.rand(1) < 0.5 else -2 # H or W
            image = torch.flip(image, dims=[dim])
            label = torch.flip(label, dims=[dim])
            
        # 2. 随机旋转 (Rotate)
        if torch.rand(1) < self.aug_probs['rotate']:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            label = torch.rot90(label, k, dims=[-2, -1])
            
        # 3. 智能裁剪 (Fire-Aware Crop) - 核心改进
        image, label = self._smart_crop(image, label, self.crop_size)

        # 4. Copy-Paste 增强 (核心改进)
        if torch.rand(1) < self.aug_probs.get('copy_paste', 0.0):
            image, label = self._copy_paste(image, label)
        
        # B. 强度增强 (Intensity) - 只对图像
        if torch.rand(1) < self.aug_probs['color']:
            # 亮度 (Brightness)
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            image = image * scale
            # 偏移 (Shift)
            shift = torch.empty(1).uniform_(-0.1, 0.1).item()
            image = image + shift

        # C. 噪声与遮挡 (Noise & Dropout)
        if torch.rand(1) < self.aug_probs['noise']:
            noise = torch.randn_like(image) * 0.05
            image = image + noise
            
        # 通道丢弃 (Channel Dropout)
        if torch.rand(1) < self.aug_probs['channel_drop']:
            c_idx = torch.randint(0, self.n_channel, (1,)).item()
            image[c_idx, :, :, :] = 0.0
            
        # 时间遮挡 (Time Masking)
        if torch.rand(1) < self.aug_probs['time_mask'] and self.ts_length > 1:
            t_idx = torch.randint(0, self.ts_length, (1,)).item()
            image[:, t_idx, :, :] = 0.0
            
        return image, label

    def _smart_crop(self, image, label, crop_size):
        """
        如果图像中有火，以高概率（如80%）强制裁剪包含火灾的区域。
        """
        _, _, H, W = image.shape
        
        # 如果图像小于裁剪尺寸，做 Padding
        if H <= crop_size or W <= crop_size:
            pad_h = max(0, crop_size - H)
            pad_w = max(0, crop_size - W)
            # image pad 0, label pad -1 (ignore index)
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
            label = F.pad(label, (0, pad_w, 0, pad_h), value=-1)
            return image, label
            
        # 检查是否有火灾像素
        # label shape: [1, T, H, W] -> 压缩成 [H, W] 只要任意时间步有火即可
        fire_mask = (label > 0).any(dim=0).any(dim=0) # [H, W]
        fire_indices = torch.nonzero(fire_mask, as_tuple=False) # [N, 2] (y, x)
        
        use_fire_crop = (len(fire_indices) > 0) and (torch.rand(1) < self.aug_probs['fire_crop'])
        
        if use_fire_crop:
            # --- 策略 A: 围绕火点裁剪 ---
            # 随机选择一个火点作为锚点
            idx = torch.randint(0, len(fire_indices), (1,)).item()
            cy, cx = fire_indices[idx]
            
            # 随机偏移，确保火点在裁剪框内
            # 裁剪框左上角 (top, left) 的范围：
            # top 必须在 [cy - crop_size + 1, cy] 之间，且限制在 [0, H - crop_size]
            min_top = max(0, cy - crop_size + 1)
            max_top = min(H - crop_size, cy)
            
            min_left = max(0, cx - crop_size + 1)
            max_left = min(W - crop_size, cx)
            
            # 修正范围（防止 max < min）
            max_top = max(max_top, min_top)
            max_left = max(max_left, min_left)
            
            top = torch.randint(min_top, max_top + 1, (1,)).item()
            left = torch.randint(min_left, max_left + 1, (1,)).item()
            
        else:
            # --- 策略 B: 完全随机裁剪 (保留背景样本) ---
            top = torch.randint(0, H - crop_size + 1, (1,)).item()
            left = torch.randint(0, W - crop_size + 1, (1,)).item()
            
        return (
            image[:, :, top:top+crop_size, left:left+crop_size],
            label[:, :, top:top+crop_size, left:left+crop_size]
        )

    def _center_crop(self, image, label, crop_size):
        """验证集使用中心裁剪"""
        _, _, H, W = image.shape
        if H <= crop_size or W <= crop_size:
            pad_h = max(0, crop_size - H)
            pad_w = max(0, crop_size - W)
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
            label = F.pad(label, (0, pad_w, 0, pad_h), value=-1)
            return image, label
            
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        return (
            image[:, :, top:top+crop_size, left:left+crop_size],
            label[:, :, top:top+crop_size, left:left+crop_size]
        )

    def _copy_paste(self, target_img, target_label):
        source_idx = -1
        for _ in range(3):
            idx = np.random.randint(0, len(self.images))
            lb = self.labels[idx][self.label_sel:self.label_sel+1, :self.ts_length]
            if (lb > 0).any():
                source_idx = idx
                break
        if source_idx == -1:
            return target_img, target_label

        src_img = self.images[source_idx][:self.n_channel, :self.ts_length].copy()
        src_label = self.labels[source_idx][self.label_sel:self.label_sel+1, :self.ts_length].copy()
        if self.transform:
            src_img = self.transform(src_img)
        src_img = torch.from_numpy(src_img).float()
        src_label = torch.from_numpy(src_label).float()
        src_img, src_label = self._smart_crop(src_img, src_label, self.crop_size)

        src_fire_mask = (src_label > 0).float()

        if self.use_random_dilation:
            kernel = np.random.randint(self.dilation_range[0], self.dilation_range[1] + 1)
            if kernel % 2 == 0:
                kernel += 1
        else:
            progress = min(1.0, self.current_epoch / self.dilation_grow_epochs)
            kernel = int(self.dilation_range[0] + progress * (self.dilation_range[1] - self.dilation_range[0]))
            if kernel % 2 == 0:
                kernel = min(kernel + 1, self.dilation_range[1] + 1)

        H, W = src_img.shape[-2:]
        max_kernel = min(H, W) // 2 * 2 + 1
        kernel = min(kernel, max_kernel)

        padding = kernel // 2
        expanded_mask = F.max_pool3d(
            src_fire_mask,
            kernel_size=(1, kernel, kernel),
            stride=1,
            padding=(0, padding, padding)
        )

        target_impossible_mask = (target_label == -1).float()
        valid_paste_mask = expanded_mask * (1 - target_impossible_mask)

        if valid_paste_mask.sum() == 0:
            return target_img, target_label

        soft_mask = valid_paste_mask
        for _ in range(3):
            soft_mask = F.avg_pool3d(
                soft_mask,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1)
            )

        target_img = src_img * soft_mask + target_img * (1 - soft_mask)
        target_label = torch.where(
            target_impossible_mask == 1,
            target_label,
            torch.max(target_label, src_fire_mask)
        )

        return target_img, target_label
