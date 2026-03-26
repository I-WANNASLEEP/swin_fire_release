import argparse
import os
import numpy as np
import torch
from monai.data import decollate_batch, DataLoader
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn
from satimg_dataset_processor.data_generator_torch import FireDataset, Normalize
from sklearn.metrics import f1_score, jaccard_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def normalization(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-8)

def load_model(model_name, n_channel, num_classes, hidden_size, num_heads, ts_length, device, feature_size=48, attn_version='cbam'):
    """创建模型"""
    from spatial_models.swin_convlstm import SwinConvLSTM
    
    if model_name == 'swin_convlstm' or model_name == 'swin_convlstm_improved':
        model = SwinConvLSTM(
            image_size=(256, 256),
            in_channels=n_channel,
            out_channels=num_classes,
            feature_size=feature_size,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            hidden_dim=hidden_size,
            dropout=0.1,
            attn_version=attn_version,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('-m', type=str, default='swin_convlstm_improved', help='Model name')
    parser.add_argument('-mode', type=str, default='af', help='BA or Pred')
    parser.add_argument('-b', type=int, default=1, help='batch size')
    parser.add_argument('-nh', type=int, default=6, help='number-of-head')
    parser.add_argument('-ed', type=int, default=64, help='embedding dimension')
    parser.add_argument('-nc', type=int, default=8, help='n_channel')
    parser.add_argument('-ts', type=int, default=10, help='ts_length')
    parser.add_argument('-it', type=int, default=1, help='interval')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-run', type=int, default=1)
    parser.add_argument('-epoch', type=int, default=295, help='Model checkpoint epoch')
    parser.add_argument('-av', type=str, default='ar', help='attention version')
    parser.add_argument('-scheduler', type=str, default='cosine', help='scheduler')
    parser.add_argument('-test_csv', type=str, help='Path to test CSV file')
    parser.add_argument('-data_dir', type=str, help='Path to test data directory')
    parser.add_argument('-output_dir', type=str, help='Output directory for visualization')
    parser.add_argument('-model_path', type=str, help='Full path to model checkpoint file')
    parser.add_argument('-threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('-feature_size', type=int, default=96, help='Feature size for Swin model')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.m
    batch_size = args.b
    num_heads = args.nh
    hidden_size = args.ed
    ts_length = args.ts
    n_channel = args.nc
    interval = args.it
    mode = args.mode
    load_epoch = args.epoch

    # 测试数据配置
    test_csv = args.test_csv or os.path.expanduser('~/CalFireMonitoring/roi/us_fire_2021_out_new.csv')
    data_dir = args.data_dir or '/home/congwei/ts-satfire-tran/dataset_test_10i_10'
    output_dir = args.output_dir or '/home/congwei/swin_fire_release/evaluation_plot_swin_convlstm_improved'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {model_name}")
    print(f"Checkpoint epoch: {load_epoch}")
    print(f"Test data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # 创建模型
    model = load_model(model_name, n_channel, 2, hidden_size, num_heads, ts_length, device, feature_size=args.feature_size, attn_version=args.av)
    # model = nn.DataParallel(model)
    model.to(device)

    # 加载检查点
    if args.model_path:
        load_path = args.model_path
    else:
        load_path = (f"saved_models/model_{model_name}_run_{args.run}_seed_{args.seed}_"
                    f"mode_{mode}_nh_{num_heads}_hs_{hidden_size}_bs_{batch_size}_epoch_{load_epoch}_"
                    f"nc_{n_channel}_ts_{ts_length}_attn_{args.av}_scheduler_{args.scheduler}.pth")

    print(f"Loading checkpoint: {load_path}")
    checkpoint = torch.load(load_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint['model_state_dict']
        # 移除 DataParallel 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        loaded_epoch = checkpoint.get('epoch', load_epoch)
        print(f"Loaded epoch: {loaded_epoch}")
    else:
        model = checkpoint
        loaded_epoch = load_epoch
        if isinstance(model, nn.DataParallel):
            model = model.module
        print(f"Loaded model directly")

    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # 读取测试数据列表 - 从数据目录自动获取
    all_files = os.listdir(data_dir)
    img_files = sorted([f for f in all_files if f.endswith('_img_seqtoseql_{}i_{}.npy'.format(ts_length, interval))])
    
    ids = []
    label_sel_list = []
    for f in img_files:
        # 文件名格式: af_US_2021_AZ3345510938920210616_img_seqtoseql_10i_3.npy
        # 去掉后缀和前缀 af_
        base_name = f.replace('_img_seqtoseql_{}i_{}.npy'.format(ts_length, interval), '')
        if base_name.startswith('af_'):
            base_name = base_name[3:]  # 去掉 af_ 前缀
        ids.append(base_name)
        label_sel_list.append(2)
    
    print(f"Found {len(ids)} test samples")
    print(f"Sample IDs: {ids[:3]}")

    # 归一化参数
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

    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])

    threshold = args.threshold

    # 全局指标
    all_preds = []
    all_labels = []
    results_list = []

    for i, (id, label_sel) in enumerate(zip(ids, label_sel_list)):
        # 文件名格式: af_US_2021_XXX_img_seqtoseql_10i_3.npy
        test_image_path = os.path.join(data_dir, f'af_{id}_img_seqtoseql_{ts_length}i_{interval}.npy')
        test_label_path = os.path.join(data_dir, f'af_{id}_label_seqtoseql_{ts_length}i_{interval}.npy')

        if not os.path.exists(test_image_path):
            print(f"Skipping {id}: file not found")
            continue

        print(f"Processing {id} ({i+1}/{len(ids)})")

        test_dataset = FireDataset(
            image_path=test_image_path,
            label_path=test_label_path,
            ts_length=ts_length,
            transform=transform,
            n_channel=n_channel,
            label_sel=label_sel,
            is_train=False,
            crop_size=256,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        f1 = 0
        iou = 0
        length = 0

        for j, batch in enumerate(test_dataloader):
            test_data_batch = batch['data'].to(device)
            test_labels_batch = batch['labels']

            with torch.no_grad():
                outputs = model(test_data_batch)  # [B, 2, T, H, W]

            outputs_post = [post_trans(o) for o in decollate_batch(outputs)]
            outputs_np = np.stack([o[1].cpu().numpy() for o in outputs_post], axis=0)  # [B, T, H, W]

            length += test_data_batch.shape[0]

            for k in range(test_data_batch.shape[0]):
                output_stack = outputs_np[k, :, :, :]  # [T, H, W]
                label = test_labels_batch[k, 0].numpy()  # [T, H, W]

                # 处理无效区域
                valid_mask = (label != -1)
                output_valid = np.where(valid_mask, output_stack > threshold, 0)
                label_valid = np.where(valid_mask, label > 0, 0)

                # 计算每个时间步的指标
                for t in range(ts_length):
                    output_t = output_valid[t].flatten()
                    label_t = label_valid[t].flatten()

                    f1_ts = f1_score(label_t, output_t, zero_division=1.0)
                    iou_ts = jaccard_score(label_t, output_t, zero_division=1.0)
                    f1 += f1_ts
                    iou += iou_ts
                    length += 1

                # 可视化最后一个时间步
                output_last = output_stack[-1]  # [H, W]
                label_last = label[-1]  # [H, W]
                valid_mask_last = (label_last != -1)
                output_last_binary = np.where(valid_mask_last, output_last > threshold, 0)
                label_last_binary = np.where(valid_mask_last, label_last > 0, 0)

                f1_last = f1_score(label_last_binary.flatten(), output_last_binary.flatten(), zero_division=1.0)
                iou_last = jaccard_score(label_last_binary.flatten(), output_last_binary.flatten(), zero_division=1.0)

                # 收集全局指标
                all_preds.extend(output_last_binary.flatten())
                all_labels.extend(label_last_binary.flatten())

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # 输入图像
                img_input = test_data_batch[k, 3, -1, :, :].cpu().numpy()  # 最后一帧，通道3
                axes[0].imshow(normalization(img_input), cmap='gray')
                axes[0].set_title(f'Input (Band 4, T={ts_length-1})')
                axes[0].axis('off')

                # 真实标签
                axes[1].imshow(label_last, cmap='hot', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                # 预测结果
                axes[2].imshow(output_last, cmap='hot', vmin=0, vmax=1)
                axes[2].set_title(f'Prediction (th={threshold})')
                axes[2].axis('off')

                # 预测 vs 真实 (TP/FP/FN)
                # 创建一个 RGB 图像来显示三种颜色
                img_rgb = np.zeros((label_last.shape[0], label_last.shape[1], 3))
                
                # 背景使用灰度图
                img_display = normalization(img_input)
                img_rgb[:, :, 0] = img_display  # R
                img_rgb[:, :, 1] = img_display  # G
                img_rgb[:, :, 2] = img_display  # B
                
                # TP = 绿色, FP = 红色, FN = 蓝色
                # 只有当 valid_mask_last 为 True 时才显示
                tp_mask = (output_last_binary == 1) & (label_last_binary == 1) & valid_mask_last
                fp_mask = (output_last_binary == 1) & (label_last_binary == 0) & valid_mask_last
                fn_mask = (output_last_binary == 0) & (label_last_binary == 1) & valid_mask_last
                
                # 叠加颜色 (绿色 TP, 红色 FP, 蓝色 FN)
                img_rgb[tp_mask, 0] = 0  # R
                img_rgb[tp_mask, 1] = 1  # G  
                img_rgb[tp_mask, 2] = 0  # B
                
                img_rgb[fp_mask, 0] = 1  # R
                img_rgb[fp_mask, 1] = 0  # G
                img_rgb[fp_mask, 2] = 0  # B
                
                img_rgb[fn_mask, 0] = 0  # R
                img_rgb[fn_mask, 1] = 0  # G
                img_rgb[fn_mask, 2] = 1  # B
                
                axes[3].imshow(img_rgb)
                axes[3].set_title('TP(Green)/FP(Red)/FN(Blue)')
                axes[3].axis('off')

                plt.suptitle(f'ID: {id} | F1: {f1_last:.4f} | IoU: {iou_last:.4f}', fontsize=14)

                plot_path = os.path.join(output_dir, f'id_{id}_batch_{j}_sample_{k}_f1_{f1_last:.4f}_iou_{iou_last:.4f}.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"  Saved: {plot_path}")

        # 计算单个样本的指标
        num_timesteps = len(test_dataset) * ts_length
        if num_timesteps > 0:
            avg_f1 = f1 / num_timesteps
            avg_iou = iou / num_timesteps
            print(f'  ID {id}: F1={avg_f1:.4f}, IoU={avg_iou:.4f}')
            results_list.append({
                'fire_id': id,
                'f1_score': avg_f1,
                'iou': avg_iou
            })

    # 计算总体指标
    print(f'\nFinal Results')
    print(f'Total samples processed: {len(ids)}')
    
    if len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        iou = jaccard_score(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        
        print(f'Overall Metrics:')
        print(f'  F1 Score: {f1:.4f}')
        print(f'  IoU: {iou:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Overall Confusion Matrix - {model_name}')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'overall_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved confusion matrix to: {cm_path}')
    
    # 保存结果到 CSV
    if results_list:
        results_df = pd.DataFrame(results_list)
        csv_path = os.path.join(output_dir, 'test_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f'Results saved to: {csv_path}')
        print('\nPer-sample results:')
        print(results_df.to_string(index=False))

    print(f'\nVisualization saved to: {output_dir}')
