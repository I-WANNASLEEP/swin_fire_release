import torch
import os
import warnings
warnings.filterwarnings('ignore')

from spatial_models.swin_convlstm import SwinConvLSTM
try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def visualize_model_architecture(pth_path, output_dir='model_visualization', 
                                 n_channel=8, ts_length=5, image_size=(256, 256), num_classes=2):
    """
    可视化模型架构，生成多种格式的可视化结果
    """
    print(f"- Loading model from: {pth_path}")
    
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型
    model = SwinConvLSTM(
        image_size=image_size,
        in_channels=n_channel,
        out_channels=num_classes,
        feature_size=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        hidden_dim=64,
        dropout=0.1,
    )
    
    # 加载权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"- Model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   F1 Score: {checkpoint.get('f1_score', 'N/A')}")
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, n_channel, ts_length, image_size[0], image_size[1])
    
    print(f"\n- Input shape: {dummy_input.shape}")
    print(f"   Batch=1, Channel={n_channel}, Time={ts_length}, H={image_size[0]}, W={image_size[1]}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n- Model Statistics:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 打印模型结构
    print(f"\n- Model Architecture:")
    print(model)
    
    # 保存模型结构到文本文件
    with open(f'{output_dir}/model_architecture.txt', 'w') as f:
        f.write("SwinConvLSTM Model Architecture\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input shape: (1, {n_channel}, {ts_length}, {image_size[0]}, {image_size[1]})\n")
        f.write(f"Output shape: (1, {num_classes}, {ts_length}, {image_size[0]}, {image_size[1]})\n\n")
        f.write(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n\n")
        f.write("="*60 + "\n\n")
        f.write(str(model))
    
    print(f"\n- Model architecture saved to: {output_dir}/model_architecture.txt")
    
    # 尝试使用torchviz生成计算图
    if TORCHVIZ_AVAILABLE and GRAPHVIZ_AVAILABLE:
        print(f"\n- Generating computational graph (torchviz)...")
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            dot = make_dot(output, params=dict(list(model.named_parameters())[:10]), 
                          show_attrs=False, show_saved=False)
            graph_path = f'{output_dir}/model_graph'
            dot.render(graph_path, format='png', cleanup=True)
            print(f"- Computational graph saved to: {graph_path}.png")
            print(f"- You can visualize it better by: ")
            print(f"   1. Install graphviz: sudo apt-get install graphviz")
            print(f"   2. Open the PNG file in any image viewer")
        except Exception as e:
            print(f"- torchviz failed: {e}")
    else:
        if not TORCHVIZ_AVAILABLE:
            print(f"\n- torchviz not installed. Install it with:")
            print(f"   pip install torchviz")
        if not GRAPHVIZ_AVAILABLE:
            print(f"- graphviz not installed. Install it with:")
            print(f"   pip install graphviz")
            print(f"   sudo apt-get install graphviz")
    
    # 尝试使用TensorBoard
    if TENSORBOARD_AVAILABLE:
        print(f"\n- Creating TensorBoard summary...")
        try:
            writer = SummaryWriter(f'{output_dir}/tensorboard')
            writer.add_graph(model, dummy_input)
            writer.close()
            print(f"- TensorBoard log created!")
            print(f"- To visualize with TensorBoard:")
            print(f"   cd {output_dir}")
            print(f"   tensorboard --logdir tensorboard")
            print(f"   Then open in browser: http://localhost:6006")
        except Exception as e:
            print(f"- TensorBoard failed: {e}")
    else:
        print(f"\n- For interactive visualization, install tensorboard:")
        print(f"   pip install tensorboard")
    
    # 生成详细的层级结构
    print(f"\n- Generating detailed layer information...")
    with open(f'{output_dir}/model_layers.txt', 'w') as f:
        f.write("Detailed Model Layers\n")
        f.write("="*80 + "\n\n")
        
        for name, param in model.named_parameters():
            f.write(f"{name:60s} | Shape: {str(param.shape):25s} | Params: {param.numel():,}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Total: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    print(f"- Layer details saved to: {output_dir}/model_layers.txt")
    
    print(f"\n" + "="*60)
    print("- Visualization completed!")
    print("="*60)
    print(f"\n- Output directory: {output_dir}/")
    print(f"   - model_architecture.txt  - Complete model architecture")
    print(f"   - model_layers.txt        - Detailed layer information")
    if TORCHVIZ_AVAILABLE:
        print(f"   - model_graph.png        - Computational graph image")
    if TENSORBOARD_AVAILABLE:
        print(f"   - tensorboard/            - TensorBoard logs (interactive)")
    
    print(f"\n- Recommended visualization tools:")
    print(f"   1. Read model_architecture.txt for structure overview")
    print(f"   2. Read model_layers.txt for detailed layer info")
    if TORCHVIZ_AVAILABLE and GRAPHVIZ_AVAILABLE:
        print(f"   3. View model_graph.png for computational graph")
    if TENSORBOARD_AVAILABLE:
        print(f"   4. Use TensorBoard for interactive exploration:")
        print(f"      cd {output_dir} && tensorboard --logdir tensorboard")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SwinConvLSTM model architecture')
    parser.add_argument('--pth_path', type=str, required=True, 
                       help='Path to the .pth model file')
    parser.add_argument('--output_dir', type=str, default='model_visualization',
                       help='Output directory for visualizations (default: model_visualization)')
    parser.add_argument('--n_channel', type=int, default=8,
                       help='Number of input channels (default: 8)')
    parser.add_argument('--ts_length', type=int, default=5,
                       help='Time sequence length (default: 5)')
    parser.add_argument('--height', type=int, default=256,
                       help='Input image height (default: 256)')
    parser.add_argument('--width', type=int, default=256,
                       help='Input image width (default: 256)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of output classes (default: 2)')
    
    args = parser.parse_args()
    
    visualize_model_architecture(
        pth_path=args.pth_path,
        output_dir=args.output_dir,
        n_channel=args.n_channel,
        ts_length=args.ts_length,
        image_size=(args.height, args.width),
        num_classes=args.num_classes
    )


if __name__ == '__main__':
    main()
