#!/usr/bin/env python3
"""
检查模型文件结构，找出真正的问题
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
import os

def inspect_model_files():
    """检查模型文件结构"""
    print("=" * 60)
    print("检查模型文件结构")
    print("=" * 60)
    
    # 检查100k和300k模型
    models = [
        "outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model",
        "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
    ]
    
    for model_path in models:
        print(f"\n检查模型: {model_path}")
        
        if not Path(model_path).exists():
            print(f"❌ 路径不存在")
            continue
        
        # 列出所有文件
        print("文件列表:")
        for file_path in Path(model_path).iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name}: {size_mb:.2f} MB")
        
        # 检查safetensors文件
        safetensors_file = Path(model_path) / "model.safetensors"
        if safetensors_file.exists():
            try:
                state_dict = load_file(safetensors_file)
                print(f"\nSafetensors内容:")
                print(f"  键的数量: {len(state_dict)}")
                
                total_params = 0
                for key, tensor in state_dict.items():
                    params = tensor.numel()
                    total_params += params
                    print(f"    {key}: {tensor.shape} ({params:,} 参数)")
                
                print(f"  总参数数量: {total_params:,}")
                
                if total_params < 1000000:
                    print("  ⚠️ 参数数量异常少！")
                
            except Exception as e:
                print(f"  ❌ 无法加载safetensors: {e}")
        
        # 检查是否有其他权重文件
        pytorch_file = Path(model_path) / "pytorch_model.bin"
        if pytorch_file.exists():
            print(f"\n发现PyTorch权重文件: {pytorch_file}")
            try:
                state_dict = torch.load(pytorch_file, map_location='cpu')
                print(f"  PyTorch文件键数量: {len(state_dict)}")
                
                total_params = 0
                for key, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        total_params += tensor.numel()
                
                print(f"  PyTorch文件总参数: {total_params:,}")
                
            except Exception as e:
                print(f"  ❌ 无法加载PyTorch文件: {e}")

def check_checkpoint_structure():
    """检查checkpoint目录结构"""
    print("\n" + "=" * 60)
    print("检查Checkpoint目录结构")
    print("=" * 60)
    
    checkpoint_dir = Path("outputs/train/diffusion_aloha_transfer_npu/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint目录不存在")
        return
    
    # 检查每个checkpoint
    for cp_dir in sorted(checkpoint_dir.iterdir()):
        if cp_dir.is_dir() and cp_dir.name.isdigit():
            print(f"\nCheckpoint {cp_dir.name}:")
            
            # 检查是否有直接的权重文件
            for file_path in cp_dir.iterdir():
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  {file_path.name}: {size_mb:.2f} MB")
            
            # 检查pretrained_model目录
            pretrained_dir = cp_dir / "pretrained_model"
            if pretrained_dir.exists():
                print(f"  pretrained_model/:")
                for file_path in pretrained_dir.iterdir():
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        print(f"    {file_path.name}: {size_mb:.2f} MB")

def find_real_model_weights():
    """寻找真正的模型权重文件"""
    print("\n" + "=" * 60)
    print("寻找真正的模型权重文件")
    print("=" * 60)
    
    # 在整个训练目录中搜索大文件
    train_dir = Path("outputs/train/diffusion_aloha_transfer_npu")
    
    print("搜索大于100MB的文件:")
    large_files = []
    
    for file_path in train_dir.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                large_files.append((file_path, size_mb))
    
    # 按大小排序
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, size_mb in large_files:
        print(f"  {file_path}: {size_mb:.2f} MB")
        
        # 如果是.pt或.pth文件，尝试加载
        if file_path.suffix in ['.pt', '.pth']:
            try:
                state_dict = torch.load(file_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    total_params = 0
                    for key, tensor in state_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            total_params += tensor.numel()
                    print(f"    -> PyTorch文件，包含 {total_params:,} 参数")
            except:
                pass
        
        # 如果是safetensors文件，尝试加载
        elif file_path.suffix == '.safetensors':
            try:
                state_dict = load_file(file_path)
                total_params = sum(tensor.numel() for tensor in state_dict.values())
                print(f"    -> Safetensors文件，包含 {total_params:,} 参数")
            except:
                pass

def main():
    """主函数"""
    print("LeRobot 模型文件结构检查")
    print("=" * 60)
    
    inspect_model_files()
    check_checkpoint_structure()
    find_real_model_weights()
    
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)
    print("如果发现:")
    print("1. safetensors文件很小但有大的.pt/.pth文件 -> 需要转换格式")
    print("2. 所有文件都很小 -> 训练可能没有正确保存权重")
    print("3. 找到大的权重文件 -> 需要正确加载这些文件")

if __name__ == "__main__":
    main()
