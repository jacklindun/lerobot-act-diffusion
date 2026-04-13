#!/usr/bin/env python3
"""
检查模型文件的完整性
"""

import os
import json
from pathlib import Path
from safetensors.torch import load_file

def check_model_files(model_path: str):
    """检查模型文件"""
    print("=" * 60)
    print("检查模型文件完整性")
    print("=" * 60)
    
    model_dir = Path(model_path)
    
    # 检查目录是否存在
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    print(f"✓ 模型目录: {model_dir}")
    
    # 列出所有文件
    print("\n文件列表:")
    for file_path in model_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_path.name}: {size_mb:.2f} MB")
    
    # 检查配置文件
    config_file = model_dir / "config.json"
    if config_file.exists():
        print(f"\n✓ 配置文件存在: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"  策略类型: {config.get('type', 'MISSING')}")
        print(f"  设备: {config.get('device', 'MISSING')}")
        print(f"  使用AMP: {config.get('use_amp', 'MISSING')}")
    else:
        print(f"❌ 配置文件不存在: {config_file}")
    
    # 检查模型权重文件
    model_file = model_dir / "model.safetensors"
    if model_file.exists():
        print(f"\n✓ 模型权重文件存在: {model_file}")
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")
        
        # 加载并检查权重
        try:
            state_dict = load_file(model_file)
            print(f"  参数数量: {len(state_dict)}")
            
            total_params = 0
            for key, tensor in state_dict.items():
                total_params += tensor.numel()
                print(f"    {key}: {tensor.shape} ({tensor.numel():,} params)")
            
            print(f"  总参数数量: {total_params:,}")
            
            if total_params < 1000000:  # 少于100万参数
                print("  ⚠️ 警告: 参数数量异常少，可能模型不完整")
            
        except Exception as e:
            print(f"  ❌ 无法加载权重文件: {e}")
    else:
        print(f"❌ 模型权重文件不存在: {model_file}")
    
    # 检查其他可能的权重文件
    pytorch_file = model_dir / "pytorch_model.bin"
    if pytorch_file.exists():
        print(f"\n✓ PyTorch权重文件存在: {pytorch_file}")
        size_mb = pytorch_file.stat().st_size / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")

def check_checkpoint_directory():
    """检查整个checkpoint目录"""
    print("\n" + "=" * 60)
    print("检查Checkpoint目录结构")
    print("=" * 60)
    
    checkpoint_dir = Path("outputs/train/diffusion_aloha_transfer_npu/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return
    
    print(f"✓ Checkpoint目录: {checkpoint_dir}")
    
    # 列出所有checkpoint
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            checkpoints.append(int(item.name))
    
    checkpoints.sort()
    print(f"\n可用的checkpoints: {checkpoints}")
    
    # 检查每个checkpoint
    for cp in checkpoints:
        cp_dir = checkpoint_dir / str(cp).zfill(6)
        pretrained_dir = cp_dir / "pretrained_model"
        
        print(f"\nCheckpoint {cp}:")
        if pretrained_dir.exists():
            print(f"  ✓ pretrained_model目录存在")
            
            # 检查文件
            for file_path in pretrained_dir.iterdir():
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"    {file_path.name}: {size_mb:.2f} MB")
        else:
            print(f"  ❌ pretrained_model目录不存在")

def main():
    print("LeRobot 模型文件完整性检查")
    print("=" * 60)
    
    # 检查指定的模型
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model"
    check_model_files(model_path)
    
    # 检查整个checkpoint目录
    check_checkpoint_directory()

if __name__ == "__main__":
    main()
