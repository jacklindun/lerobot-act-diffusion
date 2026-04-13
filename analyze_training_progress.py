#!/usr/bin/env python3
"""
分析训练进度和推理效果，提供改进建议
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_config(model_path: str):
    """分析训练配置"""
    print("=" * 60)
    print("分析训练配置")
    print("=" * 60)
    
    config_file = Path(model_path) / "train_config.json"
    
    if not config_file.exists():
        print(f"❌ 训练配置文件不存在: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("训练配置分析:")
    print(f"  总训练步数: {config.get('steps', 'UNKNOWN')}")
    print(f"  批次大小: {config.get('batch_size', 'UNKNOWN')}")
    print(f"  学习率: {config.get('policy', {}).get('optimizer_lr', 'UNKNOWN')}")
    print(f"  数据集: {config.get('dataset', {}).get('repo_id', 'UNKNOWN')}")
    
    # 检查diffusion特定参数
    policy_config = config.get('policy', {})
    print(f"\nDiffusion策略配置:")
    print(f"  观察步数: {policy_config.get('n_obs_steps', 'UNKNOWN')}")
    print(f"  动作步数: {policy_config.get('n_action_steps', 'UNKNOWN')}")
    print(f"  预测时间范围: {policy_config.get('horizon', 'UNKNOWN')}")
    print(f"  训练时间步: {policy_config.get('num_train_timesteps', 'UNKNOWN')}")
    print(f"  推理时间步: {policy_config.get('num_inference_steps', 'UNKNOWN')}")
    print(f"  视觉骨干网络: {policy_config.get('vision_backbone', 'UNKNOWN')}")
    
    return config

def check_available_checkpoints():
    """检查可用的checkpoint"""
    print("\n" + "=" * 60)
    print("检查可用的Checkpoints")
    print("=" * 60)
    
    checkpoint_dir = Path("outputs/train/diffusion_aloha_transfer_npu/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            step = int(item.name)
            pretrained_dir = item / "pretrained_model"
            if pretrained_dir.exists():
                checkpoints.append(step)
    
    checkpoints.sort()
    print(f"可用的checkpoints: {checkpoints}")
    
    if checkpoints:
        print(f"最新checkpoint: {max(checkpoints)} 步")
        print(f"最早checkpoint: {min(checkpoints)} 步")
        print(f"总共保存了 {len(checkpoints)} 个checkpoints")
    
    return checkpoints

def provide_improvement_suggestions(config, checkpoints):
    """提供改进建议"""
    print("\n" + "=" * 60)
    print("改进建议")
    print("=" * 60)
    
    current_steps = max(checkpoints) if checkpoints else 0
    total_steps = config.get('steps', 0) if config else 0
    
    print("🎯 **针对视觉感知和精确定位问题的建议:**\n")
    
    # 1. 训练步数建议
    print("1. **训练步数优化:**")
    if current_steps < 200000:
        print(f"   ⚠️  当前训练步数 ({current_steps}) 可能不足")
        print("   💡 建议继续训练到至少 200,000-500,000 步")
        print("   💡 Diffusion策略通常需要更多训练才能学好精确的视觉-动作映射")
    else:
        print(f"   ✅ 训练步数 ({current_steps}) 较为充足")
    
    # 2. 数据质量建议
    print("\n2. **数据质量检查:**")
    print("   💡 检查训练数据中红色立方体的位置变化是否足够多样")
    print("   💡 确保数据集包含不同光照条件下的场景")
    print("   💡 验证人工演示的精确性，特别是夹取和交接动作")
    
    # 3. 模型配置建议
    print("\n3. **模型配置优化:**")
    if config:
        policy_config = config.get('policy', {})
        
        # 检查观察步数
        n_obs_steps = policy_config.get('n_obs_steps', 2)
        if n_obs_steps < 2:
            print(f"   ⚠️  观察步数 ({n_obs_steps}) 可能过少")
            print("   💡 建议增加到 2-4 步以提供更多时序信息")
        
        # 检查动作步数
        n_action_steps = policy_config.get('n_action_steps', 8)
        if n_action_steps < 8:
            print(f"   ⚠️  动作步数 ({n_action_steps}) 可能过少")
            print("   💡 建议设置为 8-16 步以提高动作平滑性")
        
        # 检查推理步数
        num_inference_steps = policy_config.get('num_inference_steps')
        if num_inference_steps is None or num_inference_steps < 10:
            print("   ⚠️  推理时间步数可能过少")
            print("   💡 建议在推理时设置 num_inference_steps=10-20")
    
    # 4. 推理优化建议
    print("\n4. **推理优化:**")
    print("   💡 尝试不同的推理步数 (10, 20, 50)")
    print("   💡 调整温度参数以控制动作的随机性")
    print("   💡 使用多次推理取平均来提高稳定性")
    
    # 5. 环境配置建议
    print("\n5. **环境配置检查:**")
    print("   💡 确保推理环境与训练环境的相机配置一致")
    print("   💡 检查图像预处理参数是否与训练时相同")
    print("   💡 验证动作空间的归一化范围")

def generate_continue_training_command(checkpoints):
    """生成继续训练的命令"""
    print("\n" + "=" * 60)
    print("继续训练命令")
    print("=" * 60)
    
    if not checkpoints:
        print("❌ 没有可用的checkpoint，需要从头开始训练")
        return
    
    latest_checkpoint = max(checkpoints)
    
    print("🚀 **继续训练命令:**")
    print(f"""
xvfb-run -s -a "-screen 0 1600x900x30" python lerobot/scripts/train.py \\
  --output_dir=outputs/train/diffusion_aloha_transfer_npu \\
  --policy.type=diffusion \\
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \\
  --env.type=aloha \\
  --env.task=AlohaTransferCube-v0 \\
  --wandb.enable=false \\
  --resume=true \\
  --steps=300000
""")
    
    print(f"📝 **说明:**")
    print(f"   - 将从第 {latest_checkpoint} 步继续训练")
    print(f"   - 目标训练到 300,000 步")
    print(f"   - 使用 --resume=true 参数继续训练")

def generate_improved_inference_command():
    """生成改进的推理命令"""
    print("\n" + "=" * 60)
    print("改进的推理命令")
    print("=" * 60)
    
    print("🎯 **优化推理参数的命令:**")
    print("""
# 方法1: 增加推理步数
xvfb-run -a -s "-screen 0 1600x900x30" python -X faulthandler lerobot/scripts/eval.py \\
  --policy.path=outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model \\
  --output_dir=outputs/eval/diffusion_aloha_transfer/100000_improved \\
  --env.type=aloha \\
  --env.task=AlohaTransferCube-v0 \\
  --policy.num_inference_steps=20

# 方法2: 测试更新的checkpoint (如果继续训练后)
xvfb-run -a -s "-screen 0 1600x900x30" python -X faulthandler lerobot/scripts/eval.py \\
  --policy.path=outputs/train/diffusion_aloha_transfer_npu/checkpoints/200000/pretrained_model \\
  --output_dir=outputs/eval/diffusion_aloha_transfer/200000 \\
  --env.type=aloha \\
  --env.task=AlohaTransferCube-v0
""")

def main():
    """主函数"""
    print("LeRobot 训练进度分析和改进建议")
    print("=" * 60)
    
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model"
    
    # 分析训练配置
    config = analyze_training_config(model_path)
    
    # 检查可用checkpoints
    checkpoints = check_available_checkpoints()
    
    # 提供改进建议
    provide_improvement_suggestions(config, checkpoints)
    
    # 生成继续训练命令
    generate_continue_training_command(checkpoints)
    
    # 生成改进的推理命令
    generate_improved_inference_command()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("🎯 主要问题: 视觉感知精度不足，需要更多训练")
    print("💡 建议优先级:")
    print("   1. 继续训练到 200,000+ 步")
    print("   2. 优化推理参数 (增加推理步数)")
    print("   3. 检查数据质量和多样性")
    print("   4. 调整模型配置参数")

if __name__ == "__main__":
    main()
