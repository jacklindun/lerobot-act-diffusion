#!/usr/bin/env python3
"""
ACT策略在NPU上的训练脚本
针对红色立方体抓取任务优化的配置
"""

import subprocess
import sys
from pathlib import Path

def create_act_training_command():
    """创建优化的ACT训练命令"""
    
    # 基础训练命令
    base_cmd = [
        "xvfb-run", "-a", "-s", "-screen 0 1600x900x30",
        "python", "lerobot/scripts/train.py"
    ]
    
    # ACT策略配置 - 针对视觉任务优化
    policy_args = [
        "--policy.type=act",
        "--policy.vision_backbone=resnet50",  # 使用更强的视觉编码器
        "--policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2",  # 使用预训练权重
        "--policy.chunk_size=50",  # 动作块大小
        "--policy.n_action_steps=25",  # 每次执行的动作步数
        "--policy.n_obs_steps=1",  # ACT目前只支持单步观察
        "--policy.dim_model=512",  # Transformer维度
        "--policy.n_heads=8",  # 注意力头数
        "--policy.n_encoder_layers=6",  # 增加编码器层数
        "--policy.n_decoder_layers=1",  # 解码器层数（ACT原始设计）
        "--policy.use_vae=true",  # 使用VAE
        "--policy.latent_dim=64",  # 增加潜在维度
        "--policy.kl_weight=10.0",  # KL散度权重
        "--policy.dropout=0.1",  # Dropout
        "--policy.optimizer_lr=1e-5",  # 学习率
        "--policy.optimizer_weight_decay=1e-4",  # 权重衰减
        "--policy.device=npu",  # 使用NPU
        "--policy.use_amp=false",  # 暂时禁用混合精度
    ]
    
    # 数据集配置
    dataset_args = [
        "--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human",
        "--dataset.image_transforms.enable=true",  # 启用图像增强
    ]
    
    # 环境配置
    env_args = [
        "--env.type=aloha",
        "--env.task=AlohaTransferCube-v0",
    ]
    
    # 训练配置
    training_args = [
        "--output_dir=outputs/train/act_aloha_transfer_npu",
        "--batch_size=8",  # 批次大小
        "--steps=200000",  # 训练步数
        "--eval_freq=0",  # 禁用训练中的评估
        "--save_freq=20000",  # 保存频率
        "--log_freq=500",  # 日志频率
        "--num_workers=4",  # 数据加载器工作进程
        "--seed=1000",  # 随机种子
        "--wandb.enable=false",  # 禁用wandb
    ]
    
    # 组合所有参数
    full_cmd = base_cmd + policy_args + dataset_args + env_args + training_args
    
    return full_cmd

def create_act_training_command_advanced():
    """创建高级优化的ACT训练命令（更激进的配置）"""
    
    base_cmd = [
        "xvfb-run", "-a", "-s", "-screen 0 1600x900x30",
        "python", "lerobot/scripts/train.py"
    ]
    
    # 高级ACT配置 - 最大化视觉处理能力
    policy_args = [
        "--policy.type=act",
        "--policy.vision_backbone=resnet50",  # 更强的视觉编码器
        "--policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2",
        "--policy.chunk_size=100",  # 更大的动作块
        "--policy.n_action_steps=50",  # 更多动作步数
        "--policy.dim_model=768",  # 更大的模型维度
        "--policy.n_heads=12",  # 更多注意力头
        "--policy.dim_feedforward=3072",  # 更大的前馈网络
        "--policy.n_encoder_layers=8",  # 更多编码器层
        "--policy.n_vae_encoder_layers=6",  # 更多VAE编码器层
        "--policy.latent_dim=128",  # 更大的潜在维度
        "--policy.use_vae=true",
        "--policy.kl_weight=5.0",  # 调整KL权重
        "--policy.dropout=0.15",  # 稍微增加dropout
        "--policy.optimizer_lr=5e-6",  # 更小的学习率
        "--policy.optimizer_lr_backbone=1e-6",  # 视觉编码器更小的学习率
        "--policy.optimizer_weight_decay=1e-4",
        "--policy.device=npu",
        "--policy.use_amp=false",
    ]
    
    # 其他配置保持相同
    dataset_args = [
        "--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human",
        "--dataset.image_transforms.enable=true",
    ]
    
    env_args = [
        "--env.type=aloha",
        "--env.task=AlohaTransferCube-v0",
    ]
    
    training_args = [
        "--output_dir=outputs/train/act_aloha_transfer_npu_advanced",
        "--batch_size=6",  # 稍小的批次大小（因为模型更大）
        "--steps=300000",  # 更多训练步数
        "--eval_freq=0",
        "--save_freq=20000",
        "--log_freq=500",
        "--num_workers=4",
        "--seed=1000",
        "--wandb.enable=false",
    ]
    
    full_cmd = base_cmd + policy_args + dataset_args + env_args + training_args
    
    return full_cmd

def print_training_info():
    """打印训练信息"""
    print("=" * 60)
    print("ACT策略训练配置")
    print("=" * 60)
    
    print("🎯 ACT策略的优势:")
    print("  • Transformer架构，更强的视觉处理能力")
    print("  • 动作块预测，适合连续控制任务")
    print("  • VAE正则化，提高泛化能力")
    print("  • 预训练视觉编码器，更好的特征提取")
    
    print("\n📊 配置对比:")
    print("  标准配置:")
    print("    - ResNet50 + ImageNet预训练")
    print("    - 512维Transformer")
    print("    - 50步动作块，25步执行")
    print("    - 200k训练步数")
    
    print("  高级配置:")
    print("    - ResNet50 + ImageNet预训练")
    print("    - 768维Transformer，更多层数")
    print("    - 100步动作块，50步执行")
    print("    - 300k训练步数")
    
    print("\n⚡ NPU优化:")
    print("  • 禁用混合精度（避免兼容性问题）")
    print("  • 优化的批次大小")
    print("  • 禁用训练中评估（提高训练效率）")

def main():
    """主函数"""
    print_training_info()
    
    print("\n" + "=" * 60)
    print("选择训练配置")
    print("=" * 60)
    
    print("1. 标准配置 (推荐开始)")
    print("2. 高级配置 (更强但需要更多资源)")
    print("3. 显示命令但不执行")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        cmd = create_act_training_command()
        config_name = "标准配置"
    elif choice == "2":
        cmd = create_act_training_command_advanced()
        config_name = "高级配置"
    elif choice == "3":
        print("\n标准配置命令:")
        cmd1 = create_act_training_command()
        print(" \\\n  ".join(cmd1))
        
        print("\n高级配置命令:")
        cmd2 = create_act_training_command_advanced()
        print(" \\\n  ".join(cmd2))
        return
    else:
        print("无效选择")
        return
    
    print(f"\n🚀 开始执行{config_name}训练...")
    print("命令:")
    print(" \\\n  ".join(cmd))
    
    confirm = input("\n确认执行? (y/N): ").strip().lower()
    if confirm == 'y':
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"训练失败: {e}")
            return 1
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            return 1
        
        print("\n🎉 训练完成!")
        print("推理命令:")
        output_dir = "outputs/train/act_aloha_transfer_npu" if choice == "1" else "outputs/train/act_aloha_transfer_npu_advanced"
        print(f"xvfb-run -a -s \"-screen 0 1600x900x30\" python -X faulthandler lerobot/scripts/eval.py --policy.path={output_dir}/checkpoints/020000/pretrained_model --output_dir=outputs/eval/act_aloha_transfer/020000 --env.type=aloha --env.task=AlohaTransferCube-v0")
    else:
        print("训练取消")

if __name__ == "__main__":
    exit(main())
