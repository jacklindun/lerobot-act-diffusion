#!/usr/bin/env python3
"""
诊断NPU推理效果差的问题
分析可能的原因：数据归一化、设备一致性、模型权重等
"""

import logging
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add lerobot to path - 当前已经在lerobot-main目录中
sys.path.insert(0, os.path.dirname(__file__))

# Import torch_npu first
try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
except ImportError as e:
    print(f"torch_npu import failed: {e}")
    torch_npu = None

from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging
)
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.configs import AlohaEnv

def check_model_weights_consistency(model_path: str):
    """检查模型权重是否正确加载"""
    print("=" * 60)
    print("检查模型权重一致性")
    print("=" * 60)
    
    try:
        # 加载配置
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "cpu"  # 先在CPU上加载
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        
        # 在CPU上加载模型
        cpu_policy = make_policy(cfg=config, env_cfg=env_cfg)
        cpu_policy.eval()
        
        print(f"✓ 模型在CPU上加载成功")
        
        # 检查权重统计信息
        total_params = 0
        zero_params = 0
        inf_params = 0
        nan_params = 0
        
        for name, param in cpu_policy.named_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            inf_params += torch.isinf(param).sum().item()
            nan_params += torch.isnan(param).sum().item()
        
        print(f"✓ 总参数数量: {total_params:,}")
        print(f"✓ 零参数数量: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
        print(f"✓ 无穷参数数量: {inf_params:,}")
        print(f"✓ NaN参数数量: {nan_params:,}")
        
        if inf_params > 0 or nan_params > 0:
            print("❌ 发现异常参数值！")
            return False
        
        # 检查归一化统计信息
        print("\n检查归一化统计信息:")
        for name, module in cpu_policy.named_modules():
            if hasattr(module, 'normalize_inputs'):
                print(f"  输入归一化模块: {name}")
                for key, buffer in module.normalize_inputs.named_buffers():
                    if 'buffer_' in key:
                        buffer_name = key.replace('buffer_', '').replace('_', '.')
                        print(f"    {buffer_name}: {buffer}")
            
            if hasattr(module, 'normalize_targets'):
                print(f"  目标归一化模块: {name}")
                for key, buffer in module.normalize_targets.named_buffers():
                    if 'buffer_' in key:
                        buffer_name = key.replace('buffer_', '').replace('_', '.')
                        print(f"    {buffer_name}: {buffer}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型权重检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_device_transfer_consistency(model_path: str):
    """检查设备转移的一致性"""
    print("\n" + "=" * 60)
    print("检查设备转移一致性")
    print("=" * 60)
    
    try:
        # 加载配置
        config = PreTrainedConfig.from_pretrained(model_path)
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        
        # 在CPU上加载
        config.device = "cpu"
        cpu_policy = make_policy(cfg=config, env_cfg=env_cfg)
        cpu_policy.eval()
        
        # 创建测试数据
        batch_size = 1
        n_obs_steps = getattr(config, 'n_obs_steps', 2)
        
        test_batch = {}
        if "observation.state" in cpu_policy.config.input_features:
            state_dim = cpu_policy.config.input_features["observation.state"].shape[0]
            test_batch["observation.state"] = torch.randn(batch_size, n_obs_steps, state_dim)
        
        image_features = [key for key in cpu_policy.config.input_features.keys() if "image" in key]
        for img_key in image_features:
            img_shape = cpu_policy.config.input_features[img_key].shape
            test_batch[img_key] = torch.randn(batch_size, n_obs_steps, *img_shape)
        
        # CPU推理
        with torch.no_grad():
            cpu_action = cpu_policy.select_action(test_batch)
        
        print(f"✓ CPU推理成功: {cpu_action.shape}")
        print(f"✓ CPU动作统计: mean={cpu_action.mean():.4f}, std={cpu_action.std():.4f}")
        
        # 转移到NPU
        if torch_npu and torch_npu.npu.is_available():
            config.device = "npu"
            npu_policy = make_policy(cfg=config, env_cfg=env_cfg)
            npu_policy.eval()
            
            # 转移测试数据到NPU
            npu_batch = {k: v.to("npu") for k, v in test_batch.items()}
            
            # NPU推理
            with torch.no_grad():
                npu_action = npu_policy.select_action(npu_batch)
            
            print(f"✓ NPU推理成功: {npu_action.shape}")
            print(f"✓ NPU动作统计: mean={npu_action.mean():.4f}, std={npu_action.std():.4f}")
            
            # 比较结果
            cpu_action_np = cpu_action.cpu().numpy()
            npu_action_np = npu_action.cpu().numpy()
            
            diff = np.abs(cpu_action_np - npu_action_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"✓ 最大差异: {max_diff:.6f}")
            print(f"✓ 平均差异: {mean_diff:.6f}")
            
            if max_diff > 1e-3:
                print("⚠ CPU和NPU推理结果差异较大！")
                return False
            else:
                print("✓ CPU和NPU推理结果一致")
                return True
        else:
            print("⚠ NPU不可用，跳过设备一致性检查")
            return True
            
    except Exception as e:
        print(f"❌ 设备转移一致性检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_action_range_and_scaling(model_path: str):
    """检查动作范围和缩放"""
    print("\n" + "=" * 60)
    print("检查动作范围和缩放")
    print("=" * 60)
    
    try:
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "npu" if torch_npu and torch_npu.npu.is_available() else "cpu"
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        # 创建多个测试样本
        batch_size = 10
        n_obs_steps = getattr(config, 'n_obs_steps', 2)
        
        test_batch = {}
        if "observation.state" in policy.config.input_features:
            state_dim = policy.config.input_features["observation.state"].shape[0]
            test_batch["observation.state"] = torch.randn(batch_size, n_obs_steps, state_dim, device=config.device)
        
        image_features = [key for key in policy.config.input_features.keys() if "image" in key]
        for img_key in image_features:
            img_shape = policy.config.input_features[img_key].shape
            test_batch[img_key] = torch.randn(batch_size, n_obs_steps, *img_shape, device=config.device)
        
        # 收集多个动作样本
        actions = []
        for i in range(batch_size):
            single_batch = {k: v[i:i+1] for k, v in test_batch.items()}
            with torch.no_grad():
                action = policy.select_action(single_batch)
            actions.append(action.cpu().numpy())
        
        actions = np.concatenate(actions, axis=0)
        
        print(f"✓ 动作形状: {actions.shape}")
        print(f"✓ 动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"✓ 动作均值: {actions.mean():.4f}")
        print(f"✓ 动作标准差: {actions.std():.4f}")
        
        # 检查是否有异常值
        if np.any(np.isnan(actions)):
            print("❌ 发现NaN动作值！")
            return False
        
        if np.any(np.isinf(actions)):
            print("❌ 发现无穷动作值！")
            return False
        
        # 检查动作是否过于极端
        if np.abs(actions).max() > 100:
            print("⚠ 动作值可能过大")
            return False
        
        # 检查动作是否有变化
        if actions.std() < 1e-6:
            print("⚠ 动作值几乎没有变化，可能模型输出固定值")
            return False
        
        print("✓ 动作范围和缩放正常")
        return True
        
    except Exception as e:
        print(f"❌ 动作范围检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    init_logging()
    
    print("LeRobot NPU推理问题诊断")
    print("=" * 60)
    
    # 检查模型路径
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请确认模型路径正确")
        return 1
    
    print(f"✓ 模型路径存在: {model_path}")
    
    # 运行诊断测试
    tests = [
        ("模型权重一致性", lambda: check_model_weights_consistency(model_path)),
        ("设备转移一致性", lambda: check_device_transfer_consistency(model_path)),
        ("动作范围和缩放", lambda: check_action_range_and_scaling(model_path)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体: {passed}/{len(results)} 测试通过")
    
    if passed < len(results):
        print("\n可能的问题和解决方案:")
        print("1. 模型权重问题: 检查训练是否正常完成，权重是否正确保存")
        print("2. 设备一致性问题: 确保NPU和CPU计算结果一致")
        print("3. 数据归一化问题: 检查训练和推理时的数据预处理是否一致")
        print("4. 动作缩放问题: 检查动作空间的归一化和反归一化")
        print("5. 环境配置问题: 确保推理环境与训练环境配置一致")
        return 1
    else:
        print("\n🎉 所有诊断测试通过！问题可能在其他方面:")
        print("1. 检查训练数据质量和多样性")
        print("2. 检查训练步数是否足够")
        print("3. 检查超参数设置")
        print("4. 检查环境任务的难度设置")
        return 0

if __name__ == "__main__":
    exit(main())
