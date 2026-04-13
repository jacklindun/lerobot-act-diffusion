#!/usr/bin/env python3
"""
最终修复归一化统计信息的脚本
直接从safetensors文件中提取并应用归一化统计信息
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def extract_and_apply_normalization_stats(model_path: str):
    """提取并应用归一化统计信息"""
    print("=" * 60)
    print("最终修复归一化统计信息")
    print("=" * 60)
    
    try:
        # 1. 加载模型权重
        model_file = Path(model_path) / "model.safetensors"
        state_dict = load_file(model_file)
        
        print(f"✓ 模型权重加载成功")
        
        # 2. 检查现有的归一化统计信息
        norm_keys = [key for key in state_dict.keys() if 'normalize' in key or 'buffer_' in key]
        print(f"发现 {len(norm_keys)} 个归一化相关的键:")
        
        for key in norm_keys:
            value = state_dict[key]
            print(f"  {key}: {value}")
            
            # 检查是否有无穷值
            if torch.isinf(value).any():
                print(f"    ⚠️ 发现无穷值！")
        
        # 3. 加载数据集统计信息
        print("\n正在加载数据集统计信息...")
        dataset_path = "/root/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"
        dataset = LeRobotDataset(dataset_path)
        
        print("数据集统计信息:")
        for key, stats in dataset.meta.stats.items():
            print(f"  {key}:")
            for stat_name, stat_value in stats.items():
                print(f"    {stat_name}: {stat_value}")
        
        # 4. 手动设置正确的归一化统计信息
        print("\n修复归一化统计信息...")
        
        # 图像归一化统计信息
        if 'observation.images.top' in dataset.meta.stats:
            img_stats = dataset.meta.stats['observation.images.top']
            
            # 设置图像均值和标准差
            if 'normalize_inputs.buffer_observation_images_top.mean' in state_dict:
                mean_tensor = torch.tensor(img_stats['mean'], dtype=torch.float32).view(3, 1, 1)
                state_dict['normalize_inputs.buffer_observation_images_top.mean'] = mean_tensor
                print(f"  ✓ 设置图像均值: {mean_tensor.flatten()}")
            
            if 'normalize_inputs.buffer_observation_images_top.std' in state_dict:
                std_tensor = torch.tensor(img_stats['std'], dtype=torch.float32).view(3, 1, 1)
                state_dict['normalize_inputs.buffer_observation_images_top.std'] = std_tensor
                print(f"  ✓ 设置图像标准差: {std_tensor.flatten()}")
        
        # 状态归一化统计信息
        if 'observation.state' in dataset.meta.stats:
            state_stats = dataset.meta.stats['observation.state']
            
            if 'normalize_inputs.buffer_observation_state.min' in state_dict:
                min_tensor = torch.tensor(state_stats['min'], dtype=torch.float32)
                state_dict['normalize_inputs.buffer_observation_state.min'] = min_tensor
                print(f"  ✓ 设置状态最小值: {min_tensor}")
            
            if 'normalize_inputs.buffer_observation_state.max' in state_dict:
                max_tensor = torch.tensor(state_stats['max'], dtype=torch.float32)
                state_dict['normalize_inputs.buffer_observation_state.max'] = max_tensor
                print(f"  ✓ 设置状态最大值: {max_tensor}")
        
        # 动作归一化统计信息
        if 'action' in dataset.meta.stats:
            action_stats = dataset.meta.stats['action']
            
            # normalize_targets (训练时用)
            if 'normalize_targets.buffer_action.min' in state_dict:
                min_tensor = torch.tensor(action_stats['min'], dtype=torch.float32)
                state_dict['normalize_targets.buffer_action.min'] = min_tensor
                print(f"  ✓ 设置动作目标最小值: {min_tensor}")
            
            if 'normalize_targets.buffer_action.max' in state_dict:
                max_tensor = torch.tensor(action_stats['max'], dtype=torch.float32)
                state_dict['normalize_targets.buffer_action.max'] = max_tensor
                print(f"  ✓ 设置动作目标最大值: {max_tensor}")
            
            # unnormalize_outputs (推理时用)
            if 'unnormalize_outputs.buffer_action.min' in state_dict:
                min_tensor = torch.tensor(action_stats['min'], dtype=torch.float32)
                state_dict['unnormalize_outputs.buffer_action.min'] = min_tensor
                print(f"  ✓ 设置动作输出最小值: {min_tensor}")
            
            if 'unnormalize_outputs.buffer_action.max' in state_dict:
                max_tensor = torch.tensor(action_stats['max'], dtype=torch.float32)
                state_dict['unnormalize_outputs.buffer_action.max'] = max_tensor
                print(f"  ✓ 设置动作输出最大值: {max_tensor}")
        
        # 5. 验证修复结果
        print("\n验证修复结果:")
        inf_count = 0
        for key in norm_keys:
            if torch.isinf(state_dict[key]).any():
                inf_count += 1
                print(f"  ❌ {key}: 仍有无穷值")
            else:
                print(f"  ✓ {key}: 正常")
        
        if inf_count == 0:
            print("✓ 所有归一化统计信息已修复")
        else:
            print(f"❌ 仍有 {inf_count} 个键包含无穷值")
            return False
        
        # 6. 保存修复后的模型
        from safetensors.torch import save_file
        
        backup_file = Path(model_path) / "model_backup_final.safetensors"
        if not backup_file.exists():
            import shutil
            shutil.copy2(model_file, backup_file)
            print("✓ 已备份原始模型")
        
        save_file(state_dict, model_file)
        print("✓ 修复后的模型已保存")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_model_inference(model_path: str):
    """测试修复后的模型推理"""
    print("\n" + "=" * 60)
    print("测试修复后的模型推理")
    print("=" * 60)
    
    try:
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.common.envs.configs import AlohaEnv
        import time
        
        # 加载模型
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "npu"
        
        # 应用最优NPU设置
        import torch_npu
        torch_npu.npu.set_compile_mode(jit_compile=False)
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        print("✓ 模型加载成功")
        
        # 创建测试数据
        test_batch = {
            'observation.images.top': torch.randn(1, 2, 3, 480, 640, device="npu"),
            'observation.state': torch.randn(1, 2, 14, device="npu")
        }
        
        print("测试推理...")
        
        # 预热
        with torch.no_grad():
            _ = policy.select_action(test_batch)
        
        # 正式测试
        start_time = time.time()
        actions = []
        
        for i in range(5):
            with torch.no_grad():
                action = policy.select_action(test_batch)
                actions.append(action.cpu().numpy())
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        import numpy as np
        actions = np.array(actions)
        
        print(f"✓ 推理成功！")
        print(f"  平均推理时间: {avg_time:.3f}s")
        print(f"  动作形状: {actions.shape}")
        print(f"  动作统计: mean={actions.mean():.6f}, std={actions.std():.6f}")
        print(f"  动作范围: [{actions.min():.6f}, {actions.max():.6f}]")
        
        # 检查动作变化
        action_std = np.std(actions, axis=0)
        print(f"  动作变化性: {action_std.mean():.6f}")
        
        if action_std.mean() > 1e-6:
            print("✓ 模型输出有合理的变化")
            return True
        else:
            print("⚠️ 模型输出变化很小")
            return False
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("LeRobot 归一化统计信息最终修复工具")
    print("=" * 60)
    
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
    
    if not Path(model_path).exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return 1
    
    # 修复归一化统计信息
    if extract_and_apply_normalization_stats(model_path):
        print("\n🎉 归一化统计信息修复成功！")
        
        # 测试修复后的模型
        if test_fixed_model_inference(model_path):
            print("\n🚀 模型修复完成！现在可以进行正常推理:")
            print("xvfb-run -a -s \"-screen 0 1600x900x30\" python -X faulthandler lerobot/scripts/eval.py --policy.path=outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model --output_dir=outputs/eval/diffusion_aloha_transfer/300000_final --env.type=aloha --env.task=AlohaTransferCube-v0 --policy.num_inference_steps=20")
        else:
            print("\n⚠️ 模型推理测试有问题，但归一化已修复")
    else:
        print("\n❌ 归一化统计信息修复失败")
        return 1

if __name__ == "__main__":
    exit(main())
