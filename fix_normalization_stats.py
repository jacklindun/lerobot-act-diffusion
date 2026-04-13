#!/usr/bin/env python3
"""
修复模型归一化统计信息的脚本
"""

import logging
import torch
import sys
import os
import json
from pathlib import Path

# Import torch_npu first
try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
except ImportError as e:
    print(f"torch_npu import failed: {e}")
    torch_npu = None

from lerobot.common.utils.utils import init_logging

def fix_model_normalization_stats(model_path: str, dataset_repo_id: str):
    """修复模型的归一化统计信息"""
    print("=" * 60)
    print("修复模型归一化统计信息")
    print("=" * 60)
    
    try:
        # 1. 加载数据集获取正确的统计信息
        print("正在加载数据集...")
        
        # 创建一个临时配置来加载数据集
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        # 使用本地数据集路径
        local_dataset_path = "/root/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"

        # 直接使用LeRobotDataset加载本地数据集
        dataset = LeRobotDataset(local_dataset_path)
        
        print(f"✓ 数据集加载成功: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        print(f"✓ 数据集统计信息: {list(dataset.meta.stats.keys())}")
        
        # 2. 检查模型文件
        model_file = Path(model_path) / "model.safetensors"
        config_file = Path(model_path) / "config.json"
        
        if not model_file.exists():
            print(f"❌ 模型文件不存在: {model_file}")
            return False
        
        if not config_file.exists():
            print(f"❌ 配置文件不存在: {config_file}")
            return False
        
        # 3. 加载并修复配置文件
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # 确保配置文件有type字段
        if 'type' not in config_data:
            config_data['type'] = 'diffusion'
            print("✓ 添加了缺失的 'type' 字段")
        
        # 保存修复的配置文件
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("✓ 配置文件已修复")
        
        # 4. 加载模型权重
        from safetensors.torch import load_file, save_file
        
        print("正在加载模型权重...")
        state_dict = load_file(model_file)
        
        print(f"✓ 模型权重加载成功，包含 {len(state_dict)} 个参数")
        
        # 5. 检查并修复归一化统计信息
        stats_keys_to_fix = []
        for key in state_dict.keys():
            if 'buffer_' in key and ('mean' in key or 'std' in key or 'min' in key or 'max' in key):
                if torch.isinf(state_dict[key]).any():
                    stats_keys_to_fix.append(key)
        
        print(f"发现 {len(stats_keys_to_fix)} 个需要修复的统计信息键")
        
        # 6. 从数据集统计信息中修复
        for key in stats_keys_to_fix:
            # 解析键名，例如 "normalize_inputs.buffer_observation_state.min"
            if 'observation_state' in key:
                dataset_key = 'observation.state'
            elif 'observation_images_top' in key:
                dataset_key = 'observation.images.top'
            elif 'action' in key:
                dataset_key = 'action'
            else:
                print(f"⚠ 无法识别的键: {key}")
                continue
            
            if dataset_key not in dataset.meta.stats:
                print(f"⚠ 数据集中没有 {dataset_key} 的统计信息")
                continue
            
            # 获取对应的统计值
            if 'mean' in key:
                stat_value = dataset.meta.stats[dataset_key]['mean']
            elif 'std' in key:
                stat_value = dataset.meta.stats[dataset_key]['std']
            elif 'min' in key:
                stat_value = dataset.meta.stats[dataset_key]['min']
            elif 'max' in key:
                stat_value = dataset.meta.stats[dataset_key]['max']
            else:
                continue
            
            # 转换为tensor并替换
            if isinstance(stat_value, torch.Tensor):
                state_dict[key] = stat_value.clone()
            else:
                state_dict[key] = torch.tensor(stat_value, dtype=torch.float32)
            
            print(f"✓ 修复了 {key}: {state_dict[key]}")
        
        # 7. 检查并修复无穷值参数
        inf_keys = []
        for key, value in state_dict.items():
            if torch.isinf(value).any():
                inf_keys.append(key)
        
        print(f"发现 {len(inf_keys)} 个包含无穷值的参数")
        
        for key in inf_keys:
            if 'buffer_' not in key:  # 只修复非统计信息的参数
                # 用零替换无穷值
                state_dict[key] = torch.where(torch.isinf(state_dict[key]), 
                                            torch.zeros_like(state_dict[key]), 
                                            state_dict[key])
                print(f"✓ 修复了无穷值参数: {key}")
        
        # 8. 保存修复后的模型
        backup_file = Path(model_path) / "model_backup.safetensors"
        if not backup_file.exists():
            # 备份原始模型
            import shutil
            shutil.copy2(model_file, backup_file)
            print("✓ 已备份原始模型")
        
        # 保存修复后的模型
        save_file(state_dict, model_file)
        print("✓ 修复后的模型已保存")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    init_logging()
    
    print("LeRobot 模型归一化统计信息修复工具")
    print("=" * 60)
    
    # 模型路径和数据集
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/100000/pretrained_model"
    dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return 1
    
    print(f"✓ 模型路径: {model_path}")
    print(f"✓ 数据集: {dataset_repo_id}")
    
    # 执行修复
    success = fix_model_normalization_stats(model_path, dataset_repo_id)
    
    if success:
        print("\n🎉 模型修复成功！")
        print("现在可以重新运行推理测试:")
        print(f"xvfb-run -a -s \"-screen 0 1600x900x30\" python -X faulthandler lerobot/scripts/eval.py --policy.path={model_path} --output_dir=outputs/eval/diffusion_aloha_transfer/100000 --env.type=aloha --env.task=AlohaTransferCube-v0")
        return 0
    else:
        print("\n❌ 模型修复失败")
        return 1

if __name__ == "__main__":
    exit(main())
