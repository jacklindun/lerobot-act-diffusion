#!/usr/bin/env python3
"""
修复300k步模型的归一化统计信息
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def fix_300k_model():
    """修复300k步模型"""
    print("=" * 60)
    print("修复300k步模型的归一化统计信息")
    print("=" * 60)
    
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
    
    if not Path(model_path).exists():
        print(f"❌ 300k模型路径不存在: {model_path}")
        return False
    
    try:
        # 1. 加载数据集统计信息
        print("正在加载数据集...")
        dataset_path = "/root/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"
        dataset = LeRobotDataset(dataset_path)
        print(f"✓ 数据集加载成功")
        
        # 2. 修复配置文件
        config_file = Path(model_path) / "config.json"
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        if 'type' not in config_data:
            config_data['type'] = 'diffusion'
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print("✓ 配置文件已修复")
        
        # 3. 修复模型权重
        model_file = Path(model_path) / "model.safetensors"
        state_dict = load_file(model_file)
        
        print(f"✓ 模型权重加载成功，包含 {len(state_dict)} 个参数")
        
        # 4. 修复归一化统计信息
        stats_keys_to_fix = []
        for key in state_dict.keys():
            if 'buffer_' in key and ('mean' in key or 'std' in key or 'min' in key or 'max' in key):
                if torch.isinf(state_dict[key]).any():
                    stats_keys_to_fix.append(key)
        
        print(f"发现 {len(stats_keys_to_fix)} 个需要修复的统计信息键")
        
        for key in stats_keys_to_fix:
            if 'observation_state' in key:
                dataset_key = 'observation.state'
            elif 'observation_images_top' in key:
                dataset_key = 'observation.images.top'
            elif 'action' in key:
                dataset_key = 'action'
            else:
                continue
            
            if dataset_key not in dataset.meta.stats:
                continue
            
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
            
            if isinstance(stat_value, torch.Tensor):
                state_dict[key] = stat_value.clone()
            else:
                state_dict[key] = torch.tensor(stat_value, dtype=torch.float32)
            
            print(f"✓ 修复了 {key}")
        
        # 5. 备份并保存
        backup_file = Path(model_path) / "model_backup_300k.safetensors"
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

def test_fixed_model():
    """测试修复后的模型"""
    print("\n" + "=" * 60)
    print("测试修复后的模型")
    print("=" * 60)
    
    try:
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.common.envs.configs import AlohaEnv
        
        model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
        
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "cpu"  # 先在CPU上测试
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        print("✓ 模型加载成功")
        
        # 创建测试数据
        test_batch = {
            'observation.images.top': torch.randn(1, 2, 3, 480, 640),
            'observation.state': torch.randn(1, 2, 14)
        }
        
        with torch.no_grad():
            action = policy.select_action(test_batch)
        
        print(f"✓ 推理测试成功: {action.shape}")
        print(f"✓ 动作统计: mean={action.mean():.6f}, std={action.std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("修复300k步模型工具")
    print("=" * 60)
    
    # 修复模型
    if fix_300k_model():
        print("\n🎉 模型修复成功！")
        
        # 测试修复后的模型
        if test_fixed_model():
            print("\n🚀 现在可以重新测试推理:")
            print("xvfb-run -a -s \"-screen 0 1600x900x30\" python -X faulthandler lerobot/scripts/eval.py --policy.path=outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model --output_dir=outputs/eval/diffusion_aloha_transfer/300000_fixed --env.type=aloha --env.task=AlohaTransferCube-v0 --policy.num_inference_steps=20")
        else:
            print("\n❌ 模型测试失败")
    else:
        print("\n❌ 模型修复失败")

if __name__ == "__main__":
    main()
