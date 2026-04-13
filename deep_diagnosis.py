#!/usr/bin/env python3
"""
深度诊断脚本：分析视觉感知问题的根本原因
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Import torch_npu
try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
except ImportError:
    torch_npu = None

from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.configs import AlohaEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def analyze_dataset_visual_features():
    """分析数据集中的视觉特征"""
    print("=" * 60)
    print("分析数据集视觉特征")
    print("=" * 60)
    
    try:
        # 加载数据集
        dataset_path = "/root/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"
        dataset = LeRobotDataset(dataset_path)
        
        print(f"✓ 数据集加载成功: {dataset.num_episodes} episodes")
        
        # 分析几个样本的图像
        sample_indices = [0, 1000, 5000, 10000, 15000]
        
        for i, idx in enumerate(sample_indices):
            if idx >= len(dataset):
                continue
                
            sample = dataset[idx]
            image = sample['observation.images.top']
            
            print(f"\n样本 {idx}:")
            print(f"  图像形状: {image.shape}")
            print(f"  图像数据类型: {image.dtype}")
            print(f"  图像值范围: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  图像均值: {image.mean():.3f}")
            print(f"  图像标准差: {image.std():.3f}")
            
            # 转换为numpy用于分析
            if isinstance(image, torch.Tensor):
                img_np = image.numpy()
            else:
                img_np = image
            
            # 如果是CHW格式，转换为HWC
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # 分析红色通道
            if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                red_channel = img_np[:, :, 0]
                green_channel = img_np[:, :, 1]
                blue_channel = img_np[:, :, 2]
                
                print(f"  红色通道均值: {red_channel.mean():.3f}")
                print(f"  绿色通道均值: {green_channel.mean():.3f}")
                print(f"  蓝色通道均值: {blue_channel.mean():.3f}")
                
                # 检测红色区域
                red_mask = (red_channel > green_channel) & (red_channel > blue_channel)
                red_pixels = np.sum(red_mask)
                total_pixels = red_channel.size
                red_ratio = red_pixels / total_pixels
                
                print(f"  红色像素比例: {red_ratio:.4f} ({red_pixels}/{total_pixels})")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_training_vs_inference_preprocessing(model_path: str):
    """比较训练和推理时的预处理"""
    print("\n" + "=" * 60)
    print("比较训练vs推理预处理")
    print("=" * 60)
    
    try:
        # 加载模型配置
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "cpu"  # 先在CPU上测试
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        
        print("✓ 策略加载成功")
        
        # 检查输入特征配置
        input_features = policy.config.input_features
        print(f"✓ 输入特征: {list(input_features.keys())}")
        
        for key, feature in input_features.items():
            if 'image' in key:
                print(f"  {key}: 形状={feature.shape}, 类型={feature.type}")
        
        # 检查归一化配置
        if hasattr(policy, 'normalize_inputs'):
            print("\n归一化统计信息:")
            for name, buffer in policy.normalize_inputs.named_buffers():
                if 'observation_images_top' in name:
                    print(f"  {name}: {buffer}")
        
        # 创建测试环境
        env = make_env(env_cfg, n_envs=1)
        obs = env.reset()
        
        print(f"\n环境观察:")
        for key, value in obs.items():
            if 'image' in key:
                print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                print(f"    值范围=[{value.min():.3f}, {value.max():.3f}]")
                print(f"    均值={value.mean():.3f}, 标准差={value.std():.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 预处理比较失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_npu_vs_cpu_consistency(model_path: str):
    """测试NPU和CPU的一致性"""
    print("\n" + "=" * 60)
    print("测试NPU vs CPU一致性")
    print("=" * 60)
    
    try:
        # 创建测试数据
        batch_size = 1
        n_obs_steps = 2
        
        # 图像数据 (模拟相机输入)
        test_image = torch.randn(batch_size, n_obs_steps, 3, 480, 640)
        test_state = torch.randn(batch_size, n_obs_steps, 14)
        
        test_batch = {
            'observation.images.top': test_image,
            'observation.state': test_state
        }
        
        # CPU测试
        print("CPU测试:")
        config_cpu = PreTrainedConfig.from_pretrained(model_path)
        config_cpu.device = "cpu"
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy_cpu = make_policy(cfg=config_cpu, env_cfg=env_cfg)
        policy_cpu.eval()
        
        with torch.no_grad():
            cpu_action = policy_cpu.select_action(test_batch)
        
        print(f"  CPU动作: 形状={cpu_action.shape}")
        print(f"  CPU动作统计: mean={cpu_action.mean():.6f}, std={cpu_action.std():.6f}")
        print(f"  CPU动作范围: [{cpu_action.min():.6f}, {cpu_action.max():.6f}]")
        
        # NPU测试
        if torch_npu and torch_npu.npu.is_available():
            print("\nNPU测试:")
            config_npu = PreTrainedConfig.from_pretrained(model_path)
            config_npu.device = "npu"
            
            policy_npu = make_policy(cfg=config_npu, env_cfg=env_cfg)
            policy_npu.eval()
            
            # 转移数据到NPU
            npu_batch = {k: v.to("npu") for k, v in test_batch.items()}
            
            with torch.no_grad():
                npu_action = policy_npu.select_action(npu_batch)
            
            print(f"  NPU动作: 形状={npu_action.shape}")
            print(f"  NPU动作统计: mean={npu_action.mean():.6f}, std={npu_action.std():.6f}")
            print(f"  NPU动作范围: [{npu_action.min():.6f}, {npu_action.max():.6f}]")
            
            # 比较差异
            cpu_action_np = cpu_action.cpu().numpy()
            npu_action_np = npu_action.cpu().numpy()
            
            abs_diff = np.abs(cpu_action_np - npu_action_np)
            rel_diff = abs_diff / (np.abs(cpu_action_np) + 1e-8)
            
            print(f"\n一致性分析:")
            print(f"  最大绝对差异: {abs_diff.max():.8f}")
            print(f"  平均绝对差异: {abs_diff.mean():.8f}")
            print(f"  最大相对差异: {rel_diff.max():.8f}")
            print(f"  平均相对差异: {rel_diff.mean():.8f}")
            
            if abs_diff.max() > 1e-4:
                print("  ⚠️ 发现显著的NPU/CPU差异！")
                return False
            else:
                print("  ✓ NPU/CPU结果基本一致")
                return True
        else:
            print("  ⚠️ NPU不可用，跳过NPU测试")
            return True
            
    except Exception as e:
        print(f"❌ 一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_model_attention_to_red_objects(model_path: str):
    """分析模型对红色物体的注意力"""
    print("\n" + "=" * 60)
    print("分析模型对红色物体的注意力")
    print("=" * 60)
    
    try:
        # 创建不同的测试图像
        batch_size = 1
        n_obs_steps = 2
        
        # 测试1: 纯黑图像
        black_image = torch.zeros(batch_size, n_obs_steps, 3, 480, 640)
        
        # 测试2: 随机噪声图像
        noise_image = torch.randn(batch_size, n_obs_steps, 3, 480, 640) * 0.1 + 0.5
        
        # 测试3: 红色方块图像
        red_image = torch.zeros(batch_size, n_obs_steps, 3, 480, 640)
        # 在图像中心添加红色方块
        red_image[:, :, 0, 200:280, 280:360] = 1.0  # 红色通道
        
        # 测试4: 绿色方块图像
        green_image = torch.zeros(batch_size, n_obs_steps, 3, 480, 640)
        green_image[:, :, 1, 200:280, 280:360] = 1.0  # 绿色通道
        
        test_state = torch.zeros(batch_size, n_obs_steps, 14)
        
        test_cases = [
            ("黑色图像", black_image),
            ("噪声图像", noise_image),
            ("红色方块", red_image),
            ("绿色方块", green_image)
        ]
        
        # 加载模型
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "npu" if torch_npu and torch_npu.npu.is_available() else "cpu"
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        print(f"使用设备: {config.device}")
        
        results = []
        for name, test_image in test_cases:
            test_batch = {
                'observation.images.top': test_image.to(config.device),
                'observation.state': test_state.to(config.device)
            }
            
            with torch.no_grad():
                action = policy.select_action(test_batch)
            
            action_np = action.cpu().numpy()
            
            print(f"\n{name}:")
            print(f"  动作均值: {action_np.mean():.6f}")
            print(f"  动作标准差: {action_np.std():.6f}")
            print(f"  动作范围: [{action_np.min():.6f}, {action_np.max():.6f}]")
            
            results.append((name, action_np))
        
        # 分析动作差异
        print(f"\n动作差异分析:")
        black_action = results[0][1]
        for i, (name, action) in enumerate(results[1:], 1):
            diff = np.abs(action - black_action)
            print(f"  {name} vs 黑色图像: 最大差异={diff.max():.6f}, 平均差异={diff.mean():.6f}")
        
        # 比较红色vs绿色
        red_action = results[2][1]
        green_action = results[3][1]
        red_green_diff = np.abs(red_action - green_action)
        print(f"  红色 vs 绿色方块: 最大差异={red_green_diff.max():.6f}, 平均差异={red_green_diff.mean():.6f}")
        
        if red_green_diff.max() < 1e-4:
            print("  ⚠️ 模型对红色和绿色物体的反应几乎相同！")
            print("  💡 这表明模型可能没有学会区分颜色")
            return False
        else:
            print("  ✓ 模型对不同颜色有不同反应")
            return True
            
    except Exception as e:
        print(f"❌ 注意力分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    init_logging()
    
    print("LeRobot 深度视觉感知问题诊断")
    print("=" * 60)
    
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
    
    if not Path(model_path).exists():
        print(f"❌ 模型路径不存在: {model_path}")
        print("请确认300k步的模型已保存")
        return 1
    
    # 运行诊断测试
    tests = [
        ("数据集视觉特征分析", analyze_dataset_visual_features),
        ("训练vs推理预处理比较", lambda: compare_training_vs_inference_preprocessing(model_path)),
        ("NPU vs CPU一致性", lambda: test_npu_vs_cpu_consistency(model_path)),
        ("模型颜色感知能力", lambda: analyze_model_attention_to_red_objects(model_path)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("深度诊断总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n🔍 可能的根本原因:")
    print("1. 数据集质量问题 - 训练数据中红色物体标注不准确")
    print("2. 视觉编码器问题 - ResNet18可能不足以学习精细的颜色特征")
    print("3. 数据预处理问题 - 图像归一化可能消除了颜色信息")
    print("4. 任务设计问题 - 模型可能学会了位置模式而非视觉特征")

if __name__ == "__main__":
    exit(main())
