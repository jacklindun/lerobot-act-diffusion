#!/usr/bin/env python3
"""
测试不同模型格式在NPU上的推理效果
"""

import torch
import numpy as np
from pathlib import Path
import time

# Import torch_npu
try:
    import torch_npu
    print(f"torch_npu version: {torch_npu.__version__}")
except ImportError:
    torch_npu = None

from lerobot.common.utils.utils import init_logging
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.envs.configs import AlohaEnv

def test_original_model_npu(model_path: str):
    """测试原始模型在NPU上的推理"""
    print("=" * 60)
    print("测试原始模型在NPU上的推理")
    print("=" * 60)
    
    try:
        # 加载模型
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "npu"
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        print("✓ 模型加载成功")
        
        # 创建测试数据
        test_batch = {
            'observation.images.top': torch.randn(1, 2, 3, 480, 640, device="npu"),
            'observation.state': torch.randn(1, 2, 14, device="npu")
        }
        
        # 测试推理速度和结果
        print("测试推理...")
        
        # 预热
        with torch.no_grad():
            _ = policy.select_action(test_batch)
        
        # 计时测试
        start_time = time.time()
        actions = []
        
        for i in range(5):
            with torch.no_grad():
                action = policy.select_action(test_batch)
                actions.append(action.cpu().numpy())
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        actions = np.array(actions)
        
        print(f"✓ 推理成功")
        print(f"  平均推理时间: {avg_time:.3f}s")
        print(f"  动作形状: {actions.shape}")
        print(f"  动作统计: mean={actions.mean():.6f}, std={actions.std():.6f}")
        print(f"  动作范围: [{actions.min():.6f}, {actions.max():.6f}]")
        
        # 检查动作一致性
        action_std = np.std(actions, axis=0)
        print(f"  动作一致性(std): {action_std.mean():.6f}")
        
        if action_std.mean() < 1e-6:
            print("  ⚠️ 动作几乎没有变化，可能存在问题")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 原始模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torchscript_conversion(model_path: str):
    """测试TorchScript转换"""
    print("\n" + "=" * 60)
    print("测试TorchScript转换")
    print("=" * 60)
    
    try:
        # 加载模型
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = "cpu"  # 先在CPU上转换
        
        env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        policy.eval()
        
        print("✓ 模型加载成功")
        
        # 创建示例输入
        example_input = {
            'observation.images.top': torch.randn(1, 2, 3, 480, 640),
            'observation.state': torch.randn(1, 2, 14)
        }
        
        print("尝试TorchScript转换...")
        
        # 方法1: torch.jit.trace
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(policy.select_action, (example_input,))
            print("✓ torch.jit.trace 成功")
            
            # 测试traced模型
            with torch.no_grad():
                traced_output = traced_model(example_input)
                original_output = policy.select_action(example_input)
            
            diff = torch.abs(traced_output - original_output).max()
            print(f"  输出差异: {diff:.8f}")
            
            if diff < 1e-5:
                print("  ✓ TorchScript模型输出一致")
                return True
            else:
                print("  ⚠️ TorchScript模型输出不一致")
                
        except Exception as e:
            print(f"  ❌ torch.jit.trace 失败: {e}")
        
        # 方法2: torch.jit.script
        try:
            scripted_model = torch.jit.script(policy)
            print("✓ torch.jit.script 成功")
            return True
        except Exception as e:
            print(f"  ❌ torch.jit.script 失败: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ TorchScript转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_npu_optimization_settings():
    """测试不同的NPU优化设置"""
    print("\n" + "=" * 60)
    print("测试NPU优化设置")
    print("=" * 60)
    
    if not torch_npu or not torch_npu.npu.is_available():
        print("❌ NPU不可用")
        return False
    
    try:
        print("当前NPU设置:")
        print(f"  设备数量: {torch_npu.npu.device_count()}")
        print(f"  当前设备: {torch_npu.npu.current_device()}")
        
        # 测试不同的优化设置
        optimization_configs = [
            {"jit_compile": False, "allow_hf32": False},
            {"jit_compile": False, "allow_hf32": True},
            {"jit_compile": True, "allow_hf32": False},
            {"jit_compile": True, "allow_hf32": True},
        ]
        
        for i, config in enumerate(optimization_configs):
            print(f"\n测试配置 {i+1}: {config}")
            
            try:
                # 设置NPU优化
                if hasattr(torch_npu.npu, 'set_compile_mode'):
                    torch_npu.npu.set_compile_mode(jit_compile=config["jit_compile"])
                
                if hasattr(torch_npu.npu, 'set_option'):
                    torch_npu.npu.set_option({"ACL_OP_SELECT_IMPL_MODE": "high_performance"})
                
                # 创建简单测试
                x = torch.randn(100, 100, device="npu")
                y = torch.randn(100, 100, device="npu")
                
                start_time = time.time()
                for _ in range(10):
                    z = torch.mm(x, y)
                    torch_npu.npu.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                print(f"  矩阵乘法平均时间: {avg_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"  ❌ 配置 {i+1} 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU优化设置测试失败: {e}")
        return False

def test_mixed_precision_settings(model_path: str):
    """测试不同的混合精度设置"""
    print("\n" + "=" * 60)
    print("测试混合精度设置")
    print("=" * 60)
    
    try:
        precision_configs = [
            {"use_amp": False, "dtype": torch.float32},
            {"use_amp": True, "dtype": torch.float16},
            {"use_amp": True, "dtype": torch.bfloat16},
        ]
        
        for i, config in enumerate(precision_configs):
            print(f"\n测试精度配置 {i+1}: {config}")
            
            try:
                # 加载模型
                model_config = PreTrainedConfig.from_pretrained(model_path)
                model_config.device = "npu"
                model_config.use_amp = config["use_amp"]
                
                env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
                policy = make_policy(cfg=model_config, env_cfg=env_cfg)
                policy.eval()
                
                # 转换模型精度
                if config["dtype"] != torch.float32:
                    policy = policy.to(dtype=config["dtype"])
                
                # 创建测试数据
                test_batch = {
                    'observation.images.top': torch.randn(1, 2, 3, 480, 640, device="npu", dtype=config["dtype"]),
                    'observation.state': torch.randn(1, 2, 14, device="npu", dtype=config["dtype"])
                }
                
                # 测试推理
                start_time = time.time()
                with torch.no_grad():
                    if config["use_amp"]:
                        with torch.autocast(device_type="npu", dtype=config["dtype"]):
                            action = policy.select_action(test_batch)
                    else:
                        action = policy.select_action(test_batch)
                end_time = time.time()
                
                print(f"  ✓ 推理成功，时间: {(end_time-start_time)*1000:.2f}ms")
                print(f"  动作dtype: {action.dtype}")
                print(f"  动作统计: mean={action.mean():.6f}, std={action.std():.6f}")
                
            except Exception as e:
                print(f"  ❌ 精度配置 {i+1} 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 混合精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    init_logging()
    
    print("LeRobot NPU模型格式和优化测试")
    print("=" * 60)
    
    model_path = "outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model"
    
    if not Path(model_path).exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return 1
    
    # 运行测试
    tests = [
        ("原始模型NPU推理", lambda: test_original_model_npu(model_path)),
        ("NPU优化设置", test_npu_optimization_settings),
        ("混合精度设置", lambda: test_mixed_precision_settings(model_path)),
        ("TorchScript转换", lambda: test_torchscript_conversion(model_path)),
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
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n💡 建议:")
    print("1. 如果原始模型推理正常，问题可能不在模型格式")
    print("2. 如果TorchScript转换成功，可以尝试使用转换后的模型")
    print("3. 如果NPU优化设置有效，可以调整相应参数")
    print("4. 考虑使用不同的混合精度设置来提高性能")

if __name__ == "__main__":
    exit(main())
