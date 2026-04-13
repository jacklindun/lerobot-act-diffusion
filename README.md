<<<<<<< HEAD
# lerobot-act-diffusion
基于昇藤NPU复现lerobot中的双臂操作任务
=======
# LeRobot (Ascend NPU Reproduction): ACT + Diffusion

本项目基于 [huggingface/lerobot](https://github.com/huggingface/lerobot) 进行二次开发，目标是在昇腾 NPU 平台复现 LeRobot 中的 `ACT` 和 `Diffusion` 策略。

当前复现任务为：
- `AlohaTransferCube-v0`（双臂协作夹取/转移小方块）

项目重点是把策略训练、推理、诊断流程跑通，并针对 NPU 推理与归一化问题做排查和修复脚本。

## 1. 项目特点

- 在昇腾 NPU 上复现 ACT 与 Diffusion 策略
- 提供 ACT 一键训练脚本（标准配置/高级配置）
- 提供 Diffusion 训练续跑、推理参数调优建议
- 提供 NPU 推理诊断、模型文件检查、归一化统计修复工具

## 2. 关键文件说明

以下是仓库根目录下与本项目复现最相关的脚本：

- `train_act_npu.py`：ACT 在 NPU 上的训练入口（含标准/高级配置）
- `analyze_training_progress.py`：分析 Diffusion 训练进度并给出续训/推理建议
- `diagnose_npu_inference.py`：排查 NPU 推理问题（设备一致性、动作分布等）
- `check_model_files.py`：检查 checkpoint / 模型文件完整性
- `fix_normalization_stats.py`：修复归一化统计信息（早期版本）
- `fix_normalization_final.py`：归一化统计信息最终修复与推理验证
- `fix_300k_model.py`：针对 300k checkpoint 的专项修复脚本
- `test_model_formats.py` / `deep_diagnosis.py` / `inspect_model_structure.py`：诊断与结构检查辅助工具
- `fixed_config.json`：Diffusion 策略修复后可参考配置

## 3. 环境准备

建议使用 Linux + Conda（并已完成 Ascend CANN、驱动、`torch_npu` 的基础安装）。

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
```

说明：
- `torch` / `torch_npu` 需按你机器上的 CANN 与 PyTorch 版本矩阵安装。
- 本仓库默认使用 `policy.device=npu`（NPU 不可用时请改为 `cpu` 先验证流程）。

## 4. 数据集与任务

- 数据集：`lerobot/aloha_sim_transfer_cube_human`
- 环境：
  - `--env.type=aloha`
  - `--env.task=AlohaTransferCube-v0`

本项目默认围绕该任务进行训练与评估。

## 5. 训练与评估

### 5.1 ACT 训练（推荐入口）

使用交互脚本：

```bash
python train_act_npu.py
```

脚本内置两套配置：
- 标准配置：`outputs/train/act_aloha_transfer_npu`
- 高级配置：`outputs/train/act_aloha_transfer_npu_advanced`

训练结束后，脚本会给出对应的评估命令。

### 5.2 Diffusion 训练（基线命令）

```bash
xvfb-run -s -a "-screen 0 1600x900x30" python lerobot/scripts/train.py \
  --output_dir=outputs/train/diffusion_aloha_transfer_npu \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --wandb.enable=false \
  --steps=300000
```

### 5.3 推理评估（Diffusion 示例）

```bash
xvfb-run -a -s "-screen 0 1600x900x30" python -X faulthandler lerobot/scripts/eval.py \
  --policy.path=outputs/train/diffusion_aloha_transfer_npu/checkpoints/300000/pretrained_model \
  --output_dir=outputs/eval/diffusion_aloha_transfer/300000_final \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --policy.num_inference_steps=20
```

## 6. 常用排查流程

1. 模型文件完整性检查：

```bash
python check_model_files.py
```

2. 训练进度与续训建议：

```bash
python analyze_training_progress.py
```

3. NPU 推理问题诊断：

```bash
python diagnose_npu_inference.py
```

4. 归一化统计修复（按需要）：

```bash
python fix_normalization_final.py
```

## 7. 与原版 LeRobot 的关系

- 本仓库保留了 LeRobot 原始框架代码与结构。
- 在此基础上新增了面向 Ascend NPU 的训练/诊断/修复脚本。
- 复现重点为 `ACT` 与 `Diffusion` 在 `AlohaTransferCube-v0` 任务上的可运行性与稳定性。

## 8. 致谢

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- ACT、Diffusion 相关开源工作与数据集贡献者

## 9. 许可证

本仓库继承原项目许可证：`Apache-2.0`（详见 `LICENSE`）。
>>>>>>> c7dd226 (init Ascend NPU reproduction for ACT and Diffusion)
