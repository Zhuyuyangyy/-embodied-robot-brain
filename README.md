# EmbodiedRobotBrain - 具身智能工业协作机器人"大脑"系统

**Transformer视觉编码器 + PPO强化学习 + 非结构化环境抓取**

## 项目概述

工业协作机器人在非结构化环境下的智能抓取系统：
- **视觉感知**：ViT/DINOv2 Transformer 编码器处理RGB-D输入
- **策略学习**：PPO (Proximal Policy Optimization) 强化学习
- **仿真训练**：PyBullet 物理仿真环境
- **硬件接口**：ROS2 通信协议

## 技术架构

```
RGB-D相机 ──→ Vision Transformer ──→ 空间特征
                    ↓                    ↓
         深度特征 ──────────────→ PPO Policy Network ──→ 机械臂关节控制
                                              ↓
              PyBullet仿真 / 真实机械臂 (ROS2)
```

## 核心创新

1. **ViT视觉编码**：比CNN更鲁棒的物体位姿估计
2. **Cross-attention**：视觉特征与机械臂状态联合建模
3. **PPO + Hindsight Experience Replay**：稀疏奖励下的高效学习
4. **Domain Randomization**：仿真到真实的迁移

## 环境要求

- Python 3.9+
- PyBullet / Isaac Sim
- ROS2 (用于真实硬件)
- GPU: RTX 3060+ (建议RTX 3090)

## 快速开始

```bash
cd backend
pip install -r requirements.txt
python envs/pybullet_sim.py      # 启动仿真环境
python rl/train_ppo.py          # 开始PPO训练
python app.py                    # 启动API服务
```
