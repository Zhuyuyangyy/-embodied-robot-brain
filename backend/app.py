"""
EmbodiedRobotBrain Backend: FastAPI for Robot Arm Control
Provides inference API for the trained grasping model
"""
import sys
import io
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time

sys.path.insert(0, str(Path(__file__).parent))

from vision.transformer_encoder import RGBDEncoder, GraspQualityNetwork
from rl.ppo_agent import ActorCritic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="EmbodiedRobotBrain",
    description="具身智能工业协作机器人大脑 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vision_encoder = None
actor_critic = None
grasp_network = None


class ObservationInput(BaseModel):
    """Robot observation input"""
    rgb_image: List[List[List[int]]]  # (H, W, 3) RGB values 0-255
    depth_image: List[List[float]]    # (H, W) depth values
    robot_state: List[float]          # (14,) joint positions + velocities
    object_poses: Optional[List[List[float]]] = None


class ActionOutput(BaseModel):
    """Robot action output"""
    action: List[float]              # (4,) — delta_x, delta_y, delta_z, gripper
    value: float                     # State value estimate
    grasp_quality: Optional[float] = None


class GraspPlanningInput(BaseModel):
    """Grasp planning request"""
    target_object_position: List[float]  # (3,) — x, y, z
    rgb_image: List[List[List[int]]]
    depth_image: List[List[List[float]]]
    robot_state: List[float]


class GraspPlanningOutput(BaseModel):
    """Grasp planning response"""
    action_sequence: List[List[float]]
    predicted_success: float
    ee_trajectory: List[List[float]]
    grasp_config: Dict


def load_models():
    """Load trained models on startup"""
    global vision_encoder, actor_critic, grasp_network
    
    logger.info("Loading models...")
    
    cfg = config.get('model', {})
    d_model = cfg.get('d_model', 768)
    
    # Vision encoder
    vision_encoder = RGBDEncoder(
        d_model=d_model,
        use_dinov2=cfg.get('use_dinov2', False)
    ).to(device)
    
    # Actor-Critic
    actor_critic = ActorCritic(
        d_visual=d_model,
        d_state=d_model,
        n_actions=4
    ).to(device)
    
    # Grasp quality network
    grasp_network = GraspQualityNetwork(
        d_model=d_model,
        n_actions=4
    ).to(device)
    
    # Try to load checkpoints
    output_dir = Path(config.get('output', {}).get('output_dir', 'outputs'))
    
    vision_path = output_dir / "vision_encoder.pt"
    policy_path = output_dir / "ppo_grasp_best.pt"
    grasp_path = output_dir / "grasp_quality.pt"
    
    if vision_path.exists():
        vision_encoder.load_state_dict(torch.load(vision_path, map_location=device))
        logger.info(f"Loaded vision encoder from {vision_path}")
    else:
        logger.warning("No vision encoder checkpoint found, using random weights")
        
    if policy_path.exists():
        checkpoint = torch.load(policy_path, map_location=device)
        actor_critic.load_state_dict(checkpoint.get('actor_critic', checkpoint))
        logger.info(f"Loaded policy from {policy_path}")
    else:
        logger.warning("No policy checkpoint found, using random weights")
        
    vision_encoder.eval()
    actor_critic.eval()
    
    logger.info(f"Models loaded on {device}")


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/")
async def root():
    return {
        "service": "EmbodiedRobotBrain",
        "version": "1.0.0",
        "description": "具身智能工业协作机器人大脑"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "vision_encoder": "loaded" if vision_encoder else "not_loaded",
        "actor_critic": "loaded" if actor_critic else "not_loaded"
    }


@app.post("/predict/action", response_model=ActionOutput)
async def predict_action(obs: ObservationInput):
    """
    Predict robot action given current observation.
    
    Args:
        obs: Current observation (RGB, depth, robot state)
    Returns:
        action: (4,) — [delta_x, delta_y, delta_z, gripper]
        value: State value estimate
    """
    start_time = time.time()
    
    # Convert to tensors
    rgb = torch.tensor(obs.rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    depth = torch.tensor(obs.depth_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    robot_state = torch.tensor(obs.robot_state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Encode observation
        visual_features, state_features, _ = vision_encoder(
            rgb.to(device),
            depth.to(device),
            robot_state.to(device)
        )
        
        # Get action from policy
        action, log_prob, value = actor_critic.get_action(
            visual_features,
            state_features
        )
        
        # Get grasp quality
        grasp_quality = grasp_network(visual_features, state_features, action)
        
    inference_time = (time.time() - start_time) * 1000
    
    return ActionOutput(
        action=action.cpu().numpy()[0].tolist(),
        value=value.item(),
        grasp_quality=grasp_quality.item() if grasp_quality is not None else None
    )


@app.post("/plan/grasp", response_model=GraspPlanningOutput)
async def plan_grasp(plan_req: GraspPlanningInput):
    """
    Plan a complete grasp sequence for target object.
    
    Returns:
        action_sequence: Planned action sequence
        predicted_success: Estimated success probability
        ee_trajectory: End-effector trajectory
    """
    target_pos = np.array(plan_req.target_object_position)
    
    # Simplified grasp planning
    # Real implementation would use IK + motion planning
    
    action_sequence = []
    ee_trajectory = []
    
    # Approach
    approach_height = target_pos[2] + 0.15
    for t in range(10):
        alpha = t / 10.0
        ee_pos = [
            target_pos[0] + (0.2 - target_pos[0]) * (1 - alpha),
            target_pos[1] + (0 - target_pos[1]) * (1 - alpha),
            approach_height
        ]
        ee_trajectory.append(ee_pos)
        action_sequence.append([ee_pos[0], ee_pos[1], ee_pos[2], 1.0])  # Approach, gripper open
    
    # Descend
    for t in range(5):
        alpha = t / 5.0
        ee_pos = [
            target_pos[0],
            target_pos[1],
            approach_height - alpha * 0.1
        ]
        ee_trajectory.append(ee_pos)
        action_sequence.append([0, 0, -0.02, 1.0])  # Descend
    
    # Grasp
    action_sequence.append([0, 0, 0, -1.0])  # Close gripper
    ee_trajectory.append(target_pos.tolist())
    
    # Predict success
    predicted_success = np.random.uniform(0.7, 0.95)  # Placeholder
    
    return GraspPlanningOutput(
        action_sequence=action_sequence,
        predicted_success=predicted_success,
        ee_trajectory=ee_trajectory,
        grasp_config={
            "approach_angle": 0,
            "gripper_width": 0.03,
            "force": 10.0
        }
    )


@app.get("/robot/state")
async def get_robot_state():
    """
    Get simulated robot state.
    In real deployment, this would connect to ROS2.
    """
    # Simulated state
    return {
        "joint_positions": [0.0] * 7,
        "joint_velocities": [0.0] * 7,
        "end_effector_pose": {
            "position": [0.5, 0.0, 0.2],
            "orientation": [0, 0, 0, 1]
        },
        "gripper_width": 0.08,
        "timestamp": time.time()
    }


@app.post("/robot/command")
async def send_robot_command(command: Dict):
    """
    Send command to robot.
    In real deployment, this publishes to ROS2.
    
    Args:
        command: Dict with 'action' or 'joint_positions'
    """
    logger.info(f"Received robot command: {command}")
    
    # Simulated response
    return {
        "status": "received",
        "command_id": f"cmd_{int(time.time() * 1000)}",
        "estimated_execution_time_ms": 100
    }


@app.get("/model/info")
async def model_info():
    """Get model architecture info"""
    return {
        "model_type": "PPO + ViT",
        "vision_encoder": {
            "type": "ViT / DINOv2",
            "d_model": config.get('model', {}).get('d_model', 768)
        },
        "policy": {
            "type": "Actor-Critic",
            "n_actions": 4
        },
        "grasp_quality": {
            "type": "GraspQualityNetwork",
            "output": "success probability 0-1"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013, workers=1)
