"""
PyBullet Simulation Environment for Robotic Grasping
Industrial robot arm in non-structured environment
"""
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional
import random


class PandaRobot:
    """
    Franka Panda robot arm simulation in PyBullet.
    7-DOF arm with parallel jaw gripper.
    """
    
    def __init__(self, client_id: int):
        self.client_id = client_id
        
        # Joint indices
        self.arm_joints = list(range(7))  # 7 arm joints
        self.gripper_joints = [9, 10]  # Gripper
        
        # Joint limits
        self.lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.upper_limits = np.array([2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973])
        
        # Load robot
        self.robot_id = None
        self.ee_index = 11  # End effector link index
        
    def load(self, base_position=(0, 0, 0)):
        """Load robot into simulation"""
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=base_position,
            useFixedBase=True,
            physicsClientId=self.client_id
        )
        
        # Set up collision margins
        for ji in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            p.setJointMotorControl2(
                self.robot_id, ji,
                p.POSITION_CONTROL,
                force=500,
                physicsClientId=self.client_id
            )
            
        return self.robot_id
        
    def reset(self, joint_positions: np.ndarray = None):
        """Reset robot to initial pose"""
        if joint_positions is None:
            joint_positions = np.zeros(7)
            
        for i, pos in enumerate(joint_positions):
            p.resetJointState(
                self.robot_id, i, pos,
                physicsClientId=self.client_id
            )
            
        # Reset gripper
        p.resetJointState(self.robot_id, 9, 0.04, physicsClientId=self.client_id)
        p.resetJointState(self.robot_id, 10, 0.04, physicsClientId=self.client_id)
        
    def get_joint_states(self) -> np.ndarray:
        """Get current joint positions and velocities"""
        states = p.getJointStates(
            self.robot_id,
            self.arm_joints,
            physicsClientId=self.client_id
        )
        positions = np.array([s[0] for s in states])
        velocities = np.array([s[1] for s in states])
        return np.concatenate([positions, velocities])
        
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector position and orientation (quaternion)"""
        ee_state = p.getLinkState(
            self.robot_id, self.ee_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id
        )
        pos = np.array(ee_state[0])
        orn = np.array(ee_state[1])
        return pos, orn
        
    def set_joint_positions(self, target_positions: np.ndarray):
        """Set target joint positions"""
        for i, pos in enumerate(target_positions[:7]):
            p.setJointMotorControl2(
                self.robot_id, i,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=500,
                physicsClientId=self.client_id
            )
            
    def set_gripper(self, open: bool = True):
        """Open or close gripper"""
        if open:
            p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.08)
            p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.08)
        else:
            p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.0)
            p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.0)
            
    def get_gripper_width(self) -> float:
        """Get current gripper width"""
        state9 = p.getJointState(self.robot_id, 9, physicsClientId=self.client_id)[0]
        state10 = p.getJointState(self.robot_id, 10, physicsClientId=self.client_id)[0]
        return state9 + state10


class GraspingEnvironment:
    """
    PyBullet-based grasping environment for robotic manipulation.
    
    Features:
    - Panda robot arm
    - Random object placement
    - Multiple object types (cups, boxes, spheres)
    - Success detection via grasp stability
    - Domain randomization (lighting, object properties)
    """
    
    def __init__(
        self,
        render: bool = True,
        use_gui: bool = True,
        n_objects: int = 3,
        workspace_size: float = 0.4,
        domain_randomization: bool = True
    ):
        self.render = render
        self.use_gui = use_gui
        self.n_objects = n_objects
        self.workspace_size = workspace_size
        self.domain_randomization = domain_randomization
        
        self.client_id = None
        self.robot = None
        self.object_ids = []
        self.table_id = None
        
        # State
        self.current_step = 0
        self.max_steps = 200
        
        # Observation space: RGB image + depth + robot state
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'depth': spaces.Box(0, 1, (224, 224, 1), dtype=np.float32),
            'robot_state': spaces.Box(-np.inf, np.inf, (14,)),  # joint pos + vel
            'object_poses': spaces.Box(-1, 1, (10, 7)),  # object poses (x,y,z,rx,ry,rz,rw)
        })
        
        # Action space: delta joint positions + gripper
        # Or: delta ee position (3) + gripper (1)
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        
    def _connect(self):
        """Connect to PyBullet"""
        if self.use_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        
    def _setup_scene(self):
        """Set up the scene with robot, table, objects"""
        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # Load table
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02], rgbaColor=[0.7, 0.5, 0.3, 1])
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, -0.02],
            physicsClientId=self.client_id
        )
        
        # Load robot
        self.robot = PandaRobot(self.client_id)
        self.robot.load(base_position=[0, 0, 0])
        self.robot.reset()
        
    def _spawn_objects(self):
        """Spawn random objects on table"""
        self.object_ids = []
        
        for i in range(self.n_objects):
            # Random object type
            obj_type = random.choice(['box', 'sphere', 'cylinder'])
            
            # Random size
            if obj_type == 'box':
                size = np.random.uniform(0.02, 0.06, 3)
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size)
            elif obj_type == 'sphere':
                radius = np.random.uniform(0.02, 0.04)
                collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
            else:
                radius = np.random.uniform(0.02, 0.04)
                height = np.random.uniform(0.04, 0.1)
                collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height)
            
            # Random position on table
            x = np.random.uniform(0.35, 0.65)
            y = np.random.uniform(-0.2, 0.2)
            z = 0.05 if obj_type != 'sphere' else 0.05
            
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z],
                physicsClientId=self.client_id
            )
            
            # Random color
            color = [random.random() for _ in range(3)] + [1]
            p.changeVisualShape(obj_id, -1, rgbaColor=color, physicsClientId=self.client_id)
            
            # Friction
            p.changeDynamics(obj_id, -1, lateralFriction=1.0, physicsClientId=self.client_id)
            
            self.object_ids.append(obj_id)
            
    def reset(self) -> Dict:
        """Reset environment"""
        self._connect()
        self._setup_scene()
        self._spawn_objects()
        
        if self.domain_randomization:
            self._apply_domain_randomization()
            
        self.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        return obs
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action.
        
        Args:
            action: (4,) — [delta_x, delta_y, delta_z, gripper_cmd]
        Returns:
            obs, reward, done, info
        """
        self.current_step += 1
        
        # Parse action
        delta_xyz = action[:3] * 0.1  # Scale to 10cm max
        gripper_cmd = action[3]
        
        # Get current ee pose
        ee_pos, ee_orn = self.robot.get_ee_pose()
        
        # Compute target position
        target_pos = ee_pos + delta_xyz
        
        # Inverse kinematics to get joint positions
        target_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # Pointing down
        joint_positions = p.calculateInverseKinematics(
            self.robot.robot_id,
            self.robot.ee_index,
            target_pos,
            target_orn,
            physicsClientId=self.client_id
        )[:7]  # Only arm joints
        
        # Apply action
        self.robot.set_joint_positions(joint_positions)
        
        # Gripper
        if gripper_cmd > 0:
            self.robot.set_gripper(open=True)
        else:
            self.robot.set_gripper(open=False)
            
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward, success = self._compute_reward()
        
        # Check termination
        done = success or self.current_step >= self.max_steps
        
        info = {
            'success': success,
            'gripper_width': self.robot.get_gripper_width(),
            'ee_position': ee_pos.tolist(),
            'n_objects': len(self.object_ids)
        }
        
        return obs, reward, done, info
        
    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Robot state
        robot_state = self.robot.get_joint_states()
        
        # Camera observation (RGB + Depth)
        rgb, depth = self._render_camera()
        
        # Object poses
        object_poses = []
        for obj_id in self.object_ids:
            pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
            pose = list(pos) + list(orn)
            object_poses.append(pose)
            
        # Pad to fixed size
        object_poses = np.array(object_poses + [[0] * 7] * (10 - len(object_poses)))
        
        return {
            'rgb': rgb,
            'depth': depth,
            'robot_state': robot_state,
            'object_poses': object_poses
        }
        
    def _render_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        """Render camera view"""
        # Camera parameters
        cam_pos = [0.5, 0, 0.3]
        cam_target = [0.5, 0, 0]
        cam_up = [0, 0, 1]
        
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)
        
        # Render
        img_arr = p.getCameraImage(
            224, 224,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HELIUM,
            physicsClientId=self.client_id
        )
        
        rgb = np.array(img_arr[2]).reshape(224, 224, 4)[:, :, :3]  # RGBA → RGB
        depth = np.array(img_arr[3]).reshape(224, 224)  # Depth buffer
        depth = depth / 1000.0  # Convert to meters
        depth = np.clip(depth, 0, 2) / 2.0  # Normalize
        
        return rgb.astype(np.uint8), depth.astype(np.float32)
        
    def _compute_reward(self) -> Tuple[float, bool]:
        """Compute reward and success"""
        reward = -0.01  # Step cost
        
        # Check if any object is grasped (close to ee and gripper closed)
        ee_pos, _ = self.robot.get_ee_pose()
        gripper_width = self.robot.get_gripper_width()
        
        success = False
        for obj_id in self.object_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
            distance = np.linalg.norm(np.array(obj_pos) - ee_pos)
            
            # Grasp detection
            if distance < 0.05 and gripper_width < 0.02:
                reward += 10.0  # Big reward for grasp
                success = True
                break
                
        return reward, success
        
    def _apply_domain_randomization(self):
        """Apply random variations to scene"""
        # Lighting
        p.setLightParameter(
            self.object_ids[0] if self.object_ids else -1,
            p.LIGHT_SHADOW, 0,
            physicsClientId=self.client_id
        )
        
    def close(self):
        """Clean up"""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None


class SimToRealTransfer:
    """
    Techniques for transferring policies from simulation to real robot.
    
    - Domain randomization
    - System identification
    - Dynamics randomization
    """
    
    def __init__(self, env: GraspingEnvironment):
        self.env = env
        
    def apply_dynamics_randomization(self, magnitude: float = 0.1):
        """Randomize physics parameters"""
        p.setGravity(0, 0, -9.81 * (1 + np.random.uniform(-magnitude, magnitude)))
        
        for obj_id in self.env.object_ids:
            friction = 1.0 * (1 + np.random.uniform(-magnitude, magnitude))
            p.changeDynamics(obj_id, -1, lateralFriction=friction)
            
    def apply_visual_randomization(self):
        """Randomize visual appearance"""
        for obj_id in self.env.object_ids:
            color = [np.random.uniform(0.3, 0.9) for _ in range(3)] + [1]
            p.changeVisualShape(obj_id, -1, rgbaColor=color)
