"""
ROS2 Interface for Real Robot Control
Publishes joint commands and subscribes to robot state
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import threading


class RobotController(Node):
    """
    ROS2-based robot controller for real hardware.
    Supports:
    - Joint position/velocity control
    - End-effector pose control
    - Gripper control
    """
    
    def __init__(self):
        super().__init__('robot_controller')
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/effort_joint_trajectory_controller/commands', 10
        )
        
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/equilibrium_pose', 10
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        
        self.ee_state_sub = self.create_subscription(
            PoseStamped, '/franka_ee_state', self.ee_state_callback, 10
        )
        
        # State
        self.current_joint_positions = np.zeros(7)
        self.current_joint_velocities = np.zeros(7)
        self.current_ee_pose = np.zeros(7)  # pos(3) + orn(4)
        self.gripper_width = 0.08
        
        self.lock = threading.Lock()
        
    def joint_state_callback(self, msg: JointState):
        """Update joint state from robot feedback"""
        with self.lock:
            # Map joint names to indices
            for i, name in enumerate(msg.name):
                if 'panda_joint' in name:
                    joint_idx = int(name.split('panda_joint')[1]) - 1
                    if joint_idx < 7:
                        self.current_joint_positions[joint_idx] = msg.position[i]
                        self.current_joint_velocities[joint_idx] = msg.velocity[i]
                        
            # Gripper
            if 'panda_finger_joint1' in msg.name:
                idx = msg.name.index('panda_finger_joint1')
                self.gripper_width = msg.position[idx] + msg.position[idx + 1]
                
    def ee_state_callback(self, msg: PoseStamped):
        """Update EE pose"""
        with self.lock:
            self.current_ee_pose[:3] = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]
            self.current_ee_pose[3:] = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            
    def send_joint_positions(self, positions: np.ndarray, duration: float = 0.1):
        """
        Send joint position command.
        
        Args:
            positions: (7,) joint positions in radians
            duration: command duration in seconds
        """
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.joint_cmd_pub.publish(msg)
        
    def get_robot_state(self) -> dict:
        """Get current robot state"""
        with self.lock:
            return {
                'joint_positions': self.current_joint_positions.copy(),
                'joint_velocities': self.current_joint_velocities.copy(),
                'ee_pose': self.current_ee_pose.copy(),
                'gripper_width': self.gripper_width
            }
            
    def move_to_pose(self, target_pos: np.ndarray, target_orn: np.ndarray = None):
        """
        Move end-effector to target pose.
        
        Args:
            target_pos: (3,) target position
            target_orn: (4,) target orientation quaternion (optional)
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = target_pos[0]
        pose_msg.pose.position.y = target_pos[1]
        pose_msg.pose.position.z = target_pos[2]
        
        if target_orn is not None:
            pose_msg.pose.orientation.x = target_orn[0]
            pose_msg.pose.orientation.y = target_orn[1]
            pose_msg.pose.orientation.z = target_orn[2]
            pose_msg.pose.orientation.w = target_orn[3]
        else:
            # Default: pointing down
            pose_msg.pose.orientation.w = 1.0
            
        self.ee_pose_pub.publish(pose_msg)
        
    def set_gripper(self, open: bool = True):
        """Open or close gripper"""
        if open:
            self.send_joint_positions(np.append(self.current_joint_positions, [0.08]))
        else:
            self.send_joint_positions(np.append(self.current_joint_positions, [0.0]))


class CameraSubscriber(Node):
    """Subscribe to RGB-D camera feed"""
    
    def __init__(self):
        super().__init__('camera_subscriber')
        
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10
        )
        
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        
        self.lock = threading.Lock()
        
    def rgb_callback(self, msg: Image):
        with self.lock:
            # Convert ROS Image to numpy
            # Assuming RGB8 encoding
            self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            
    def depth_callback(self, msg: Image):
        with self.lock:
            # Convert ROS Image to numpy
            # Assuming 32-bit float (meters)
            self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
            
    def camera_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.camera_info = {
                'K': np.array(msg.k).reshape(3, 3),
                'D': np.array(msg.d),
                'P': np.array(msg.p).reshape(3, 4),
                'resolution': (msg.width, msg.height)
            }
            
    def get_latest_frame(self) -> tuple:
        """Get latest RGB-D frame"""
        with self.lock:
            return self.latest_rgb.copy() if self.latest_rgb is not None else None, \
                   self.latest_depth.copy() if self.latest_depth is not None else None


def run_ros2_node(node):
    """Run ROS2 node in separate thread"""
    rclpy.spin(node)


class RobotInterface:
    """
    High-level interface combining ROS2 controllers and camera.
    Used by the policy to get observations and send commands.
    """
    
    def __init__(self, use_ros2: bool = False):
        self.use_ros2 = use_ros2
        
        if use_ros2:
            rclpy.init()
            self.controller = RobotController()
            self.camera = CameraSubscriber()
            self.running = True
            self.thread = threading.Thread(target=run_ros2_node, args=(self.controller,))
            self.thread.start()
        else:
            self.controller = None
            self.camera = None
            
    def get_observation(self) -> dict:
        """Get current observation (RGB, depth, robot state)"""
        if self.use_ros2:
            rgb, depth = self.camera.get_latest_frame()
            state = self.controller.get_robot_state()
            
            return {
                'rgb': rgb,
                'depth': depth,
                'robot_state': np.concatenate([
                    state['joint_positions'],
                    state['joint_velocities']
                ]),
                'gripper_width': state['gripper_width'],
                'ee_pose': state['ee_pose']
            }
        else:
            # Simulation mode - would connect to PyBullet
            raise NotImplementedError("Simulation mode not implemented in RobotInterface")
            
    def send_action(self, action: np.ndarray):
        """
        Send action to robot.
        
        Args:
            action: (4,) — [delta_x, delta_y, delta_z, gripper_cmd]
        """
        if self.use_ros2:
            # Convert delta to joint positions (simplified)
            # Real impl would use IK
            joint_positions = self.controller.current_joint_positions + action[:7] * 0.1
            self.controller.send_joint_positions(joint_positions)
            
            # Gripper
            if action[3] > 0:
                self.controller.set_gripper(open=True)
            else:
                self.controller.set_gripper(open=False)
        else:
            raise NotImplementedError("Simulation mode not implemented")
            
    def shutdown(self):
        """Clean shutdown"""
        if self.use_ros2:
            self.running = False
            self.controller.destroy_node()
            rclpy.shutdown()
