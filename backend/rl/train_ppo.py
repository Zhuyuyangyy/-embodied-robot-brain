"""
PPO Training Script for Robotic Grasping
Multi-GPU training with TensorBoard logging
"""
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.ppo_agent import PPOAgent, PPOMemory, HindsightExperienceReplay
from vision.transformer_encoder import RGBDEncoder
from envs.pybullet_sim import GraspingEnvironment


class GraspingTrainer:
    """
    Trainer for robotic grasping with PPO.
    Handles environment interaction, policy update, and logging.
    """
    
    def __init__(
        self,
        config_path: str = None,
        output_dir: str = "outputs",
        log_dir: str = "runs"
    ):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        # Create environment
        self.env = GraspingEnvironment(
            render=False,
            use_gui=False,
            n_objects=self.config.get('n_objects', 3),
            domain_randomization=True
        )
        
        # Vision encoder
        self.vision_encoder = RGBDEncoder(
            d_model=self.config.get('d_model', 768),
            use_dinov2=self.config.get('use_dinov2', False)
        ).to(self.device)
        
        # PPO Agent
        self.agent = PPOAgent(
            d_visual=self.config.get('d_model', 768),
            d_state=self.config.get('d_model', 768),
            n_actions=4,
            lr=self.config.get('lr', 3e-4),
            gamma=self.config.get('gamma', 0.99),
            lam=self.config.get('lam', 0.95),
            clip_eps=self.config.get('clip_eps', 0.2),
            n_epochs=self.config.get('n_epochs', 10),
            batch_size=self.config.get('batch_size', 64),
            use_her=True
        )
        
        # HER
        self.her = HindsightExperienceReplay(
            strategy=self.config.get('her_strategy', 'future'),
            k=self.config.get('her_k', 4)
        )
        
        # Memory buffer
        self.memory = PPOMemory(batch_size=self.config.get('batch_size', 64))
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
    def _load_config(self, config_path: str = None) -> dict:
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            'd_model': 768,
            'use_dinov2': False,
            'lr': 3e-4,
            'gamma': 0.99,
            'lam': 0.95,
            'clip_eps': 0.2,
            'n_epochs': 10,
            'batch_size': 64,
            'n_objects': 3,
            'n_steps': 2048,
            'n_envs': 4,
            'update_freq': 2048,
            'save_freq': 100,
            'log_freq': 10,
            'her_strategy': 'future',
            'her_k': 4
        }
        
    def collect_rollout(self, n_steps: int = None):
        """
        Collect n_steps of experience from environment.
        """
        n_steps = n_steps or self.config.get('n_steps', 2048)
        
        obs = self.env.reset()
        
        for step in range(n_steps):
            # Encode observation
            rgb = torch.from_numpy(obs['rgb']).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            depth = torch.from_numpy(obs['depth']).unsqueeze(0).unsqueeze(0).float()
            robot_state = torch.from_numpy(obs['robot_state']).float().unsqueeze(0)
            
            with torch.no_grad():
                visual_features, state_features, _ = self.vision_encoder(
                    rgb.to(self.device),
                    depth.to(self.device),
                    robot_state.to(self.device)
                )
            
            # Select action
            action, log_prob, value = self.agent.select_action(
                visual_features.squeeze(0),
                state_features.squeeze(0)
            )
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Store in memory
            self.memory.add(
                visual_features.cpu(),
                state_features.cpu(),
                torch.tensor(action).unsqueeze(0),
                torch.tensor(log_prob).unsqueeze(0).unsqueeze(0),
                torch.tensor(reward).unsqueeze(0),
                torch.tensor(value).unsqueeze(0).unsqueeze(0),
                torch.tensor(done).unsqueeze(0).unsqueeze(0)
            )
            
            obs = next_obs
            
            if done:
                self.episode_rewards.append(sum(self.memory.rewards[-self.env.current_step:]))
                self.episode_lengths.append(self.env.current_step)
                self.success_rates.append(1.0 if info['success'] else 0.0)
                obs = self.env.reset()
                
        # Update stats
        mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        success_rate = np.mean(self.success_rates[-100:]) if self.success_rates else 0
        
        return {
            'mean_reward': mean_reward,
            'mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'success_rate': success_rate
        }
        
    def train(self, n_iterations: int = 1000):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {n_iterations} iterations")
        
        best_success = 0
        
        for iteration in range(n_iterations):
            # Collect experience
            stats = self.collect_rollout()
            
            # Update policy
            update_stats = self.agent.update(self.memory)
            
            # Clear memory
            self.memory.clear()
            
            # Logging
            if iteration % self.config.get('log_freq', 10) == 0:
                logger.info(
                    f"Iter {iteration}/{n_iterations} | "
                    f"Reward: {stats['mean_reward']:.2f} | "
                    f"Success: {stats['success_rate']:.2%} | "
                    f"Loss: {update_stats['total_loss']:.4f}"
                )
                
            # Save checkpoint
            if iteration % self.config.get('save_freq', 100) == 0 and iteration > 0:
                save_path = self.output_dir / f"ppo_grasp_iter{iteration}.pt"
                self.agent.save(str(save_path))
                logger.info(f"Saved checkpoint to {save_path}")
                
                if stats['success_rate'] > best_success:
                    best_success = stats['success_rate']
                    best_path = self.output_dir / "ppo_grasp_best.pt"
                    self.agent.save(str(best_path))
                    logger.info(f"New best! Success rate: {best_success:.2%}")
                    
        logger.info("Training complete!")
        logger.info(f"Final success rate: {stats['success_rate']:.2%}")
        
    def evaluate(self, n_episodes: int = 20) -> dict:
        """
        Evaluate current policy.
        """
        successes = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                rgb = torch.from_numpy(obs['rgb']).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                depth = torch.from_numpy(obs['depth']).unsqueeze(0).unsqueeze(0).float()
                robot_state = torch.from_numpy(obs['robot_state']).float().unsqueeze(0)
                
                with torch.no_grad():
                    visual_features, state_features, _ = self.vision_encoder(
                        rgb.to(self.device),
                        depth.to(self.device),
                        robot_state.to(self.device)
                    )
                    
                action, _, _ = self.agent.select_action(
                    visual_features.squeeze(0),
                    state_features.squeeze(0),
                    deterministic=True
                )
                
                obs, _, done, info = self.env.step(action)
                
            successes.append(1.0 if info['success'] else 0.0)
            
        success_rate = np.mean(successes)
        logger.info(f"Evaluation: {success_rate:.2%} success rate ({n_episodes} episodes)")
        
        return {'success_rate': success_rate}
        
    def save(self, path: str = None):
        """Save full training state"""
        path = path or str(self.output_dir / "ppo_grasp_final.pt")
        
        torch.save({
            'agent': self.agent.actor_critic.state_dict(),
            'vision_encoder': self.vision_encoder.state_dict(),
            'config': self.config
        }, path)
        
        logger.info(f"Saved to {path}")
        
    def load(self, path: str):
        """Load training state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.actor_critic.load_state_dict(checkpoint['agent'])
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        self.config = checkpoint.get('config', self.config)
        
        logger.info(f"Loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description='Train PPO Robotic Grasping')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint')
    args = parser.parse_args()
    
    trainer = GraspingTrainer(output_dir=args.output_dir)
    
    if args.load:
        trainer.load(args.load)
        
    trainer.train(n_iterations=args.iterations)
    
    # Final evaluation
    trainer.evaluate(n_episodes=50)


if __name__ == '__main__':
    main()
