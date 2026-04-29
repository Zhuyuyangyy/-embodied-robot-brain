"""
PPO Agent for Robotic Grasping
Proximal Policy Optimization with custom environment integration
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Optional, Dict
import yaml
from pathlib import Path


class ActorCritic(nn.Module):
    """
    Unified Actor-Critic network for PPO.
    Actor outputs action mean + log_std.
    Critic outputs state value.
    """
    
    def __init__(
        self,
        d_visual: int = 768,
        d_state: int = 768,
        n_actions: int = 4,
        hidden_dim: int = 512,
        action_std: float = 0.5
    ):
        super().__init__()
        
        self.action_std = action_std
        
        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_visual + d_state, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        
        # Log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, visual_features, state_features):
        """
        Args:
            visual_features: (B, d_visual)
            state_features: (B, d_state)
        Returns:
            values: (B, 1)
        """
        combined = torch.cat([visual_features, state_features], dim=-1)
        shared = self.shared_encoder(combined)
        values = self.critic(shared)
        return values
    
    def get_action(self, visual_features, state_features):
        """
        Args:
            visual_features: (B, d_visual)
            state_features: (B, d_state)
        Returns:
            action: (B, n_actions)
            log_prob: (B, 1)
            value: (B, 1)
            std: (B, n_actions)
        """
        combined = torch.cat([visual_features, state_features], dim=-1)
        shared = self.shared_encoder(combined)
        
        # Actor
        action_mean = self.actor(shared)
        std = torch.exp(self.log_std)
        
        # Sample action from Gaussian
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Value
        value = self.critic(shared)
        
        return action, log_prob, value, std
    
    def evaluate(self, visual_features, state_features, action):
        """
        Evaluate actions (for PPO update).
        """
        combined = torch.cat([visual_features, state_features], dim=-1)
        shared = self.shared_encoder(combined)
        
        action_mean = self.actor(shared)
        std = torch.exp(self.log_std)
        
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        value = self.critic(shared)
        
        return log_prob, value, entropy


class PPOMemory:
    """
    Replay buffer for PPO.
    Stores trajectories for batch update.
    """
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        
        self.states_visual = []
        self.states_robot = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.ptr = 0
        self.size = 0
        
    def add(self, state_visual, state_robot, action, log_prob, reward, value, done):
        """Add a transition to memory"""
        self.states_visual.append(state_visual)
        self.states_robot.append(state_robot)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        self.size += 1
        
    def get_batch(self):
        """Get all stored transitions as batched tensors"""
        B = len(self.states_visual)
        
        states_v = torch.stack(self.states_visual).squeeze(1)  # (T, B, d_visual)
        states_r = torch.stack(self.states_robot).squeeze(1)   # (T, B, d_state)
        actions = torch.stack(self.actions).squeeze(1)          # (T, B, n_actions)
        old_log_probs = torch.stack(self.log_probs).squeeze(-1)  # (T, B)
        rewards = torch.tensor(self.rewards).unsqueeze(-1)     # (T, B, 1)
        values = torch.stack(self.values).squeeze(-1)          # (T, B)
        dones = torch.tensor(self.dones, dtype=torch.float32).unsqueeze(-1)  # (T, B, 1)
        
        return states_v, states_r, actions, old_log_probs, rewards, values, dones
        
    def clear(self):
        """Reset memory"""
        self.states_visual = []
        self.states_robot = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent for robotic grasping.
    
    Key features:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Hindsight Experience Replay (HER) for sparse rewards
    - Domain randomization for sim-to-real transfer
    """
    
    def __init__(
        self,
        d_visual: int = 768,
        d_state: int = 768,
        n_actions: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_her: bool = True,
        her_strategy: str = "future",
        her_k: int = 4
    ):
        # Networks
        self.actor_critic = ActorCritic(d_visual, d_state, n_actions)
        self.actor_critic_old = ActorCritic(d_visual, d_state, n_actions)
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # HER
        self.use_her = use_her
        self.her_strategy = her_strategy
        self.her_k = her_k
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_critic.to(self.device)
        self.actor_critic_old.to(self.device)
        
    def select_action(self, state_visual, state_robot, deterministic: bool = False):
        """
        Select action given state.
        
        Args:
            state_visual: (1, d_visual) or (d_visual,)
            state_robot: (1, d_state) or (d_state,)
            deterministic: if True, return mean action
        Returns:
            action: (n_actions,)
            log_prob: scalar
            value: scalar
        """
        self.actor_critic.eval()
        
        # Ensure batch dimension
        if state_visual.dim() == 1:
            state_visual = state_visual.unsqueeze(0)
        if state_robot.dim() == 1:
            state_robot = state_robot.unsqueeze(0)
            
        state_v = state_visual.to(self.device)
        state_r = state_robot.to(self.device)
        
        if deterministic:
            with torch.no_grad():
                combined = torch.cat([state_v, state_r], dim=-1)
                shared = self.actor_critic.shared_encoder(combined)
                action_mean = self.actor_critic.actor(shared)
                action = action_mean
                log_prob = None
                value = self.actor_critic.critic(shared)
        else:
            action, log_prob, value, std = self.actor_critic.get_action(state_v, state_r)
            
        return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """
        Update policy using collected experience.
        
        Args:
            memory: PPOMemory with stored trajectories
        Returns:
            metrics: dict of training metrics
        """
        self.actor_critic.train()
        
        # Get batched data
        states_v, states_r, actions, old_log_probs, rewards, values, dones = memory.get_batch()
        
        B, T = states_v.shape[0], states_v.shape[1]
        
        # Flatten: (T, B, ...) → (T*B, ...)
        states_v = states_v.reshape(T * B, -1).to(self.device)
        states_r = states_r.reshape(T * B, -1).to(self.device)
        actions = actions.reshape(T * B, -1).to(self.device)
        old_log_probs = old_log_probs.reshape(T * B, 1).to(self.device)
        rewards = rewards.reshape(T * B, 1).to(self.device)
        values = values.reshape(T * B, 1).to(self.device)
        dones = dones.reshape(T * B, 1).to(self.device)
        
        # Compute advantages (GAE)
        advantages = self._compute_gae(rewards, values, dones)
        
        # Target values for critic
        targets = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        for epoch in range(self.n_epochs):
            # Reshuffle indices
            indices = torch.randperm(T * B)
            
            for start in range(0, T * B, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_v = states_v[batch_idx]
                batch_r = states_r[batch_idx]
                batch_a = actions[batch_idx]
                batch_old_log_prob = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_targets = targets[batch_idx]
                
                # Evaluate actions
                log_probs, values_pred, entropy = self.actor_critic.evaluate(
                    batch_v, batch_r, batch_a
                )
                
                # PPO policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch_old_log_prob)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values_pred, batch_targets)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.vf_coef * value_loss +
                    self.ent_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy_loss.item()
                total_loss += loss.item()
        
        # Update old network
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        
        n_batches = (T * B // self.batch_size) * self.n_epochs
        
        return {
            'total_loss': total_loss / n_batches,
            'policy_loss': policy_loss_sum / n_batches,
            'value_loss': value_loss_sum / n_batches,
            'entropy': entropy_sum / n_batches,
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
    def _compute_gae(self, rewards, values, dones, next_value=None):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: (T*B, 1)
            values: (T*B, 1)
            dones: (T*B, 1)
            next_value: terminal value
        Returns:
            advantages: (T*B, 1)
        """
        if next_value is None:
            next_value = torch.zeros_like(values[-1])
            
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages
        
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'actor_critic_old': self.actor_critic_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.actor_critic_old.load_state_dict(checkpoint['actor_critic_old'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER) for sparse reward tasks.
    Relabels failed trajectories as successful by changing the goal.
    """
    
    def __init__(self, strategy: str = "future", k: int = 4, n_goals: int = 1):
        self.strategy = strategy
        self.k = k  # Number of goals to relabel per episode
        self.n_goals = n_goals
        
    def relabel_episode(self, episode_data: Dict):
        """
        Relabel episode with hindsight goals.
        
        Args:
            episode_data: dict with keys 'observations', 'actions', 'rewards', etc.
        Returns:
            List of relabeled episodes
        """
        obs = episode_data['observations']
        achieved_goals = episode_data.get('achieved_goals', None)
        desired_goal = episode_data.get('desired_goal', None)
        
        if achieved_goals is None or desired_goal is None:
            return [episode_data]
            
        episodes = [episode_data]  # Original episode
        
        # Relabel failed experiences
        episode_length = len(obs)
        
        for _ in range(self.k):
            # Strategy: sample future state as new goal
            if self.strategy == "future":
                future_idx = np.random.randint(0, episode_length)
                new_goal = achieved_goals[future_idx]
            else:
                # Random goal
                new_goal = achieved_goals[np.random.randint(0, episode_length)]
                
            # Relabel rewards and goals
            relabeled = self._relabel_with_goal(episode_data, new_goal)
            episodes.append(relabeled)
            
        return episodes
        
    def _relabel_with_goal(self, episode_data, new_goal):
        """Relabel episode with new goal"""
        relabeled = {k: v.copy() if isinstance(v, list) else v for k, v in episode_data.items()}
        
        # Compute new rewards based on new goal
        achieved_goals = episode_data['achieved_goals']
        new_rewards = []
        
        for ag in achieved_goals:
            distance = np.linalg.norm(ag - new_goal)
            reward = -distance  # Sparse: closer is better
            new_rewards.append(reward)
            
        relabeled['rewards'] = new_rewards
        relabeled['desired_goal'] = new_goal
        relabeled['is_relabeled'] = True
        
        return relabeled
