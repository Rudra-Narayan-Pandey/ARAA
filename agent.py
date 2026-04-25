from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class AgentStep:
    action: np.ndarray
    action_mean: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    belief: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


class ARAAAgent(nn.Module):
    def __init__(self, state_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.belief_head = nn.Linear(hidden_dim, state_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((state_dim,), -1.8))

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(obs)
        action_mean = 1.25 * torch.tanh(self.action_head(h))
        belief = torch.tanh(self.belief_head(h))
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        std = torch.exp(self.log_std).clamp(0.05, 0.35)
        return {
            "action_mean": action_mean,
            "belief": belief,
            "confidence": confidence,
            "value": value,
            "std": std,
        }

    def act(self, obs: np.ndarray, deterministic: bool = False, device: str = "cpu") -> AgentStep:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        outputs = self.forward(obs_tensor)
        dist = Normal(outputs["action_mean"], outputs["std"])
        if deterministic:
            action = outputs["action_mean"]
        else:
            action = dist.rsample()
        action = torch.clamp(action, -1.5, 1.5)
        log_prob = dist.log_prob(action).sum(dim=-1).squeeze(0)
        entropy = dist.entropy().sum(dim=-1).squeeze(0)
        return AgentStep(
            action=action.squeeze(0).detach().cpu().numpy().astype(np.float32),
            action_mean=outputs["action_mean"].squeeze(0),
            log_prob=log_prob,
            value=outputs["value"].squeeze(0),
            belief=outputs["belief"].squeeze(0),
            confidence=outputs["confidence"].squeeze(0),
            entropy=entropy,
        )
