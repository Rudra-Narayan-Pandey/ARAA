from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agent import ARAAAgent
from env import ARAAEnv


@dataclass
class EpisodeBatch:
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[float]
    entropies: List[torch.Tensor]
    belief_losses: List[torch.Tensor]
    calibration_losses: List[torch.Tensor]
    action_anchor_losses: List[torch.Tensor]
    visible_rewards: List[float]
    true_rewards: List[float]
    state_norms: List[float]
    belief_errors: List[float]
    epistemic_scores: List[float]
    backdoor_hits: int


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def compute_epistemics(
    belief: torch.Tensor, true_state: np.ndarray, confidence: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    true_tensor = torch.as_tensor(true_state, dtype=torch.float32, device=belief.device)
    belief_error = torch.norm(belief - true_tensor, p=2)
    normalized_error = belief_error / (belief_error + 1.0)
    calibration_target = 1.0 - normalized_error.detach()
    calibration_error = torch.abs(confidence - calibration_target)
    epistemic_score = (1.0 / (1.0 + belief_error)) + (1.0 - calibration_error)
    return belief_error, normalized_error, calibration_error, epistemic_score


def curriculum_value(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + progress * (end - start)


def collect_episode(
    env: ARAAEnv,
    agent: ARAAAgent,
    agent_kind: str,
    device: str,
    attack_probability: float,
    volatility: float,
    seed: int,
) -> EpisodeBatch:
    obs, _ = env.reset_legacy(
        seed=seed,
        options={
            "attack_probability": attack_probability,
            "volatility": volatility,
            "scenario_name": env.scenario_name,
        },
    )
    batch = EpisodeBatch([], [], [], [], [], [], [], [], [], [], [], [], 0)

    done = False
    while not done:
        step = agent.act(obs, deterministic=False, device=device)
        next_obs, visible_reward, done, info = env.step_legacy(step.action)
        belief_error, _, calibration_error, epistemic_score = compute_epistemics(
            step.belief, info["true_state"], step.confidence
        )
        reward_gap = abs(visible_reward - info["true_reward"])
        state_norm = info["state_norm"]
        current_obs_norm = float(np.linalg.norm(obs))

        if agent_kind == "baseline":
            train_reward = float(visible_reward)
            aux_weight = 0.08
            calibration_weight = 0.05
            action_anchor_loss = torch.zeros((), dtype=torch.float32)
        else:
            target_norm = info.get("safe_operating_norm", env.safe_operating_norm())
            low_norm_penalty = max(0.0, env.backdoor_threshold + 0.03 - state_norm)
            high_norm_penalty = max(0.0, state_norm - (target_norm + 0.10))
            norm_band_penalty = abs(state_norm - target_norm)
            backdoor_penalty = 60.0 if info["backdoor_triggered"] else 0.0
            safe_band_bonus = 22.0 if env.backdoor_threshold <= state_norm <= (target_norm + 0.06) else 0.0
            train_reward = (
                float(info["true_reward"])
                - 0.48 * reward_gap
                - 0.75 * norm_band_penalty
                - 18.0 * low_norm_penalty
                - 12.0 * high_norm_penalty
                - backdoor_penalty
                - 0.35 * float(belief_error.detach().cpu().item())
                - 0.20 * float(calibration_error.detach().cpu().item())
                + safe_band_bonus
            )
            aux_weight = 0.22
            calibration_weight = 0.18
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            obs_direction = obs_tensor / (torch.norm(obs_tensor, p=2) + 1e-6)
            if current_obs_norm < env.backdoor_threshold + 0.04:
                desired_scale = 0.90
            elif current_obs_norm > target_norm + 0.12:
                desired_scale = -0.35
            else:
                desired_scale = 0.12
            safe_action = desired_scale * obs_direction
            action_anchor_loss = (
                0.42 * F.mse_loss(step.action_mean, safe_action)
                + 0.02 * step.action_mean.pow(2).mean()
            )

        batch.log_probs.append(step.log_prob)
        batch.values.append(step.value)
        batch.rewards.append(train_reward)
        batch.entropies.append(step.entropy)
        batch.belief_losses.append(aux_weight * F.mse_loss(step.belief, torch.as_tensor(info["true_state"], dtype=torch.float32)))
        batch.calibration_losses.append(calibration_weight * calibration_error.pow(2))
        batch.action_anchor_losses.append(action_anchor_loss)
        batch.visible_rewards.append(float(visible_reward))
        batch.true_rewards.append(float(info["true_reward"]))
        batch.state_norms.append(float(state_norm))
        batch.belief_errors.append(float(belief_error.detach().cpu().item()))
        batch.epistemic_scores.append(float(epistemic_score.detach().cpu().item()))
        batch.backdoor_hits += int(info["backdoor_triggered"])

        env.oversight.log(batch.belief_errors[-1], float(step.confidence.detach().cpu().item()), info)
        obs = next_obs

    return batch


def discounted_returns(rewards: List[float], gamma: float = 0.97) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    return (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)


def optimize_episode(
    agent: ARAAAgent,
    optimizer: torch.optim.Optimizer,
    batch: EpisodeBatch,
) -> Dict[str, float]:
    returns = discounted_returns(batch.rewards)
    values = torch.stack(batch.values)
    log_probs = torch.stack(batch.log_probs)
    entropies = torch.stack(batch.entropies)
    belief_loss = torch.stack(batch.belief_losses).mean()
    calibration_loss = torch.stack(batch.calibration_losses).mean()
    action_anchor_loss = torch.stack(batch.action_anchor_losses).mean()

    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = 0.5 * F.mse_loss(values, returns)
    entropy_bonus = 0.0015 * entropies.mean()

    total_loss = policy_loss + value_loss + belief_loss + calibration_loss + action_anchor_loss - entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(total_loss.detach().cpu().item()),
        "episode_visible_reward": float(np.sum(batch.visible_rewards)),
        "episode_true_reward": float(np.sum(batch.true_rewards)),
        "episode_train_reward": float(np.sum(batch.rewards)),
        "episode_epistemic_score": float(np.mean(batch.epistemic_scores)),
        "episode_state_norm": float(np.mean(batch.state_norms)),
        "episode_belief_error": float(np.mean(batch.belief_errors)),
        "backdoor_hits": int(batch.backdoor_hits),
    }


def train_agent(
    agent_kind: str,
    seed: int,
    episodes: int = 70,
    device: str = "cpu",
    scenario_name: str = "adversarial",
) -> Tuple[ARAAAgent, Dict[str, List[float]]]:
    set_global_seeds(seed)
    env = ARAAEnv.from_preset(scenario_name, seed=seed)
    agent = ARAAAgent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=2e-3)

    history: Dict[str, List[float]] = {
        "visible_reward": [],
        "true_reward": [],
        "train_reward": [],
        "epistemic_score": [],
        "state_norm": [],
        "belief_error": [],
        "attack_probability": [],
        "volatility": [],
        "backdoor_hits": [],
        "loss": [],
    }

    for episode in range(episodes):
        progress = episode / max(episodes - 1, 1)
        attack_probability = curriculum_value(0.03, 0.32, progress)
        volatility = curriculum_value(0.08, 0.28, progress)
        batch = collect_episode(
            env=env,
            agent=agent,
            agent_kind=agent_kind,
            device=device,
            attack_probability=attack_probability,
            volatility=volatility,
            seed=seed + episode,
        )
        metrics = optimize_episode(agent, optimizer, batch)
        for key in ["visible_reward", "true_reward", "train_reward", "epistemic_score", "state_norm", "belief_error", "loss"]:
            history[key].append(metrics[f"episode_{key}"] if key != "loss" else metrics["loss"])
        history["attack_probability"].append(float(attack_probability))
        history["volatility"].append(float(volatility))
        history["backdoor_hits"].append(float(metrics["backdoor_hits"]))

    return agent, history
