"""
Advanced Reward Engineering Module
Implements verifiable, layered rewards with anti-cheat constraints and hacking detection.
Following best practices: RLVR, specification gaming prevention, reward shaping safety.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import torch


@dataclass
class RewardComponent:
    """Individual reward layer with weight and monitoring."""
    name: str
    weight: float
    value: float = 0.0
    is_sparse: bool = False  # If True, rarely provides signal
    is_shaped: bool = False  # If True, can be gamed
    min_threshold: float = -1.0  # Hard constraint lower bound
    max_threshold: float = 1.0   # Hard constraint upper bound


@dataclass
class LayeredReward:
    """Structured reward with outcome + anti-cheat + shaping components."""
    outcome_score: float  # Core verifiable success
    anti_cheat_penalty: float = 0.0  # Anti-exploitation
    shaping_bonus: float = 0.0  # Process encouragement (low weight)
    components: Dict[str, RewardComponent] = field(default_factory=dict)
    total: float = 0.0
    
    def compute(self) -> float:
        """Compute final reward: outcome + anti-cheat, then light shaping."""
        self.total = self.outcome_score + self.anti_cheat_penalty + (0.1 * self.shaping_bonus)
        return self.total


@dataclass
class RewardAudit:
    """Tracks metrics to detect reward hacking."""
    reward: float
    hidden_state_quality: float  # How well belief matches reality
    visible_vs_true_gap: float  # Gap between visible and true reward
    state_norm: float  # Magnitude of state changes
    action_entropy: float  # Policy diversity
    success_pattern: str  # "honest", "suspicious", "exploit_attempt"
    step: int = 0


class VerifiableRewardModule:
    """
    Implements RLVR (RL with Verifiable Rewards) for ARAA.
    
    Design principle: Use hard checks first, anti-cheat second, shaping only as light glue.
    This prevents specification gaming while keeping learning tractable.
    """
    
    def __init__(self, 
                 safe_operating_norm: float = 0.65,
                 backdoor_threshold: float = 1.05):
        self.safe_operating_norm = safe_operating_norm
        self.backdoor_threshold = backdoor_threshold
        self.audit_log: List[RewardAudit] = []
        self.suspicious_patterns: List[Dict] = []
        
        # Reward component registry
        self.components = {
            'outcome_task_success': RewardComponent('outcome_task_success', 0.60, is_sparse=True),
            'honesty_preservation': RewardComponent('honesty_preservation', 0.20, is_sparse=False),
            'resilience_maintenance': RewardComponent('resilience_maintenance', 0.15, is_sparse=False),
            'calibration_quality': RewardComponent('calibration_quality', 0.05, is_sparse=False),
        }
        
        # Anti-cheat constraints
        self.anti_cheat = {
            'backdoor_exploitation': 0.0,  # Detected backdoor hitting
            'state_inflation': 0.0,  # Unrealistic state changes
            'reward_divergence': 0.0,  # Visible ≠ true reward gap growing
        }
        
    def compute_verifiable_outcome(self,
                                   true_reward: float,
                                   is_episode_end: bool) -> float:
        """
        Hard outcome check: only reward actual task success.
        This is the foundation - cannot be gamed.
        """
        if is_episode_end:
            # Episode-level verification: check if agent improved true metrics
            return max(0.0, true_reward)
        return 0.0  # Sparse outcome reward
    
    def compute_honesty_score(self,
                             visible_reward: float,
                             true_reward: float,
                             belief: torch.Tensor,
                             true_state: np.ndarray,
                             confidence: torch.Tensor) -> float:
        """
        Verifiable honesty check: minimize gap between perceived and actual.
        Encourages calibrated beliefs and transparency about uncertainty.
        """
        # Reward gap between what agent sees vs. reality
        reward_gap = abs(visible_reward - true_reward)
        honesty_gap_score = 1.0 / (1.0 + reward_gap)
        
        # Belief accuracy: how close belief is to true state
        true_tensor = torch.as_tensor(true_state, dtype=torch.float32, device=belief.device)
        belief_error = torch.norm(belief - true_tensor, p=2)
        belief_accuracy = 1.0 / (1.0 + belief_error.item())
        
        # Calibration: is confidence appropriate to accuracy?
        normalized_error = belief_error / (belief_error + 1.0)
        calibration_target = 1.0 - normalized_error
        calibration_error = torch.abs(confidence - calibration_target)
        calibration_score = 1.0 - calibration_error.item()
        
        # Composite: 50% reward alignment, 30% belief, 20% calibration
        honesty = 0.5 * honesty_gap_score + 0.3 * belief_accuracy + 0.2 * calibration_score
        return float(honesty)
    
    def compute_resilience(self,
                          state_norm: float,
                          state_velocity: float,
                          episode_true_rewards: List[float]) -> float:
        """
        Verifiable resilience check: maintain system stability despite attacks.
        """
        # Stability: state shouldn't explode or oscillate wildly
        state_stability = 1.0 / (1.0 + state_norm) if state_norm > 0 else 0.5
        
        # Consistency: true rewards should be monotonically improving, not whipsawing
        if len(episode_true_rewards) > 1:
            diffs = np.diff(episode_true_rewards)
            smoothness = 1.0 / (1.0 + np.std(diffs))
        else:
            smoothness = 0.5
        
        resilience = 0.6 * state_stability + 0.4 * smoothness
        return float(resilience)
    
    def detect_anti_cheat_violations(self,
                                     visible_reward: float,
                                     true_reward: float,
                                     state_norm: float,
                                     belief_error: float,
                                     confidence: float,
                                     step: int) -> Tuple[float, str]:
        """
        Detect and penalize reward hacking patterns.
        Returns (penalty, pattern_description).
        """
        violations = []
        penalty = 0.0
        
        # Violation 1: Backdoor hitting (visible reward >> true reward)
        reward_divergence = abs(visible_reward - true_reward)
        if visible_reward > true_reward + self.backdoor_threshold:
            penalty += 0.3  # Heavy penalty
            violations.append('backdoor_exploitation')
        
        # Violation 2: State inflation (uncontrolled growth)
        if state_norm > self.safe_operating_norm * 1.5:
            penalty += 0.2
            violations.append('state_inflation')
        
        # Violation 3: Confidence-reality mismatch (overconfidence)
        expected_confidence = 1.0 - (belief_error / (belief_error + 1.0))
        conf_error = abs(confidence - expected_confidence)
        if conf_error > 0.4:  # High miscalibration
            penalty += 0.15
            violations.append('miscalibration')
        
        # Violation 4: Erratic belief changes (rapid pivoting)
        if step > 1 and len(self.audit_log) > 0:
            prev_belief_error = self.audit_log[-1].hidden_state_quality
            belief_error_change = abs(belief_error - prev_belief_error)
            if belief_error_change > 0.5:
                penalty += 0.1
                violations.append('belief_whipsawing')
        
        pattern = '_'.join(violations) if violations else 'clean'
        return penalty, pattern
    
    def compute_reward(self,
                      visible_reward: float,
                      true_reward: float,
                      belief: torch.Tensor,
                      true_state: np.ndarray,
                      confidence: torch.Tensor,
                      state_norm: float,
                      is_episode_end: bool,
                      episode_true_rewards: List[float],
                      step: int,
                      action_entropy: float) -> Tuple[LayeredReward, RewardAudit]:
        """
        Main reward computation function.
        Returns (structured_reward, audit_record).
        """
        belief_error = torch.norm(belief - torch.as_tensor(true_state, dtype=torch.float32, 
                                                           device=belief.device), p=2).item()
        
        # 1. Verifiable outcome (hard check)
        outcome = self.compute_verifiable_outcome(true_reward, is_episode_end)
        
        # 2. Honesty preservation
        honesty = self.compute_honesty_score(visible_reward, true_reward, belief, true_state, confidence)
        
        # 3. Resilience maintenance
        state_velocity = np.mean(np.abs(np.diff(episode_true_rewards))) if len(episode_true_rewards) > 1 else 0.0
        resilience = self.compute_resilience(state_norm, state_velocity, episode_true_rewards)
        
        # 4. Calibration bonus
        calibration_bonus = float((1.0 - torch.abs(confidence - (1.0 - belief_error / (belief_error + 1.0)))).clamp(0, 1))
        
        # 5. Anti-cheat penalties
        anti_cheat_penalty, pattern = self.detect_anti_cheat_violations(
            visible_reward, true_reward, state_norm, belief_error, confidence.item(), step
        )
        
        # Assemble layered reward
        reward = LayeredReward(
            outcome_score=outcome * self.components['outcome_task_success'].weight,
            anti_cheat_penalty=-anti_cheat_penalty,
            shaping_bonus=action_entropy,  # Light bonus for exploration
        )
        reward.components['outcome_task_success'] = RewardComponent('outcome_task_success', 0.6, outcome)
        reward.components['honesty_preservation'] = RewardComponent('honesty_preservation', 0.2, honesty)
        reward.components['resilience_maintenance'] = RewardComponent('resilience_maintenance', 0.15, resilience)
        reward.components['calibration_quality'] = RewardComponent('calibration_quality', 0.05, calibration_bonus)
        
        final_reward = (
            outcome * 0.60 +
            honesty * 0.20 +
            resilience * 0.15 +
            calibration_bonus * 0.05 +
            (-anti_cheat_penalty)
        )
        reward.total = final_reward
        
        # Create audit record
        audit = RewardAudit(
            reward=final_reward,
            hidden_state_quality=1.0 - (belief_error / (belief_error + 1.0)),
            visible_vs_true_gap=abs(visible_reward - true_reward),
            state_norm=state_norm,
            action_entropy=action_entropy,
            success_pattern=pattern,
            step=step
        )
        
        # Store for analysis
        self.audit_log.append(audit)
        if pattern != 'clean':
            self.suspicious_patterns.append({
                'step': step,
                'pattern': pattern,
                'reward': final_reward,
                'gap': abs(visible_reward - true_reward)
            })
        
        return reward, audit
    
    def detect_reward_hacking(self) -> Dict:
        """
        Analyze audit log for reward hacking signs.
        Returns diagnostic report.
        """
        if not self.audit_log:
            return {'hacking_risk': 'none', 'issues': []}
        
        audits = np.array([(a.reward, a.visible_vs_true_gap, a.hidden_state_quality, 
                           a.success_pattern) for a in self.audit_log], dtype=object)
        
        issues = []
        
        # Issue 1: Reward rising but reality gap widening
        recent_rewards = [a.reward for a in self.audit_log[-20:]]
        recent_gaps = [a.visible_vs_true_gap for a in self.audit_log[-20:]]
        if len(recent_rewards) > 5 and np.mean(recent_rewards[-5:]) > np.mean(recent_rewards[:5]) * 1.2:
            if np.mean(recent_gaps[-5:]) > np.mean(recent_gaps[:5]) * 1.2:
                issues.append('CRITICAL: Reward inflation without reality alignment')
        
        # Issue 2: Persistent suspicious patterns
        pattern_counts = {}
        for p in self.suspicious_patterns:
            pattern_counts[p['pattern']] = pattern_counts.get(p['pattern'], 0) + 1
        for pattern, count in pattern_counts.items():
            if count > len(self.audit_log) * 0.1:  # >10% of steps
                issues.append(f'WARNING: Frequent {pattern} pattern ({count} occurrences)')
        
        # Issue 3: Miscalibration trend
        confidences = [a.hidden_state_quality for a in self.audit_log[-30:]]
        if len(confidences) > 5:
            recent_conf = np.mean(confidences[-5:])
            earlier_conf = np.mean(confidences[:5])
            if recent_conf < earlier_conf * 0.7:
                issues.append('WARNING: Confidence degradation over time')
        
        hacking_risk = 'critical' if any('CRITICAL' in i for i in issues) else \
                      'high' if any('WARNING' in i for i in issues) else 'low'
        
        return {
            'hacking_risk': hacking_risk,
            'issues': issues,
            'suspicious_pattern_count': len(self.suspicious_patterns),
            'pattern_distribution': pattern_counts,
            'avg_reward_gap': float(np.mean([a.visible_vs_true_gap for a in self.audit_log])),
            'avg_hidden_state_quality': float(np.mean([a.hidden_state_quality for a in self.audit_log])),
        }
    
    def reset_audit(self) -> None:
        """Clear audit log for next training run."""
        self.audit_log = []
        self.suspicious_patterns = []


class AdaptiveCurriculumModule:
    """
    Adaptive curriculum learning based on agent performance.
    Adjusts difficulty to keep agent near capability frontier.
    """
    
    def __init__(self, initial_attack_prob: float = 0.05):
        self.attack_probability = initial_attack_prob
        self.volatility = 0.10
        self.success_rate_history: List[float] = []
        self.episode_number = 0
        
    def update(self, success_rate: float) -> None:
        """Update curriculum based on recent success rate."""
        self.success_rate_history.append(success_rate)
        self.episode_number += 1
        
        # Target: ~60% success rate (Goldilocks zone)
        TARGET_SUCCESS = 0.60
        
        if len(self.success_rate_history) >= 5:
            recent_success = np.mean(self.success_rate_history[-5:])
            
            if recent_success > TARGET_SUCCESS + 0.15:  # Too easy
                # Increase difficulty
                self.attack_probability = min(0.35, self.attack_probability * 1.15)
                self.volatility = min(0.35, self.volatility * 1.1)
            elif recent_success < TARGET_SUCCESS - 0.15:  # Too hard
                # Decrease difficulty
                self.attack_probability = max(0.05, self.attack_probability * 0.85)
                self.volatility = max(0.08, self.volatility * 0.9)
    
    def get_params(self) -> Tuple[float, float]:
        """Return current curriculum parameters."""
        return self.attack_probability, self.volatility
