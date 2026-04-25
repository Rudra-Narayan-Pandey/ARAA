"""
Advanced Monitoring and Analysis Module
Real-time tracking of training metrics, reward hacking detection, and diagnostic insights.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import json
from datetime import datetime


@dataclass
class TrainingMetrics:
    """Snapshot of training state."""
    episode: int
    avg_reward: float
    avg_true_reward: float
    avg_visible_reward: float
    reward_gap: float  # |visible - true|
    avg_belief_error: float
    avg_confidence: float
    pass_rate: bool  # % of episodes with positive reward
    success_rate: float
    avg_rollout_length: int
    suspicious_pattern_count: int
    state_norm_mean: float
    state_norm_max: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RealTimeMonitor:
    """
    Monitors training in real-time, tracking all key metrics and detecting anomalies.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size  # Rolling window for averages
        
        # Metric buffers
        self.rewards_buffer: List[float] = []
        self.true_rewards_buffer: List[float] = []
        self.visible_rewards_buffer: List[float] = []
        self.belief_errors_buffer: List[float] = []
        self.confidences_buffer: List[float] = []
        self.state_norms_buffer: List[float] = []
        self.rollout_lengths_buffer: List[int] = []
        self.success_flags_buffer: List[bool] = []
        
        # Pattern tracking
        self.suspicious_patterns: List[str] = []
        self.pattern_counts: Dict[str, int] = {}
        
        # Episode history
        self.history: List[TrainingMetrics] = []
        self.episode_counter = 0
        
        # Thresholds for anomaly detection
        self.thresholds = {
            'reward_gap_critical': 1.5,  # When visible >> true
            'belief_error_high': 0.8,
            'confidence_miscalibration': 0.3,
            'state_norm_unsafe': 1.0,
        }
    
    def record_step(self,
                   reward: float,
                   true_reward: float,
                   visible_reward: float,
                   belief_error: float,
                   confidence: float,
                   state_norm: float,
                   rollout_length: int,
                   pattern: str = 'clean') -> None:
        """Record metrics for a single step/episode."""
        self.rewards_buffer.append(reward)
        self.true_rewards_buffer.append(true_reward)
        self.visible_rewards_buffer.append(visible_reward)
        self.belief_errors_buffer.append(belief_error)
        self.confidences_buffer.append(confidence)
        self.state_norms_buffer.append(state_norm)
        self.rollout_lengths_buffer.append(rollout_length)
        self.success_flags_buffer.append(reward > 0)
        
        # Track pattern
        self.suspicious_patterns.append(pattern)
        self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
        
        # Keep only recent data
        if len(self.rewards_buffer) > self.window_size:
            self.rewards_buffer.pop(0)
            self.true_rewards_buffer.pop(0)
            self.visible_rewards_buffer.pop(0)
            self.belief_errors_buffer.pop(0)
            self.confidences_buffer.pop(0)
            self.state_norms_buffer.pop(0)
            self.rollout_lengths_buffer.pop(0)
            self.success_flags_buffer.pop(0)
    
    def get_snapshot(self, episode: int) -> TrainingMetrics:
        """Get current metrics snapshot."""
        if not self.rewards_buffer:
            return TrainingMetrics(
                episode=episode, avg_reward=0, avg_true_reward=0, avg_visible_reward=0,
                reward_gap=0, avg_belief_error=0, avg_confidence=0, pass_rate=0,
                success_rate=0, avg_rollout_length=0, suspicious_pattern_count=0,
                state_norm_mean=0, state_norm_max=0
            )
        
        suspicious_count = sum(1 for p in self.suspicious_patterns if p != 'clean')
        
        return TrainingMetrics(
            episode=episode,
            avg_reward=float(np.mean(self.rewards_buffer)),
            avg_true_reward=float(np.mean(self.true_rewards_buffer)),
            avg_visible_reward=float(np.mean(self.visible_rewards_buffer)),
            reward_gap=float(np.mean(np.abs(np.array(self.visible_rewards_buffer) - 
                                           np.array(self.true_rewards_buffer)))),
            avg_belief_error=float(np.mean(self.belief_errors_buffer)),
            avg_confidence=float(np.mean(self.confidences_buffer)),
            pass_rate=float(np.mean(self.success_flags_buffer)),
            success_rate=float(np.mean(self.success_flags_buffer)),
            avg_rollout_length=int(np.mean(self.rollout_lengths_buffer)),
            suspicious_pattern_count=suspicious_count,
            state_norm_mean=float(np.mean(self.state_norms_buffer)),
            state_norm_max=float(np.max(self.state_norms_buffer)) if self.state_norms_buffer else 0
        )
    
    def detect_anomalies(self) -> Dict[str, List[str]]:
        """Detect training anomalies and return diagnostic."""
        anomalies = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        if len(self.rewards_buffer) < 5:
            return anomalies
        
        # Anomaly 1: Reward gap too wide (backdoor hitting)
        reward_gap = np.mean(np.abs(np.array(self.visible_rewards_buffer) - 
                                   np.array(self.true_rewards_buffer)))
        if reward_gap > self.thresholds['reward_gap_critical']:
            anomalies['critical'].append(
                f"BACKDOOR RISK: Reward gap {reward_gap:.3f} exceeds threshold "
                f"{self.thresholds['reward_gap_critical']}"
            )
        
        # Anomaly 2: Belief errors growing
        recent_belief = np.mean(self.belief_errors_buffer[-5:])
        earlier_belief = np.mean(self.belief_errors_buffer[:5])
        if recent_belief > earlier_belief * 1.3:
            anomalies['warning'].append(
                f"Belief accuracy degrading: {earlier_belief:.3f} → {recent_belief:.3f}"
            )
        
        # Anomaly 3: Miscalibration (overconfidence/underconfidence)
        avg_confidence = np.mean(self.confidences_buffer)
        avg_accuracy = 1.0 - np.mean(self.belief_errors_buffer)
        calibration_error = abs(avg_confidence - avg_accuracy)
        if calibration_error > self.thresholds['confidence_miscalibration']:
            anomalies['warning'].append(
                f"Miscalibration: Confidence {avg_confidence:.3f} vs Accuracy {avg_accuracy:.3f}"
            )
        
        # Anomaly 4: State explosion
        max_state_norm = np.max(self.state_norms_buffer)
        if max_state_norm > self.thresholds['state_norm_unsafe']:
            anomalies['critical'].append(
                f"STATE EXPLOSION: Max norm {max_state_norm:.3f} exceeds safe limit "
                f"{self.thresholds['state_norm_unsafe']}"
            )
        
        # Anomaly 5: Success rate too low (stuck learning)
        success_rate = np.mean(self.success_flags_buffer)
        if success_rate < 0.1:
            anomalies['critical'].append(
                f"LEARNING STALLED: Success rate {success_rate:.1%} - reward too sparse"
            )
        
        # Anomaly 6: Suspicious pattern frequency
        pattern_freq = self.pattern_counts.get('backdoor_exploitation', 0) / max(1, len(self.suspicious_patterns))
        if pattern_freq > 0.15:
            anomalies['critical'].append(
                f"PATTERN ALERT: {pattern_freq:.1%} of steps show backdoor exploitation"
            )
        
        # Info: Rollout efficiency
        avg_length = np.mean(self.rollout_lengths_buffer)
        anomalies['info'].append(
            f"Avg rollout length: {avg_length:.0f}, Avg reward gap: {reward_gap:.3f}"
        )
        
        return anomalies
    
    def record_episode(self, episode: int) -> TrainingMetrics:
        """Record episode and return metrics."""
        snapshot = self.get_snapshot(episode)
        self.history.append(snapshot)
        self.episode_counter = episode
        return snapshot
    
    def get_history(self) -> List[TrainingMetrics]:
        """Get full training history."""
        return self.history
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON."""
        data = {
            'episode': self.episode_counter,
            'timestamp': datetime.now().isoformat(),
            'metrics': [
                {
                    'episode': m.episode,
                    'avg_reward': m.avg_reward,
                    'avg_true_reward': m.avg_true_reward,
                    'reward_gap': m.reward_gap,
                    'success_rate': m.success_rate,
                    'avg_belief_error': m.avg_belief_error,
                    'suspicious_patterns': m.suspicious_pattern_count,
                } for m in self.history
            ],
            'pattern_distribution': self.pattern_counts,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class EnvironmentValidator:
    """
    Validates environment before training.
    Implements the "best debugging order" from RL best practices.
    """
    
    def __init__(self, env):
        self.env = env
        self.validation_results = {}
    
    def step_1_manual_environment_check(self) -> Dict[str, bool]:
        """Debug 1: Manual environment sanity checks."""
        results = {}
        
        # Check 1: Reset works
        try:
            obs, info = self.env.reset_legacy(seed=0, options={})
            results['reset_works'] = True
        except Exception as e:
            results['reset_works'] = False
            print(f"❌ Reset failed: {e}")
        
        # Check 2: Step works
        try:
            action = np.zeros(self.env.state_dim, dtype=np.float32)
            obs, reward, done, info = self.env.step_legacy(action)
            results['step_works'] = True
        except Exception as e:
            results['step_works'] = False
            print(f"❌ Step failed: {e}")
        
        # Check 3: Required fields in info
        required_fields = ['true_reward', 'visible_reward', 'state_norm', 'attacked']
        missing = [f for f in required_fields if f not in info]
        results['required_fields'] = len(missing) == 0
        if missing:
            print(f"❌ Missing fields: {missing}")
        
        return results
    
    def step_2_verifier_validation(self) -> Dict[str, bool]:
        """Debug 2: Check verifier consistency."""
        results = {}
        
        # Run 10 episodes and check reward consistency
        reward_consistency = []
        for i in range(10):
            obs, _ = self.env.reset_legacy(seed=100 + i, options={})
            episode_rewards = []
            done = False
            steps = 0
            while not done and steps < 20:
                action = np.random.randn(self.env.state_dim).astype(np.float32) * 0.1
                obs, reward, done, info = self.env.step_legacy(action)
                episode_rewards.append(info.get('true_reward', 0))
                steps += 1
            
            # Reward should be bounded
            if episode_rewards and max(episode_rewards) < 100:
                reward_consistency.append(True)
        
        results['verifier_stable'] = len(reward_consistency) == 10
        return results
    
    def step_3_baseline_policy_check(self) -> Dict[str, bool]:
        """Debug 3: Run frozen (random) policy."""
        results = {}
        
        baseline_rewards = []
        for i in range(5):
            obs, _ = self.env.reset_legacy(seed=200 + i, options={})
            episode_reward = 0
            done = False
            while not done:
                action = np.random.randn(self.env.state_dim).astype(np.float32) * 0.05
                obs, reward, done, info = self.env.step_legacy(action)
                episode_reward += info.get('true_reward', 0)
            baseline_rewards.append(episode_reward)
        
        avg_baseline = np.mean(baseline_rewards)
        results['baseline_nonzero'] = avg_baseline != 0  # Some signal exists
        results['avg_baseline_reward'] = float(avg_baseline)
        
        return results
    
    def validate_all(self) -> Dict:
        """Run full validation suite."""
        print("\n🔍 ENVIRONMENT VALIDATION SUITE")
        print("=" * 50)
        
        print("\n1️⃣  Manual Environment Checks...")
        step1 = self.step_1_manual_environment_check()
        for k, v in step1.items():
            print(f"  {'✅' if v else '❌'} {k}")
        
        print("\n2️⃣  Verifier Validation...")
        step2 = self.step_2_verifier_validation()
        for k, v in step2.items():
            print(f"  {'✅' if v else '❌'} {k}")
        
        print("\n3️⃣  Baseline Policy Check...")
        step3 = self.step_3_baseline_policy_check()
        for k, v in step3.items():
            if isinstance(v, bool):
                print(f"  {'✅' if v else '❌'} {k}")
            else:
                print(f"  ℹ️  {k}: {v:.4f}")
        
        all_pass = all(step1.values()) and all(step2.values()) and step3['baseline_nonzero']
        print(f"\n{'✅ READY FOR TRAINING' if all_pass else '❌ ISSUES DETECTED'}")
        
        return {
            'manual_checks': step1,
            'verifier': step2,
            'baseline': step3,
            'ready': all_pass
        }
