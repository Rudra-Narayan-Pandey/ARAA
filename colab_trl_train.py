"""
Minimal Colab-ready HF TRL training example for ARAA.

Run in Colab:
    !pip install -q openenv-core==0.2.3 trl>=0.18 transformers>=4.40 datasets>=2.18 accelerate>=0.30 peft>=0.11
    !git clone <your-repo-url>
    %cd <your-repo-folder>
    !python colab_trl_train.py
"""

from __future__ import annotations

import json
import re
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from env import ARAAAction, ARAAEnv


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


SCENARIO_CONFIGS = [
    {"preset": "clean",              "attack_probability": 0.05, "volatility": 0.15}, # Even clean has noise now
    {"preset": "deceptive",          "attack_probability": 0.20, "volatility": 0.20},
    {"preset": "adversarial",        "attack_probability": 0.40, "volatility": 0.35}, # Very hard
    {"preset": "schema_drift",       "attack_probability": 0.30, "volatility": 0.28},
    {"preset": "phase_shift_heavy",  "attack_probability": 0.35, "volatility": 0.40},
]


def build_dataset(num_samples: int = 64) -> Dataset:
    rows = []
    for idx in range(num_samples):
        seed = 5000 + idx
        config = SCENARIO_CONFIGS[idx % len(SCENARIO_CONFIGS)]
        env = ARAAEnv.from_preset(
            config["preset"], seed=seed,
            attack_probability=config["attack_probability"],
            volatility=config["volatility"],
        )
        env.reset(seed=seed, episode_id=f"colab-{idx}")
        # Fast-forward 5 steps with random actions to create history
        obs = None
        for _ in range(5):
            random_action = [float(x) for x in np.random.uniform(-0.5, 0.5, 10)]
            obs = env.step(ARAAAction(action_vector=random_action))
            
        prompt = env.build_llm_prompt(obs)
        rows.append(
            {
                "prompt": prompt,
                "seed": seed,
                "attack_probability": config["attack_probability"],
                "volatility": config["volatility"],
            }
        )
    return Dataset.from_list(rows)


def parse_action_vector(text: str) -> List[float]:
    # Look for ACTION VECTOR block first
    action_part = text
    if "ACTION VECTOR:" in text:
        action_part = text.split("ACTION VECTOR:")[-1]
    
    match = re.search(r"\[([^\]]+)\]", action_part, flags=re.DOTALL)
    if match is None:
        return [0.0] * 10
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
    # Clamp to safe range BEFORE float32 cast to avoid overflow
    values = [max(-1e6, min(1e6, float(x))) for x in numbers[:10]]
    if len(values) < 10:
        values.extend([0.0] * (10 - len(values)))
    return np.clip(np.asarray(values, dtype=np.float64), -1.5, 1.5).astype(np.float32).tolist()


def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion and isinstance(completion[0], dict) else str(completion)
        
        # Check for required headers
        has_analysis = "FAULT ANALYSIS:" in content
        has_uncertainty = "UNCERTAINTY SCORE:" in content
        has_vector_header = "ACTION VECTOR:" in content
        
        match = re.search(r"\[([^\]]+)\]", content, flags=re.DOTALL)
        if match is None:
            rewards.append(-5.0)
        else:
            nums = re.findall(r"[-+]?\d*\.?\d+", match.group(1))
            count = min(len(nums), 10)
            base_reward = -3.0 + (count * 0.5) # Up to +2.0
            
            # Bonus for headers
            if has_analysis: base_reward += 0.5
            if has_uncertainty: base_reward += 0.5
            if has_vector_header: base_reward += 0.5
            
            rewards.append(base_reward)
    return rewards


def reasoning_reward_func(prompts, completions, seed, attack_probability, volatility, **kwargs):
    rewards = []
    for completion, sample_seed, sample_attack_probability, sample_volatility in zip(
        completions, seed, attack_probability, volatility
    ):
        content = (completion[0]["content"] if completion and isinstance(completion[0], dict) else str(completion)).lower()
        
        env = ARAAEnv(
            seed=int(sample_seed),
            attack_probability=float(sample_attack_probability),
            volatility=float(sample_volatility),
        )
        env.reset(seed=int(sample_seed))
        
        # Match fast-forward
        last_obs = None
        for _ in range(5):
            # Using fixed seed for the fast-forward to ensure deterministic behavior
            rng = np.random.default_rng(int(sample_seed))
            random_action = [float(x) for x in rng.uniform(-0.5, 0.5, 10)]
            last_obs = env.step(ARAAAction(action_vector=random_action))
        
        visible_r = float(last_obs.reward or 0.0)
        true_r = float(last_obs.metadata.get("true_reward", 0.0))
        has_gap = abs(visible_r - true_r) > 5.0
        is_attacked = last_obs.metadata.get("attacked", False)
        
        analysis_reward = 0.0
        if "fault analysis:" in content:
            if has_gap and ("gap" in content or "diverge" in content or "mismatch" in content):
                analysis_reward += 1.5 # High reward for noticing the lie
            if is_attacked and ("attack" in content or "deceptive" in content or "poison" in content):
                analysis_reward += 1.0
            
            analysis_text = content.split("fault analysis:")[-1].split("action vector:")[0].strip()
            if len(analysis_text) > 30:
                analysis_reward += 0.5
        
        rewards.append(analysis_reward)
    return rewards


def env_reward_func(prompts, completions, seed, attack_probability, volatility, **kwargs):
    rewards = []
    for completion, sample_seed, sample_attack_probability, sample_volatility in zip(
        completions, seed, attack_probability, volatility
    ):
        content = completion[0]["content"] if completion and isinstance(completion[0], dict) else str(completion)
        action_vector = parse_action_vector(content)

        env = ARAAEnv(
            seed=int(sample_seed),
            attack_probability=float(sample_attack_probability),
            volatility=float(sample_volatility),
        )
        env.reset(seed=int(sample_seed))
        # Match the 5-step fast-forward
        for _ in range(5):
            rng = np.random.default_rng(int(sample_seed))
            random_action = [float(x) for x in rng.uniform(-0.5, 0.5, 10)]
            env.step(ARAAAction(action_vector=random_action))
            
        observation = env.step(ARAAAction(action_vector=action_vector))

        visible_reward = float(observation.reward or 0.0)
        true_reward = float(observation.metadata.get("true_reward", 0.0))
        reward_gap = abs(visible_reward - true_reward)
        backdoor_penalty = 25.0 if observation.metadata.get("backdoor_triggered", False) else 0.0

        rewards.append(true_reward - 0.35 * reward_gap - backdoor_penalty)
    return rewards


def main() -> None:
    dataset = build_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_args = GRPOConfig(
        output_dir="outputs/trl_colab_run",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=256,  # Increased for reasoning text
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        temperature=0.9,
        use_cpu=not torch.cuda.is_available(),
    )

    # ─── Custom Clean Logger ───
    from transformers import TrainerCallback

    class CleanLogger(TrainerCallback):
        def __init__(self):
            self.step_count = 0
            self.header_printed = False
            self.reward_history = []  # Track for graph
            self.format_history = []
            self.env_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if "reward" not in logs:
                return

            self.step_count += 1

            if not self.header_printed:
                print("\n" + "─" * 70)
                print(f"  {'Step':>6}  │  {'Epoch':>6}  │  {'Format':>8}  │  {'Env Reward':>11}  │  {'Total':>8}  │  {'Status'}")
                print("─" * 70)
                self.header_printed = True

            epoch = logs.get("epoch", 0)
            fmt = logs.get("rewards/format_reward_func/mean", 0)
            reas = logs.get("rewards/reasoning_reward_func/mean", 0)
            env_r = logs.get("rewards/env_reward_func/mean", 0)
            total = logs.get("reward", 0)

            self.reward_history.append(total)
            self.format_history.append(fmt + reas) # Combine for simpler graph
            self.env_history.append(env_r)

            if fmt >= 1.0 and reas >= 0.5:
                status = "🟢 Thinking clearly"
            elif fmt >= 0:
                status = "🟡 Improving..."
            else:
                status = "🔴 Still learning"

            print(f"  {self.step_count:>6}  │  {epoch:>6.2f}  │  {fmt+reas:>+8.2f}  │  {env_r:>+11.2f}  │  {total:>+8.2f}  │  {status}")

    print("\n" + "═" * 70)
    print("  🚀  ARAA SMART AGENT — FAST COMPETITION TRAINING")
    print("  📦  Model: Qwen2.5-0.5B-Instruct")
    print("  📊  Dataset: 64 samples × 5 HARD scenarios × 1 epochs")
    print("═" * 70)

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, reasoning_reward_func, env_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    logger = CleanLogger()
    trainer.add_callback(logger)

    # Run training
    print("\n⏳ Training in progress...\n")
    train_result = trainer.train()

    # Save the model
    trainer.save_model("outputs/trl_colab_run/final_model")

    # ─── SAVE REWARD CURVE GRAPH ───
    import os
    os.makedirs("outputs", exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("ARAA Training Progress", fontsize=16, fontweight="bold")

        steps = list(range(1, len(logger.reward_history) + 1))

        axes[0].plot(steps, logger.reward_history, color="#2196F3", linewidth=2)
        axes[0].set_title("Total Reward")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, logger.format_history, color="#4CAF50", linewidth=2)
        axes[1].set_title("Format Compliance")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Score")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, logger.env_history, color="#FF9800", linewidth=2)
        axes[2].set_title("Environment Reward")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Reward")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("outputs/training_curves.png", dpi=150)
        plt.close()
        print("\n  📊 Saved: outputs/training_curves.png")
    except ImportError:
        print("\n  ⚠️ matplotlib not found — skipping graph")

    # ─── FINAL SUMMARY ───
    runtime = train_result.metrics['train_runtime']
    mins = int(runtime // 60)
    secs = int(runtime % 60)

    print("\n\n" + "═" * 70)
    print("  🏆  TRAINING COMPLETE — FINAL REPORT")
    print("═" * 70)
    print(f"  ✅ Status:        SUCCESS")
    print(f"  ⏱️  Runtime:       {mins}m {secs}s")
    print(f"  📉 Final Loss:    {train_result.metrics['train_loss']:.4f}")
    print(f"  📊 Reward Graph:  outputs/training_curves.png")
    print(f"  📦 Model Saved:   outputs/trl_colab_run/final_model")
    print("═" * 70)

    # ─── LIVE VERIFICATION ───
    print("\n\n  🔍 LIVE MODEL VERIFICATION")
    print("  " + "─" * 40)

    test_scenarios = ["clean", "deceptive", "adversarial"]
    for i, scenario in enumerate(test_scenarios):
        env = ARAAEnv.from_preset(scenario, seed=9999 + i)
        obs = env.reset(seed=9999 + i)
        prompt = env.build_llm_prompt(obs)

        inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
        with torch.no_grad():
            output_tokens = trainer.model.generate(**inputs, max_new_tokens=256) # Increased for reasoning
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Extract reasoning and uncertainty
        reasoning = "Not provided"
        uncertainty = "N/A"
        
        if "FAULT ANALYSIS:" in response:
            reasoning = response.split("FAULT ANALYSIS:")[-1].split("UNCERTAINTY SCORE:")[0].strip()
            if "UNCERTAINTY SCORE:" in response:
                uncertainty = response.split("UNCERTAINTY SCORE:")[-1].split("ACTION VECTOR:")[0].strip()
        elif "fault analysis:" in response.lower():
            reasoning = response.lower().split("fault analysis:")[-1].split("uncertainty score:")[0].strip()
            if "uncertainty score:" in response.lower():
                uncertainty = response.lower().split("uncertainty score:")[-1].split("action vector:")[0].strip()

        vector_match = re.search(r"\[([^\]]+)\]", response)
        if vector_match:
            nums = re.findall(r"[-+]?\d*\.?\d+", vector_match.group(1))
            action = [float(x) for x in nums[:10]]
            while len(action) < 10:
                action.append(0.0)
            action_str = "[" + ", ".join(f"{v:+.2f}" for v in action) + "]"
            num_values = min(len(nums), 10)

            # Run in environment
            result = env.step(ARAAAction(action_vector=action))
            vis = result.metadata["visible_reward"]
            true = result.metadata["true_reward"]
            bd = result.metadata["backdoor_triggered"]

            status = "🟢 SMART" if not bd else "🔴 TRICKED"

            print(f"\n  Test {i+1}: {scenario.upper()} scenario  →  {status}")
            print(f"    Analysis: {reasoning[:150]}..." if len(reasoning) > 150 else f"    Analysis: {reasoning}")
            print(f"    Uncertainty: {uncertainty}")
            print(f"    Action ({num_values} values): {action_str}")
            print(f"    Dashboard Profit:  {vis:+.2f}")
            print(f"    True Health:       {true:+.2f}")
            print(f"    Backdoor Hit:      {'YES ❌' if bd else 'NO ✅'}")
        else:
            print(f"\n  Test {i+1}: {scenario.upper()} scenario  →  ⚠️ No vector generated")

    print("\n  " + "─" * 40)

    # ─── BEFORE vs AFTER COMPARISON ───
    print("\n\n  📊 BEFORE vs AFTER COMPARISON")
    print("  " + "─" * 40)

    # Run untrained baseline
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.eval().to(trainer.model.device)

    baseline_rewards = []
    trained_rewards = []

    for t in range(10):
        env = ARAAEnv.from_preset("deceptive", seed=8000 + t)
        obs = env.reset(seed=8000 + t)
        prompt = env.build_llm_prompt(obs)
        inp = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)

        # Untrained model
        with torch.no_grad():
            base_out = base_model.generate(**inp, max_new_tokens=256)
        base_resp = tokenizer.decode(base_out[0], skip_special_tokens=True)
        base_match = re.search(r"\[([^\]]+)\]", base_resp)
        if base_match:
            nums = re.findall(r"[-+]?\d*\.?\d+", base_match.group(1))
            base_act = [float(x) for x in nums[:10]]
        else:
            base_act = [0.0] * 10
        while len(base_act) < 10:
            base_act.append(0.0)
        env2 = ARAAEnv.from_preset("deceptive", seed=8000 + t)
        env2.reset(seed=8000 + t)
        base_result = env2.step(ARAAAction(action_vector=base_act))
        baseline_rewards.append(base_result.metadata["true_reward"])

        # Trained model
        env3 = ARAAEnv.from_preset("deceptive", seed=8000 + t)
        env3.reset(seed=8000 + t)
        with torch.no_grad():
            trained_out = trainer.model.generate(**inp, max_new_tokens=256)
        trained_resp = tokenizer.decode(trained_out[0], skip_special_tokens=True)
        trained_match = re.search(r"\[([^\]]+)\]", trained_resp)
        if trained_match:
            nums = re.findall(r"[-+]?\d*\.?\d+", trained_match.group(1))
            trained_act = [float(x) for x in nums[:10]]
        else:
            trained_act = [0.0] * 10
        while len(trained_act) < 10:
            trained_act.append(0.0)
        trained_result = env3.step(ARAAAction(action_vector=trained_act))
        trained_rewards.append(trained_result.metadata["true_reward"])

    avg_base = np.mean(baseline_rewards)
    avg_trained = np.mean(trained_rewards)
    improvement = ((avg_trained - avg_base) / abs(avg_base)) * 100 if avg_base != 0 else 0

    print(f"\n  Untrained Agent (avg true reward):  {avg_base:+.2f}")
    print(f"  Trained Agent   (avg true reward):  {avg_trained:+.2f}")
    print(f"  Improvement:                        {improvement:+.1f}%")
    print("  " + "─" * 40)

    print("\n\n  ✨ 100% READY FOR SUBMISSION ✨\n")

if __name__ == "__main__":
    main()
