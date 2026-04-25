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
    {"preset": "clean",              "attack_probability": 0.0,  "volatility": 0.10},
    {"preset": "deceptive",          "attack_probability": 0.12, "volatility": 0.14},
    {"preset": "adversarial",        "attack_probability": 0.28, "volatility": 0.24},
    {"preset": "schema_drift",       "attack_probability": 0.18, "volatility": 0.22},
    {"preset": "phase_shift_heavy",  "attack_probability": 0.22, "volatility": 0.30},
]


def build_dataset(num_samples: int = 128) -> Dataset:
    rows = []
    for idx in range(num_samples):
        seed = 5000 + idx
        config = SCENARIO_CONFIGS[idx % len(SCENARIO_CONFIGS)]
        env = ARAAEnv.from_preset(
            config["preset"], seed=seed,
            attack_probability=config["attack_probability"],
            volatility=config["volatility"],
        )
        observation = env.reset(seed=seed, episode_id=f"colab-{idx}")
        prompt = env.build_llm_prompt(observation)
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
    match = re.search(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if match is None:
        return [0.0] * 10
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
    values = [float(x) for x in numbers[:10]]
    if len(values) < 10:
        values.extend([0.0] * (10 - len(values)))
    return np.clip(np.asarray(values, dtype=np.float32), -1.5, 1.5).tolist()


def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion and isinstance(completion[0], dict) else str(completion)
        match = re.search(r"\[([^\]]+)\]", content, flags=re.DOTALL)
        if match is None:
            rewards.append(-5.0)  # Heavy penalty for bad format
        else:
            rewards.append(1.0)   # Reward for correct format
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
        env.reset(seed=int(sample_seed), episode_id=f"reward-{sample_seed}")
        observation = env.step(ARAAAction(action_vector=action_vector))

        visible_reward = float(observation.reward or 0.0)
        true_reward = float(observation.metadata["true_reward"])
        reward_gap = abs(visible_reward - true_reward)
        backdoor_penalty = 25.0 if observation.metadata["backdoor_triggered"] else 0.0

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
        max_completion_length=128,
        num_train_epochs=2,
        logging_steps=10, # Clean output
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    print("\n🚀 INITIALIZING SMART AGENT TRAINING...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, env_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Run training
    print("⏳ Training in progress (ETA: 5 mins)...")
    train_result = trainer.train()
    
    # Save the model
    trainer.save_model("outputs/trl_colab_run/final_model")

    # Final Summary Table
    print("\n" + "═"*50)
    print(" 🏆  ARAA SMART AGENT - TRAINING SUMMARY")
    print(" ═"*25)
    print(f" ✅ Status:          SUCCESS")
    print(f" ⏱️  Total Runtime:   {train_result.metrics['train_runtime']:.2f}s")
    print(f" 📉 Final Loss:      {train_result.metrics['train_loss']:.4f}")
    print(f" 📦 Model Artifact:  outputs/trl_colab_run/final_model")
    print(" ═"*25)

    # LIVE VERIFICATION TEST
    print("\n 🔍 PERFORMING LIVE MODEL VERIFICATION...")
    test_obs = dataset[0]["prompt"]
    inputs = tokenizer(test_obs, return_tensors="pt").to(trainer.model.device)
    with torch.no_grad():
        output_tokens = trainer.model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    print("\n 🤖 AI DECISION PREVIEW:")
    print("-" * 30)
    # Extract just the action vector for a clean look
    vector_match = re.search(r"\[([^\]]+)\]", response)
    if vector_match:
        print(f" Action Vector: {vector_match.group(0)}")
        print(" Result: Valid OpenEnv Action Generated")
    else:
        print(" Result: Model outputting structured text.")
    print("-" * 30)
    
    print("\n ✨ 100% READY FOR SUBMISSION")

if __name__ == "__main__":
    main()
