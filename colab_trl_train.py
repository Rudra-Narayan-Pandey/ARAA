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


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_dataset(num_samples: int = 64) -> Dataset:
    rows = []
    for idx in range(num_samples):
        seed = 5000 + idx
        env = ARAAEnv.from_preset("deceptive", seed=seed, attack_probability=0.12, volatility=0.14)
        observation = env.reset(seed=seed, episode_id=f"colab-{idx}")
        prompt = env.build_llm_prompt(observation)
        rows.append(
            {
                "prompt": prompt,
                "seed": seed,
                "attack_probability": 0.12,
                "volatility": 0.14,
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
        max_completion_length=64,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    print("\n🚀 Starting Clean Training Loop...\n")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[format_reward_func, env_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Run training
    train_result = trainer.train()
    
    # Final Summary Table
    print("\n" + "="*50)
    print("🏆 TRAINING COMPLETE - SUMMARY")
    print("="*50)
    print(f"⏱️  Total Runtime:  {train_result.metrics['train_runtime']:.2f}s")
    print(f"📉 Final Loss:     {train_result.metrics['train_loss']:.4f}")
    print(f"📦 Model Saved:    outputs/trl_colab_run/final_model")
    print("="*50)
    
    trainer.save_model("outputs/trl_colab_run/final_model")
    print("\n✅ READY FOR EVALUATION")

if __name__ == "__main__":
    main()
