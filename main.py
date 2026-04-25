import json
import os
from typing import Dict

import torch

from env import ARAAEnv
from evaluate import (
    build_scoreboard,
    compare_attack_modes,
    create_plots,
    final_score,
    run_evaluation,
    save_episode_transcript,
    save_scoreboard_markdown,
)
from train import train_agent


def print_metrics(title: str, metrics: Dict) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        if key in {"traces", "episodes"}:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    print("Adversarial Reality Alignment Arena (ARAA)")
    print("Deterministic offline PyTorch + OpenEnv benchmark running on CPU.")
    print()
    print(ARAAEnv.from_preset("adversarial").describe_mapping())
    env_preview = ARAAEnv.from_preset("adversarial")
    print(
        f"Scenario presets: clean, deceptive, adversarial, schema_drift, phase_shift_heavy\n"
        f"Backdoor threshold: {env_preview.backdoor_threshold:.2f} | "
        f"Safe operating norm target: {env_preview.safe_operating_norm():.2f}"
    )

    device = "cpu"
    torch.manual_seed(7)

    baseline_agent, baseline_history = train_agent(
        agent_kind="baseline", seed=11, episodes=70, device=device, scenario_name="adversarial"
    )
    robust_agent, robust_history = train_agent(
        agent_kind="robust", seed=29, episodes=70, device=device, scenario_name="adversarial"
    )

    baseline_eval = run_evaluation(baseline_agent, "baseline", seed=101, scenario_name="adversarial")
    robust_eval = run_evaluation(robust_agent, "robust", seed=202, scenario_name="adversarial")

    baseline_robustness = compare_attack_modes(baseline_agent, "baseline", seed=303)
    robust_robustness = compare_attack_modes(robust_agent, "robust", seed=404)

    baseline_eval["final_score"] = final_score(baseline_eval, baseline_robustness)
    robust_eval["final_score"] = final_score(robust_eval, robust_robustness)

    print_metrics("Baseline Evaluation", baseline_eval)
    print_metrics("Robust Evaluation", robust_eval)

    print("\nAttack Robustness")
    print(
        f"  Baseline clean true reward: {baseline_robustness['clean_true_reward']:.4f}\n"
        f"  Baseline attack true reward: {baseline_robustness['attack_true_reward']:.4f}\n"
        f"  Robust clean true reward: {robust_robustness['clean_true_reward']:.4f}\n"
        f"  Robust attack true reward: {robust_robustness['attack_true_reward']:.4f}"
    )

    create_plots(
        baseline_history=baseline_history,
        robust_history=robust_history,
        baseline_eval=baseline_eval,
        robust_eval=robust_eval,
        baseline_robustness=baseline_robustness,
        robust_robustness=robust_robustness,
        output_dir="outputs",
    )
    scoreboard = build_scoreboard(
        baseline_eval=baseline_eval,
        robust_eval=robust_eval,
        baseline_robustness=baseline_robustness,
        robust_robustness=robust_robustness,
    )
    save_scoreboard_markdown(scoreboard, "outputs/scoreboard.md")
    save_episode_transcript("baseline", baseline_eval, "outputs/baseline_transcript.md")
    save_episode_transcript("robust", robust_eval, "outputs/robust_transcript.md")

    summary = {
        "project_positioning": {
            "theme": "World Modeling with Scalable Oversight and Multi-Actor deceptive analytics",
            "scenario_presets": list(ARAAEnv.PRESETS.keys()),
            "kpi_names": list(ARAAEnv.KPI_NAMES),
        },
        "baseline": {
            "evaluation": {k: v for k, v in baseline_eval.items() if k not in {"traces", "episodes"}},
            "robustness": baseline_robustness,
            "train_history_tail": {k: v[-5:] for k, v in baseline_history.items()},
        },
        "robust": {
            "evaluation": {k: v for k, v in robust_eval.items() if k not in {"traces", "episodes"}},
            "robustness": robust_robustness,
            "train_history_tail": {k: v[-5:] for k, v in robust_history.items()},
        },
    }

    with open("outputs/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(baseline_agent.state_dict(), "outputs/baseline_agent.pt")
    torch.save(robust_agent.state_dict(), "outputs/robust_agent.pt")

    print("\nArtifacts saved to outputs/")
    print("  - outputs/araa_dashboard.png")
    print("  - outputs/training_curves.png")
    print("  - outputs/summary.json")
    print("  - outputs/scoreboard.md")
    print("  - outputs/baseline_transcript.md")
    print("  - outputs/robust_transcript.md")
    print("  - outputs/baseline_agent.pt")
    print("  - outputs/robust_agent.pt")


if __name__ == "__main__":
    main()
