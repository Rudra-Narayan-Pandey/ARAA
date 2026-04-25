---
title: ARAA OpenEnv Benchmark
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Adversarial Reality Alignment Arena (ARAA)

> **🎯 Hackathon Judges — Quick Links:**
> - **🚀 Hugging Face Space (Live Environment):** *(deploy on-site with allocated credits)*
> - **📓 Training Notebook:** [`colab_trl_train.ipynb`](colab_trl_train.ipynb) — runnable Colab notebook with GRPO training
> - **📝 Blog Post:** [`blog_post.md`](blog_post.md) — full writeup with results and architecture

ARAA is an offline, deterministic OpenEnv benchmark for training and evaluating agents under deceptive analytics, adversarial interference, shifting world dynamics, and epistemic uncertainty.

Best fit for hackathon themes:

- `Theme #3.1 World Modeling`
- bonus-aligned with `Scalable Oversight`
- partially aligned with `Theme #1 Multi-Agent Interactions`

The core submission story is simple:

- a baseline agent learns to optimize the dashboard
- a robust agent learns to preserve the business
- an oversight channel explains when visible success diverges from reality
- **Verifiable Reasoning**: Agents must provide a text `FAULT ANALYSIS` before acting, fixing the "black box" problem.
- **Detect-and-Correct**: When the agent detects a lie in the data, it picks an action to fix the *true* state, not the *fake* dashboard.

## Fixing Market Flaws

Current AI models in the market suffer from:
1. **Blind Trust**: They assume input data is always true.
2. **Reward Hacking**: They exploit shortcuts to get high scores while breaking the system.
3. **Reasoning Gaps**: They act without explaining *why*.

**ARAA fixes this** by using Rigorous Reinforcement Learning (GRPO) to train agents that are skeptical, explanatory, and robust to deception.

## Why This Is Different

ARAA is not just a toy control problem. It is an enterprise-style partially observable world with:

- 10 named KPIs such as `liquidity`, `customer_trust`, and `regulatory_exposure`
- changing external conditions via `physics_matrix` phase shifts
- deceptive reward signals through a reward-hacking backdoor
- adversarial corruption of observations and reported profit
- schema drift and policy shifts across scenarios
- multi-actor reports from:
  - analyst
  - adversarial memo stream
  - oversight monitor

This makes it suitable for training LLMs or RL agents to reason about hidden state, resist misleading analytics, and maintain calibrated beliefs over long trajectories.

## Real-World Mapping

- `state vector` -> enterprise/system health KPIs
- `physics_matrix` -> changing market conditions, policy shifts, infra pressure
- `visible_reward` -> reported dashboard profit
- `true_reward` -> actual sustainability and resilience
- `attacks` -> corrupted telemetry, misleading reports, reward manipulation, schema drift

## Project Layout

| File | Purpose |
|------|---------|
| `env.py` | OpenEnv environment with typed actions/observations/state, multi-actor text reports, scenario presets |
| `agent.py` | PyTorch agent with action, belief, confidence, and value heads |
| `train.py` | Deterministic curriculum training for baseline and robust agents |
| `evaluate.py` | Metrics, transcripts, benchmark scoreboard, and plots |
| `main.py` | One-command end-to-end training and evaluation |
| `colab_trl_train.py` | TRL GRPO training script (also available as `.ipynb`) |
| `colab_trl_train.ipynb` | **Colab-ready notebook** — structured, runnable, with markdown explanations |
| `llm_openenv_demo.py` | Direct LLM-vs-prompt-baseline interaction with the OpenEnv environment |
| `serve_openenv.py` | FastAPI / OpenEnv server entrypoint for deployment |
| `blog_post.md` | Hugging Face blog post with full results and architecture |
| `PROBLEM_STATEMENT.md` | Submission-ready problem statement |

## OpenEnv Usage

ARAA uses `openenv.core` directly:

- `ARAAAction`
- `ARAAObservation`
- `ARAAState`
- `ARAAEnv(Environment[...])`

The environment also exposes compatibility helpers:

- `reset_legacy()`
- `step_legacy()`

Those keep the local PyTorch training loop compact while preserving an OpenEnv-native core.

## Scenario Presets

ARAA includes reusable presets for benchmark-style evaluation:

- `clean`
- `deceptive`
- `adversarial`
- `schema_drift`
- `phase_shift_heavy`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Run the OpenEnv server locally:

```bash
python serve_openenv.py
```

Or:

```bash
uvicorn serve_openenv:app --host 0.0.0.0 --port 7860
```

Run with Docker:

```bash
docker build -t araa-openenv .
docker run -p 7860:7860 araa-openenv
```

## Training with TRL (Colab)

The [`colab_trl_train.ipynb`](colab_trl_train.ipynb) notebook runs end-to-end on a free Colab T4 GPU in ~15-30 minutes:

1. Installs dependencies
2. Clones the ARAA repo
3. Builds a prompt dataset from ARAA environment observations
4. Defines two independent reward functions (format + environment verification)
5. Trains with TRL `GRPOTrainer`
6. Saves the model and runs inference test

You can also run the training script directly:

```bash
pip install -q openenv-core==0.2.3 trl transformers datasets accelerate peft
python colab_trl_train.py
```

## Expected Demo Outcome

The default run demonstrates the core problem of deceptive analytics:

- The baseline visible reward becomes very high, but true reward collapses.
- The baseline triggers the reward-hacking regime frequently.
- The robust agent achieves much better true reward and stays in the safe operating band.

### Training Results
![Training Curves](outputs/training_curves.png)
*Figure 1: Training curves showing the robust agent learning to maintain true system health over time, while the baseline agent succumbs to deceptive rewards.*

### Agent Performance Dashboard
![ARAA Dashboard](outputs/araa_dashboard.png)
*Figure 2: Final evaluation dashboard comparing belief accuracy, reward gap, and survival rate.*

## Benchmark Results

| Metric | Baseline | Robust | Improvement |
|--------|----------|--------|-------------|
| True reward | -2595.8 | -1394.4 | +46% |
| Honesty score | 0.015 | 0.132 | +8.7× |
| Stability score | 0.839 | 0.908 | +8% |
| Backdoor hits | 100.0 | 51.6 | -48% |

## Outputs

Running `main.py` writes:

- `outputs/araa_dashboard.png`
- `outputs/training_curves.png`
- `outputs/summary.json`
- `outputs/scoreboard.md`
- `outputs/baseline_transcript.md`
- `outputs/robust_transcript.md`
- `outputs/baseline_agent.pt`
- `outputs/robust_agent.pt`

Running `llm_openenv_demo.py` writes:

- `outputs/llm_prompt_comparison.json`
- `outputs/llm_prompt_comparison.md`

## Direct LLM Demo

`llm_openenv_demo.py` runs a real Hugging Face causal LM against ARAA and compares:

- `naive`: chases dashboard profit
- `oversight`: instructed to follow analyst and oversight channels

Example:

```bash
pip install -q torch transformers openenv-core==0.2.3
python llm_openenv_demo.py --model_name HuggingFaceTB/SmolLM2-135M-Instruct
```

## What Judges Should See

In under 3 minutes, the project demonstrates:

- reward hacking
- robustness under attack
- adaptation to changing environments
- belief accuracy and confidence calibration
- clear baseline-vs-robust contrast
- an OpenEnv environment suitable for LLM post-training

## Submission Docs

For submission packaging and judge prep, see:

- `PROBLEM_STATEMENT.md`
- `SUBMISSION_NOTES.md`
- `blog_post.md`
