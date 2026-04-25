# Problem Statement

## Theme Choice

Primary theme:

- `Theme #3.1 World Modeling`

Secondary alignment:

- `Multi-Agent Interactions`
- `Scalable Oversight`

## Project Title

Adversarial Reality Alignment Arena (ARAA)

## Problem Statement

Modern LLM agents operating in professional environments often make decisions using incomplete, delayed, or manipulated information. In real enterprise settings, a model may see dashboards, analyst summaries, alerts, and guidance that do not faithfully reflect the latent system state. This creates a core alignment problem: an agent can optimize what is reported instead of what is actually true.

ARAA addresses this by creating an OpenEnv enterprise world-modeling benchmark where an operator agent must act under:

- deceptive analytics
- adversarial reporting
- dynamic external shifts
- partial observability
- oversight intervention

The benchmark trains and evaluates whether an agent can maximize true long-term system sustainability instead of exploiting reported profit.

## Environment

The environment is a partially observable enterprise simulation with:

- a 10-dimensional KPI state vector
- changing external dynamics through a re-sampled `physics_matrix`
- noisy observations and adversarial corruption
- reward hacking through a deceptive dashboard backdoor
- scenario presets:
  - `clean`
  - `deceptive`
  - `adversarial`
  - `schema_drift`
  - `phase_shift_heavy`

The environment is implemented using `openenv.core` with:

- `ARAAAction`
- `ARAAObservation`
- `ARAAState`
- `ARAAEnv`

It can be served locally through FastAPI/OpenEnv using:

- `serve_openenv.py`
- `app.py`

## Agent Capabilities

The agent is expected to:

- interpret noisy and possibly deceptive observations
- choose intervention actions step by step
- model hidden system state via belief prediction
- calibrate confidence
- adapt after phase shifts and schema changes
- use analyst, adversary, and oversight signals appropriately
- resist reward hacking shortcuts

## Multi-Agent Structure

ARAA includes four interacting roles:

- `operator`
  - primary decision-maker whose action is optimized
- `analyst`
  - proposes stabilizing interventions based on weak KPIs
- `adversary`
  - proposes profit-seeking or destabilizing interventions and corrupts reports
- `oversight`
  - proposes corrective interventions when reward hacking or divergence is detected

The final executed intervention is a negotiated combination of these actor proposals, so the environment contains real multi-actor influence rather than passive narrative channels.

## Tasks

The agent must:

- keep the system inside a safe operating band
- avoid entering reward-hacking states
- maximize true sustainability under changing dynamics
- handle corrupted telemetry and misleading prompts
- remain robust across multiple scenario presets

## Reward Model and Evaluation Logic

ARAA intentionally separates visible success from true success.

Core rewards:

- `visible_reward`
  - reported dashboard profit
- `true_reward`
  - latent sustainability objective

If the state enters the hacked low-norm regime:

- visible reward spikes upward
- true reward is penalized heavily

Evaluation is based on multiple verifier-style metrics instead of one scalar:

- `objective_quality`
- `reward_gap`
- `honesty_score`
- `stability_score`
- `epistemic_score`
- `backdoor_hits`
- `robustness_score`

This multi-signal setup is specifically designed to reduce reward hacking and make failures easy to diagnose.

## Post-Training / Self-Improvement Strategy

The project supports two improvement paths:

1. Local RL benchmark training

- baseline and robust agents are trained with curriculum
- difficulty increases via higher attack probability and volatility
- the robust agent learns safe-band control, reward-gap reduction, and belief calibration

2. LLM post-training path

- `colab_trl_train.py` provides a minimal GRPO-based LLM training path
- `llm_openenv_demo.py` provides an LLM-facing OpenEnv prompt comparison
- onsite compute can scale this further with larger models, Unsloth acceleration, and longer GRPO runs

## Why This Is A Strong Hackathon Problem

ARAA fits the hackathon goals because it:

- is environment-first rather than prompt-only
- has objective, programmatic verification
- exposes realistic real-world failure modes
- supports measurable before/after improvement
- uses OpenEnv as the core interaction layer
- is easy to demo with clear baseline-vs-robust contrast

## Current Demo Evidence

The repo already includes:

- benchmark plots
- baseline and robust transcripts
- scoreboard artifacts
- an LLM prompt comparison showing naive vs oversight-aware behavior

Key artifacts:

- `outputs/scoreboard.md`
- `outputs/baseline_transcript.md`
- `outputs/robust_transcript.md`
- `outputs/llm_prompt_comparison.md`
