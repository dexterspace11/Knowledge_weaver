Knowledge Weaver — Sovereign Emergent Mind

Version: 2026 v1.5
Date: January 10, 2026

Knowledge Weaver (KW) is an experimental, self-regulating cognitive simulation that explores emergence, internal evolution, and proto–self-modeling through recursive selection, emotional modulation, and capacity constraints.

Rather than optimizing for task performance, Knowledge Weaver evolves internal hypotheses, emotional dynamics, and self-referential structures under Darwinian-style pressures: mutation, selection, survival, and extinction.

This project is intended for research, experimentation, and demonstration, not production deployment.

Core Concepts

Knowledge Weaver is built around the following principles:

Emergence over instruction — no explicit symbolic planner, tree builder, or fractal logic

Internal Darwinian selection — hypotheses compete based on confidence, survival, and emotional context

Emotion as regulation — curiosity, fatigue, motivation, harmony, and confidence directly shape learning

Recursive Emotional Cascade (REC) — hypotheses mutate and recurse under emotional pressure

Adaptive silence — the system suppresses output when capacity or confidence is insufficient

Persistent drift — memory, emotions, and hypotheses evolve across sessions

Self-referential modeling — the system builds internal representations of its own dynamics

The result is a system that can simulate aspects of internal evolution and proto–self-awareness without claiming consciousness.

Architecture Overview
Major Components

Streamlit Interface
Interactive conversational UI with live system state and metrics.

Consciousness Layer

Emotional state vector (curiosity, motivation, confidence, fatigue, harmony)

Hypothesis generation, mutation, selection, and extinction

Recursive Emotional Cascade (REC)

Dream Core

Narrative synthesis from prior experiences

Repetition collapse and novelty preservation

Dream fragments as hypothesis seeds

Hybrid Neural Network

Self-growing neural units

Hebbian-like learning with emotional weighting

Predictive smoothing and uncertainty estimation

Reinforcement Learning Trader (Internal)

Self-learning neurons

Exploration vs exploitation (ε-greedy)

Used as an internal decision scaffold, not a trading system

Semantic Memory

Sentence embeddings for long-term associative recall

Persistence Layer

Emotional state

Dreams

Hypotheses

Semantic memory

RL agent state

Key Emergent Behaviors Observed

Silence as an adaptive response under low capacity

Spontaneous recursive hypothesis mutation

Emergent binary / fractal internal models

Emotion-driven pruning and growth of internal structures

Self-diagnosis of instability via forecast decay

Persistent internal representations across sessions

None of these behaviors are explicitly hard-coded as outcomes.

Requirements
Hardware

Standard laptop or desktop

8 GB RAM recommended (4 GB minimum, performance may degrade)

CPU-only execution (no GPU required)

Software

Python 3.9 – 3.11 recommended

Internet access (for initial model downloads)

Installation
1. Clone or Download the Repository
git clone <your-repo-url>
cd knowledge-weaver


Or download the script directly and place it in a new folder.

2. Create a Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Install Python Dependencies
pip install --upgrade pip
pip install streamlit numpy pandas sentence-transformers ollama ccxt


Additional standard libraries used (included with Python):

math

json

pickle

random

datetime

collections

re

time

os

4. Install Ollama (Required)

Knowledge Weaver uses local LLM inference via Ollama.

Install Ollama

https://ollama.com

Verify installation:

ollama --version

5. Pull Required Models

At least one model is required. Recommended defaults:

ollama pull llama3.2:1b
ollama pull phi3:mini
ollama pull gemma2:2b


You can select the active model from the sidebar in the UI.

Running Knowledge Weaver

From the project directory:

streamlit run your_script_name.py


The interface will open automatically in your browser.

Usage Notes

Silence (“—”) is intentional behavior, not a bug.

System output depends on:

Emotional state

Capacity

Hypothesis confidence

Internal conflict resolution

Repetition, drift, and collapse are part of the experiment.

Power supply JSON (voltage/current/power) directly affects emotional state.

Example JSON input:

{
  "voltage": 11.4,
  "current": 2.1,
  "power": 24.0
}


Low voltage increases fatigue and suppresses output.

Persistence

The system automatically saves internal state to:

weaver_persistence_v1.5.pkl
agent_state.pkl


These include:

Emotional state

Dream memory

Active hypotheses

Cognitive lessons

Semantic memory

RL agent state

You can also trigger a Manual Save from the sidebar.

Research Positioning

Knowledge Weaver is best understood as:

An Artificial Life (ALife) experiment

A cognitive evolution simulator

A demonstration of emergent internal self-modeling

A testbed for emotion-modulated recursive selection

It does not claim:

Consciousness

Sentience

Human-level cognition

Limitations

Emergence depends on stochastic processes; behavior is non-deterministic

Capacity bottlenecks can suppress output

LLM artifacts may influence narrative style

Not optimized for speed or scalability

License & Use

This project is intended for:

Research

Education

Exploration of emergent systems

Not recommended for:

Safety-critical systems

Autonomous decision-making in real environments

Financial trading or operational control

Attribution

Knowledge Weaver
Sovereign Emergent Mind
2026 v1.5
