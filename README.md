# CARL — Curriculum Adaptive Reinforcement Learning

A deep reinforcement learning project for fixed-wing UAV autonomous path-following, using curriculum RL, TD3, LSTM, and Transformer architectures.

---

## Project Layout

```text
CARL-Draft/
├── carl/                   # Main importable Python package (library code)
│   ├── agents/             # RL agent implementations (BaseAgent, PioneerV1, PioneerSurvival, …)
│   ├── baselines/          # Classical baselines (PID autopilot, PID baseline)
│   ├── curriculum/         # Curriculum managers & visualisation
│   ├── envs/               # UAV Gym-compatible environment
│   ├── evaluation/         # Evaluation loops, shadow testing, policy interrogation
│   ├── logging/            # Spatial trajectory logger
│   ├── models/             # Neural network architectures (MLP, LSTM, Transformer)
│   ├── registry/           # POH (aircraft performance) registry
│   ├── reward/             # Shaped reward engine
│   ├── training/           # Training loops (standard, staged, curriculum, fine-tuning)
│   ├── utils/              # Path helpers, config loader
│   └── viz/                # Interactive viewers (path, curriculum)
│
├── cli/                    # Runnable entry-point scripts
│   ├── train_standard.py
│   ├── train_staged.py
│   ├── train_curriculum.py
│   ├── eval.py
│   ├── launcher.py
│   └── …
│
├── analysis/               # Offline analysis & paper-figure scripts
│   ├── generate_paper_plots.py
│   ├── generate_shap_plots.py
│   └── …
│
├── research/
│   ├── papers/             # Reference PDFs (add manually — not committed)
│   ├── artifacts/          # Architecture notes & run walkthroughs
│   └── scratch/            # Experimental / diagnostic scripts
│
├── tools/
│   └── windows/            # Windows .bat helpers
│
├── assets/                 # Static images / media
├── tests/                  # Smoke & unit tests
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train

```bash
# Standard TD3
python cli/train_standard.py --run-name my_run

# Staged curriculum
python cli/train_staged.py --run-name staged_run

# Adaptive curriculum
python cli/train_curriculum.py --run-name curriculum_run

# LSTM / Transformer variants
python cli/train_standard_lstm.py --run-name lstm_run
python cli/train_staged_transformer.py --run-name transformer_run
```

### 3. Evaluate

```bash
python cli/eval.py --checkpoint outputs/runs/my_run/checkpoints/best.pt --episodes 20
```

### 4. Analysis

```bash
# Generate paper-quality plots
python analysis/generate_paper_plots.py --results-dir outputs/runs/my_run --output-dir outputs/paper_figs

# SHAP explanation
python analysis/generate_shap_plots.py --checkpoint outputs/runs/my_run/checkpoints/best.pt
```

### 5. Export trajectory

```bash
python cli/export_trajectory.py --run-dir outputs/runs/my_run --output-format csv
```

---

## Windows helpers

```batch
tools\windows\launch_training.bat        # launches train_standard
tools\windows\train_all.bat              # runs all training variants
tools\windows\setup_pioneer_env.bat      # creates venv + installs deps
```

---

## Tests

```bash
pytest tests/
```

---

## Package import example

```python
from carl.envs import UAVEnv
from carl.agents.pioneer import PioneerAgentV1
from carl.reward import RewardEngine

env = UAVEnv()
agent = PioneerAgentV1(state_dim=env.observation_space.shape[0],
                       action_dim=env.action_space.shape[0])
```
