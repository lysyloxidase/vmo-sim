# Changelog

## 0.1.0 - 2026-04-04

### Phase 1 - Biomechanics Foundation

- Created the full project structure, packaging metadata, MIT license, notebooks, scripts, and CI workflow.
- Implemented the differentiable biomechanics core: activation, force-length, force-velocity, tendon, pennation, Hill muscle, quadriceps, patellofemoral mechanics, and fatigue.
- Added the initial test suite for biomechanics and package-level documentation.

### Phase 2 - Scientific ML and EMG

- Implemented physics-informed losses, the VMO PINN surrogate, Neural ODE discrepancy model, EMG CNN/LSTM models, and the surrogate trainer.
- Built the EMG preprocessing pipeline, EMG-to-activation conversion, clinical VMO:VL ratio analysis, and default electrode placement metadata.
- Expanded tests for PINN, Neural ODE, losses, EMG preprocessing, and VMO:VL analysis.

### Phase 3 - Reinforcement Learning

- Implemented the VMO rehabilitation Gymnasium environment with scenario-specific pathology settings.
- Added the multi-objective rehabilitation reward, staged curriculum, and `stable-baselines3` training wrapper with heuristic fallback.
- Added RL CLI support and environment/reward test coverage.

### Phase 4 - Analysis, Visualization, and Documentation

- Implemented Sobol sensitivity analysis, gradient-based parameter identification, FES protocol optimization, and model validation helpers.
- Added publication-style visualization modules, a Streamlit dashboard, and an end-to-end demo script.
- Rewrote the README, added results placeholders, completed missing CLI utilities, exported public APIs, and ran final cleanup for formatting, typing, and tests.
