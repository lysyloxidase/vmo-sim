# VMO-Sim

> Differentiable simulation of the Vastus Medialis Oblique with physics-informed neural networks, Neural ODEs, and reinforcement learning for rehabilitation.

![MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![CI](https://img.shields.io/badge/CI-pytest%20%7C%20mypy%20%7C%20ruff-success)

VMO-Sim is a research-grade Python platform for modeling the Vastus Medialis Oblique (VMO) as a distinct actuator within the quadriceps. It combines classical Hill-type biomechanics with scientific machine learning, EMG processing, reinforcement learning, and validation/analysis tooling for clinically relevant patellofemoral studies.

## Why VMO-Sim?

- Differentiable: the Hill-type muscle model, parameter identification workflow, and FES optimization pipeline all support gradient-based methods.
- VMO-specific: VMO and VML are represented as separate actuators with distinct fiber lengths, pennation angles, and patellar force directions.
- 5 computational layers: biomechanics, PINN surrogates, Neural ODEs, RL rehabilitation, and EMG/clinical analysis live in one coherent codebase.

## Architecture

```text
Layer 5  EMG + Clinical Signal Analysis
         preprocessing -> activation inference -> VMO:VL ratio -> validation

Layer 4  Reinforcement Learning
         VMORehabEnv -> reward shaping -> curriculum -> PPO / SAC wrappers

Layer 3  Scientific Machine Learning Dynamics
         Neural ODE discrepancy models for muscle state evolution

Layer 2  Fast Physics-Informed Surrogates
         PINN force surrogate constrained by Hill-type mechanics

Layer 1  Classical Biomechanics Core
         activation -> force-length -> force-velocity -> tendon -> pennation
         -> Hill muscle -> quadriceps -> patellofemoral mechanics -> fatigue
```

## Biomechanical Model

The quadriceps subsystem models VMO, VML, VL, RF, and VI as separate muscle-tendon units. This separation is the key difference from generic lower-limb simulators when the research question depends on medial patellar stabilization.

| Muscle | Max isometric force (N) | Optimal fiber length (m) | Pennation at optimal | Clinical role | Sources |
| --- | ---: | ---: | ---: | --- | --- |
| VMO | 450 | 0.070 | 50 deg | Primary medial patellar stabilizer | Castanov 2019, Benjafield 2015 |
| VML | 844 | 0.089 | 15 deg | Proximal vastus medialis contribution | Castanov 2019, Ward 2009 |
| VL | 1871 | 0.084 | 5 deg | Dominant lateral patellar pull | Ward 2009, Arnold 2010 |
| RF | 1169 | 0.084 | 5 deg | Biarticular quadriceps actuator | Rajagopal 2016, Arnold 2010 |
| VI | 1365 | 0.087 | 3 deg | Deep knee extensor under RF | Ward 2009, Arnold 2010 |

The Hill-type unit implements:

- First-order excitation-to-activation dynamics following Winters/Thelen-style time constants
- Active and passive force-length relationships
- Concentric and eccentric force-velocity relationships
- Nonlinear compliant tendon mechanics
- Pennation-driven force projection onto the tendon axis
- Quadriceps force balance and simplified patellofemoral tracking
- Xia and Frey fatigue dynamics for long-duration tasks

## Clinical Relevance

- Patellofemoral pain syndrome (PFPS): PFPS accounts for roughly 25-40% of sports-related knee injuries, and VMO weakness or delayed VMO recruitment is a common mechanistic hypothesis.
- ACL rehabilitation: quadriceps inhibition and altered medial-lateral balance after ACL reconstruction can be explored with scenario-specific force reductions and activation delays.
- Knee osteoarthritis (OA): patellofemoral load redistribution and contact pressure changes can be studied under altered quadriceps coordination.

## Machine Learning

- PINN surrogate: learns a force surrogate constrained by Hill-type mechanics. Target use case is rapid approximation of force output with large inference speedup over direct integration.
- Neural ODE: combines physics-based dynamics with a learned correction term for history-dependent effects the classical Hill model does not capture well.
- EMG-to-force models: includes a compact 1D CNN and a bidirectional LSTM for processed EMG mapping.

Current design targets:

- PINN speedup: up to ~1000x relative to repeated equilibrium solves or trajectory integration
- PINN accuracy target: <2% force error for deployment studies, with test-suite guardrails currently set to <5% RMSE on sampled conditions
- CPU only workflow for development and evaluation

## Reinforcement Learning

The RL layer is built around `VMORehabEnv`, a Gymnasium environment for VMO-focused rehabilitation control.

- 6 scenarios: healthy, PFPS mild, PFPS moderate, post-ACL, post-surgical, custom
- 5-stage curriculum: isometric, slow isokinetic, sit-to-stand, stair climb, fatigue/pathology
- Multi-objective reward: tracking, energy, safety, VMO:VL balance, smoothness
- PPO and SAC wrappers through `stable-baselines3` when installed, with a heuristic fallback so the environment still works without the optional RL extra

## Analysis Tools

- Sobol global sensitivity analysis on key VMO parameters
- Gradient-based subject-specific parameter identification
- Gradient-based FES protocol optimization
- Validation helpers against published force, timing, and patellar-tracking trends
- Publication-quality plotting utilities and a Streamlit dashboard

## Quick Start

```bash
git clone <your-fork-or-repo-url>
cd vmo-sim/vmo_sim
python -m pip install -e ".[dev]"
pytest tests/ -v
python scripts/demo.py --quick
```

Optional extras:

```bash
python -m pip install -e ".[dev,rl]"
streamlit run vmo_sim/viz/interactive_dashboard.py
python scripts/train_rl.py --curriculum --scenario pfps_mild --timesteps 100000
```

## Results

The repository includes placeholder result tables in `RESULTS.md` for:

- PINN accuracy and inference speed
- Neural ODE trajectory error and correction magnitude
- RL performance by scenario
- Validation metrics against published studies
- Sensitivity ranking across output variables

## Comparison with Existing Tools

| Tool | VMO/VML separated | Differentiable core | PINN / Neural ODE | RL rehab environment | EMG clinical ratio tools |
| --- | --- | --- | --- | --- | --- |
| VMO-Sim | Yes | Yes | Yes | Yes | Yes |
| OpenSim | Typically no | No | No | No | Limited |
| MyoSuite | No | Limited | No | Whole-body focus | No |
| CEINMS | No | Limited | No | No | Strong EMG integration |
| pymuscle-style models | Rarely | Sometimes | Usually no | No | No |

## Hardware

- CPU only
- Single simulation: <1 s
- PINN training: ~10 min
- RL training: ~1 hour

## References

- Ward, S. R. et al. (2009). Lower limb muscle architecture data used in musculoskeletal modeling.
- Castanov, V. et al. (2019). Architectural separation of VMO and VML.
- Benjafield, A. P. et al. (2015). Ultrasound characterization of VMO fiber orientation.
- De Groote, F. et al. (2016). Smooth muscle-tendon formulations for optimal control.
- Thelen, D. G. (2003). Parameterized muscle mechanics for dynamic contractions.
- Ma, X. et al. (2024). Physics-informed musculoskeletal surrogates for real-time inference.
- Zhang, Y. et al. (2023). PINN formulations for biomechanical system approximation.
- Chen, R. T. Q. et al. (2018). Neural ODEs.
- Xia, T. and Frey Law, L. A. (2008). Three-compartment muscle fatigue model.

```

## License

MIT. See `LICENSE`.
