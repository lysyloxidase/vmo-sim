# Results

This file is a placeholder for benchmark and validation outputs generated after running the training, analysis, and demo scripts.

## PINN Surrogate

| Metric | Hill baseline | PINN | Notes |
| --- | ---: | ---: | --- |
| RMSE | - | TBD | Normalized force RMSE |
| MAE | - | TBD | Normalized force MAE |
| R2 | 1.000 | TBD | Relative to Hill-type model |
| Inference time (ms) | TBD | TBD | Batch inference timing |
| Speedup vs Hill | 1.0x | TBD | Higher is better |

## Neural ODE

| Metric | Value | Notes |
| --- | ---: | --- |
| Trajectory RMSE vs Hill | TBD | Fiber length / activation / fatigue |
| Activation RMSE | TBD | Time-series agreement |
| Fiber-length RMSE | TBD | Time-series agreement |
| Neural correction magnitude | TBD | Mean norm relative to physics term |
| Inference time (ms) | TBD | Rollout timing |

## Reinforcement Learning

| Scenario | Mean reward | Mean VMO:VL ratio | Max lateral displacement (mm) | Notes |
| --- | ---: | ---: | ---: | --- |
| Healthy | TBD | TBD | TBD | Evaluation over held-out episodes |
| PFPS mild | TBD | TBD | TBD | |
| PFPS moderate | TBD | TBD | TBD | |
| Post-ACL | TBD | TBD | TBD | |
| Post-surgical | TBD | TBD | TBD | |

## Validation

| Metric | Predicted | Experimental | Relative error | Reference |
| --- | ---: | ---: | ---: | --- |
| VMO isometric force | TBD | TBD | TBD | Ward 2009 |
| VMO-VL onset timing healthy | TBD | TBD | TBD | Cowan 2001 |
| VMO weakness lateral tilt increase | TBD | TBD | TBD | Sakai 2000 |

## Sensitivity

| Output variable | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| --- | --- | --- | --- | --- | --- |
| Peak force | TBD | TBD | TBD | TBD | TBD |
| Patellar displacement | TBD | TBD | TBD | TBD | TBD |
| VMO:VL ratio | TBD | TBD | TBD | TBD | TBD |
| Time to peak | TBD | TBD | TBD | TBD | TBD |
| Fatigue rate | TBD | TBD | TBD | TBD | TBD |
