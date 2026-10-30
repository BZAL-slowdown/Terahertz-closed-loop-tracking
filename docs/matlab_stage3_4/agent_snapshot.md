# Agent Snapshot: THz Stage 3/4 Feedforward Integration

## [Continuation Instructions]

- Continue in the current conversation. Before the next modification round, create a fresh workspace by copying this project and its existing outputs; keep the current workspace unchanged.
- Keep the report concise and natural: describe completed work, show the relevant simulation figures, and explain the observed curves with independent engineering judgment.
- Avoid repeated background, generic filler, and template-like wording. Separate local THz results, the LED baseline, and author MAT redraws.
- State limitations plainly. Do not call a run converged unless the recorded errors support that conclusion.

## [System Status]

- Workspace: `D:\太赫兹跟瞄\阶段三四太赫兹融合复现`
- Entry point: `run_thz_stage3_4.m`; common simulator: `simulate_thz_stage34.m`.
- EKF interface: `ekf_bidirectional_step.m` delegates six-state updates to `thz_iekf_step.m`.
- State: `x=[q,q_dot,e_H,e_E,omega_H,omega_E]^T`; angle covariance is in degree squared, angular rate in degree/s.
- Antenna workbook: `D:\太赫兹跟瞄\太赫兹\LWB_CRS-220-100MM-2406-2.xlsx`. Local MATLAB extraction gives HPBW H/V = `0.7026/0.8328 deg`.
- Scheduler: `ns=8`, `dt=0.5 s`, 16-beat mutually exclusive schedule; R1 Active for first 8 beats, then R2 Active for 8 beats. Passive scan radius and motor command are zero.
- Passive update: scalar q/P11 update only; x(2:6), P(2:6,2:6), P(2:6,1), P(1,2:6) copied unchanged. Schur lower bound protects PSD. Static 160-beat protected-state/covariance max delta was `0`, below `1e-12`.
- Feedforward: `prediction_delay_s=2*ns*dt=8 s`, local long-slot actuation-delay assumption. It is not the default Python mechanical command delay. Active law in current MATLAB sign convention is `u=Kp*e+Ki*integral(e)+prediction_delay_s*omega_hat`, then command clipping. Passive output remains zero.
- Python audit: `_AgeAwarePolicy` adds `estimate_velocity * prediction_delay_s` to feedback; `MotionPredictivePolicy` inherits that path. Python source treats sample ages in the estimator separately from the causal delay to command effect; do not add sample ages to the feedforward horizon.
- Dynamic test: speed factor `0.2`, 200 steps, distance `2.3000 -> 2.3087 m`, fixed-seed single runs. Results:

  | Radius mode | FF | Rover final error | Elevator final error |
  |---|---:|---:|---:|
  | paper 2-5 deg | off | 31.0172 deg | 22.3610 deg |
  | paper 2-5 deg | on | 17.0313 deg | 30.6400 deg |
  | THz 0.0703-0.3513 deg | off | 20.6321 deg | 19.3862 deg |
  | THz 0.0703-0.3513 deg | on | 25.7939 deg | 14.4918 deg |

- Interpretation: feedforward is implemented, but stable tracking is not demonstrated. It improves Rover and worsens Elevator in paper-radius mode; under THz radius it worsens Rover and improves Elevator. Do not claim successful interception or main-lobe lock.
- Output files: four dynamic time-series CSVs, static and dynamic plots, `radius_mode_summary.csv`, `feedforward_ablation_summary.csv`, author MAT archive CSVs/plots in `outputs/`.
- Narrative: `阶段三四技术说明与汇报正文.md` contains the technical explanation and copy-ready meeting script. The narrative does not claim feedforward success.

## [Mathematical Anchors]

- Active constant-rate transition over `dt_a`: `Phi(1,2)=dt_a`, `Phi(3,5)=dt_a`, `Phi(4,6)=dt_a`; `x-=Phi*x`, `P-=Phi*P*Phi'+Q`. Same-terminal refresh is `dt_a=8 s`.
- Asynchronous measured-pattern observation: `z_i=q-q_dot*tau_i-L_H(e_H-omega_H*tau_i-s_H)-L_V(e_E-omega_E*tau_i-s_E)+v_i`; Jacobian `[1,-tau_i,g_H,g_E,-tau_i*g_H,-tau_i*g_E]`, where `g=-dL/dangle`.
- Long-slot prediction: `e_H(t+tau_ff)=e_H(t)+tau_ff*omega_H`, `e_E(t+tau_ff)=e_E(t)+tau_ff*omega_E`, `tau_ff=2*ns*dt=8 s`.
- Control migration: `u=feedback+feedforward`; current code uses `feedback=Kp*[e_H,e_E]+Ki*integral`, `feedforward=tau_ff*[omega_H,omega_E]`, then componentwise saturation. Sign is tied to current plant convention and needs independent axis testing.
- Passive scalar update: `q-=q`; `P11-=P11+Q11`; `K1=P11-*C1/(C1^2*P11-+R)`; update q and P11 only. All other states/covariance elements are exactly carried.
- Schur PSD constraint for block covariance: `P11 >= p' * pinv(P_rest) * p`, with `p=P(2:6,1)`, `P_rest=P(2:6,2:6)`. Implementation adds a few machine epsilons to the floor. This ensures a numerical PSD boundary when the block/range assumptions hold; it does not create angular observability or guarantee a specific minimum eigenvalue.
- Adaptive radius: `lambda_k=lambda_max(P_ang)=||P_ang||_2` for symmetric PSD angular covariance. Paper policy uses `sat_[2 deg,5 deg](K_delta*sum(window(lambda)))`; `K_delta=1/(7000*ns)` matches the NodeEKF mean-window scale. THz engineering policy uses `0.1-0.5` times the narrower measured HPBW; these bounds are local choices.

## [Remaining Todo & Traps]

- First validate the sign and reference semantics of `omega_H/E`: LOS rate versus error rate, and each terminal's local tangent-frame orientation. Current dynamic ablation is not a successful stability result.
- Add zero-speed equivalence tests and injected known positive/negative angular-rate tests, then record unclipped feedback/feedforward components and saturation events.
- Confirm command application timing and plant movement ordering against Python timestamps. `8 s` here is the local long-slot delay assumption, not Python's nominal gimbal actuation delay.
- Check Schur conditions for singular `P_rest`: generalized-inverse PSD requires the cross-vector to lie in the range of `P_rest`; fail diagnostically if not. Avoid interpreting an eigenvalue near floating-point zero as a fixed mathematical guarantee.
- THz main-lobe scale is sub-degree; paper 2-5 degree radius is not a local main-lobe scan. Add acquisition/reacquisition separately before claiming capture performance.
- MATLAB link measurement is still simplified relative to Python, including full two-sided beam coupling, receiver noise-floor conversion, continuous payload observation and local/world tangent transforms.
- Use paired multi-seed runs, velocity sweeps and per-axis metrics only after deterministic sign/timing tests pass. Author Fig. 8/10 MAT redraws are not local THz experiments.
