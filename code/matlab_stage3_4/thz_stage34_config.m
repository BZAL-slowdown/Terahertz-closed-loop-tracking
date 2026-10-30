function cfg = thz_stage34_config(root)
% Configuration for the 220 GHz Stage 3/4 integration.
% All angular state variables are degrees.  Covariances therefore use
% degree^2 (the unit in the six-state Python estimator and author classes).

cfg.root = root;
% Resolve experiment assets from the repository root so the MATLAB run is
% portable and does not depend on the original desktop workspace.
repo_root = fullfile(root, '..', '..');
cfg.pattern_file = fullfile(repo_root, 'data', 'LWB_CRS-220-100MM-2406-2.xlsx');
cfg.author_data_dir = fullfile(repo_root, 'data', 'author_reference_mat');

cfg.dt_s = 0.5;
cfg.steps = 200;
cfg.stage3_steps = 160;
cfg.ns = 8;
% 同一终端完成 8 点扫描后才再次 Active 更新，因此 IEKF 的参考时刻
% 间隔为 2*ns 个全局采样周期；不能误用单拍 dt_s。
cfg.active_interval_s = 2*cfg.ns*cfg.dt_s;
% Local long-slot actuation-delay assumption for the requested THz ablation.
% It is distinct from sample age and from the Python model's default gimbal delay.
cfg.prediction_delay_s = 2*cfg.ns*cfg.dt_s;
cfg.enable_velocity_feedforward = true;
cfg.scan_step_deg = 45;
cfg.scan_radius_initial_deg = 4; % 论文半径模式初值
cfg.delta_l_deg = 2;             % 论文原始半径下限
cfg.delta_h_deg = 5;             % 论文原始半径上限
cfg.led_hpbw_deg = 35.3;         % 论文高斯模型对应的参考 HPBW
% 作者 NodeEKF.m 对窗口特征值取均值后除以 7000；式(26)写为窗口求和，
% 因而与作者实现对齐的增益为 1/(7000*ns)。它不是论文通用常数。
cfg.K_delta = 1/(7000*cfg.ns);
% THz engineering variant: use the narrower measured cut as a conservative
% circular scan scale. 0.1..0.5 HPBW stays within the measured main-lobe
% half-power radius; these fractions are local design choices, not paper values.
cfg.thz_radius_min_hpbw_fraction = 0.10;
cfg.thz_radius_max_hpbw_fraction = 0.50;
cfg.thz_uncertainty_saturation_fraction = 0.50;

cfg.kp = 0.98;
cfg.ki = 0.10;
cfg.passive_assert_tol = 1e-12;
cfg.passive_schur_margin_ulps = 8;
cfg.seed = 20261030;

% 220 GHz link constants from the existing Python model.  The gaseous
% attenuation is evaluated from the exact P.676-12 values recorded in the
% stage 1/2 THz report; the link budget is kept in dB for numerical scale.
cfg.frequency_hz = 220e9;
cfg.tx_power_dbm = 20;
cfg.bandwidth_hz = 800e6;
cfg.noise_figure_db = 8;
cfg.temperature_k = 290;
cfg.oxygen_db_per_km = 0.015443602907698663;
cfg.water_db_per_km = 2.4713707475850386;
cfg.h_peak_dbi = 44.955;
cfg.v_peak_dbi = 44.691;
cfg.initial_distance_m = 2.3;
cfg.rng_measurement_sigma_db = 0.10;

% Six-state IEKF process model.  The magnitude is deliberately close to
% the Python stage 1/2 model, but is documented here as a local Stage 3/4
% validation configuration rather than a paper parameter claim.
cfg.q0_dbm = link_q0(cfg, cfg.initial_distance_m);
cfg.initial_x = [cfg.q0_dbm; 0; 0; 0; 0; 0];
cfg.initial_P = diag([4.0, 0.08, 0.55^2, 0.55^2, 0.12^2, 0.12^2]);
cfg.q_dot_accel_sigma_db_s2 = 0.015;
cfg.angular_accel_sigma_deg_s2 = 0.003;
cfg.measurement_sigma_db = 0.18;
cfg.iekf_iterations = 8;
cfg.iekf_convergence_tol = 1e-7;
cfg.q_process_scale = 1.0;
cfg.angular_process_scale = 1.0;
cfg.max_command_deg = 1.5;
cfg.output_dir = fullfile(repo_root, 'outputs', 'matlab_stage3_4');
end

function q0 = link_q0(cfg, distance_m)
% Friis loss plus the P.676 gaseous loss.  The antenna pattern contributes
% relative directional loss during measurement, not to q0's peak baseline.
c = 299792458;
fspl_db = 20*log10(4*pi*distance_m*cfg.frequency_hz/c);
gas_db = (cfg.oxygen_db_per_km + cfg.water_db_per_km)*distance_m/1000;
q0 = cfg.tx_power_dbm + cfg.h_peak_dbi + cfg.v_peak_dbi - fspl_db - gas_db;
end
