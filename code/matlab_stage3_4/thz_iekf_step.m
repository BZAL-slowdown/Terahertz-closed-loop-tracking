function [filter, out] = thz_iekf_step(filter, samples, active, cfg, pattern)
% One six-state IEKF step.
% Active uses full prediction/update.  Passive uses equations (27)-(31):
% only q and P(1,1) are propagated/updated, while the remaining state
% entries and every covariance row/column touching them are copied exactly.

if active
    [filter, out] = active_step(filter, samples, cfg, pattern);
else
    [filter, out] = passive_scalar_step(filter, samples, cfg, pattern);
end
end

function [filter, out] = active_step(filter, samples, cfg, pattern)
% Active 批次跨越该终端完整的双向 TDD 周期。首样本预测到批次
% 末端时刻，之后每个样本再按 age_s 对齐，不可只预测一个全局采样周期。
dt = cfg.active_interval_s;
Phi = eye(6);
Phi(1,2) = dt;
Phi(3,5) = dt;
Phi(4,6) = dt;
Q = zeros(6);
q_block = cfg.q_dot_accel_sigma_db_s2^2 * ...
    [dt^4/4, dt^3/2; dt^3/2, dt^2];
ang_block = cfg.angular_accel_sigma_deg_s2^2 * ...
    [dt^4/4, dt^3/2; dt^3/2, dt^2];
Q([1 2],[1 2]) = q_block;
Q([3 5],[3 5]) = ang_block;
Q([4 6],[4 6]) = ang_block;

x_prior = Phi*filter.x;
P_prior = Phi*filter.P*Phi' + Q;
P_prior = (P_prior + P_prior')/2;

z = reshape([samples.z],[],1);
R = cfg.measurement_sigma_db^2*eye(numel(samples));
% Re-linearize all asynchronous scan samples around the current iterate.
% The IEKF correction is anchored to x_prior to avoid treating each
% iteration as a new independent measurement update.
x_post=x_prior; H=zeros(numel(samples),6); h=zeros(numel(samples),1);
for iteration=1:cfg.iekf_iterations
    for i=1:numel(samples)
        [h(i),H(i,:)]=thz_pattern_measurement(x_post,samples(i).scan,samples(i).age,pattern);
    end
    S=H*P_prior*H'+R;
    gain=(P_prior*H')/S;
    residual=z-h-H*(x_prior-x_post);
    updated=x_prior+gain*residual;
    if norm(updated-x_post)<cfg.iekf_convergence_tol
        x_post=updated;
        break;
    end
    x_post=updated;
end
for i=1:numel(samples)
    [h(i),H(i,:)]=thz_pattern_measurement(x_post,samples(i).scan,samples(i).age,pattern);
end
residual=z-h;
S=H*P_prior*H'+R;
gain=(P_prior*H')/S;
I = eye(6); A = I-gain*H;
P_post = A*P_prior*A' + gain*R*gain';
filter.x = x_post;
filter.P = (P_post+P_post')/2;
filter.x(1) = max(filter.x(1), -200);
% PI is evaluated once per completed Active scan batch, not once per global
% sample. Integrate over the elapsed time represented by this prediction.
filter.integral(3:4) = filter.integral(3:4) + dt*filter.x(3:4);
out.x_prior = x_prior; out.P_prior = P_prior; out.C = H;
out.y_hat = h; out.innovation = residual; out.K = gain;
out.control_deg = control_from_state(filter.x, filter, cfg);
out.mode = 'Active';
end

function [filter, out] = passive_scalar_step(filter, samples, cfg, pattern)
% Keep a byte-for-byte logical copy of the untouched states and covariance
% entries so the caller can assert the paper's observability safeguard.
x_before = filter.x;
P_before = filter.P;

sample = samples(1);
% The passive scan radius is zero.  Its scalar measurement derivative C1 is
% evaluated at the carried angular state; no angular rows are updated.
[h, C] = thz_pattern_measurement(filter.x, sample.scan, sample.age, pattern);
q_forecast = filter.x(1);                         % (27)
q11 = cfg.q_dot_accel_sigma_db_s2^2 * cfg.dt_s^4 / 4;
p11_forecast = filter.P(1,1) + q11;             % (28), Q(1,1)
innovation = sample.z-h;
den = C(1)^2*p11_forecast + cfg.measurement_sigma_db^2;
K1 = p11_forecast*C(1)/max(den, realmin);         % (29)
filter.x(1) = q_forecast + K1*innovation;         % (30)
p11_post = max((1-K1*C(1))*p11_forecast, realmin); % (31)

% Explicit carry-forward of all non-q information, including q cross terms.
filter.x(2:6) = x_before(2:6);
filter.P(2:6,:) = P_before(2:6,:);
filter.P(:,2:6) = P_before(:,2:6);
filter.P(1,2:6) = P_before(1,2:6);
filter.P(2:6,1) = P_before(2:6,1);
% Retaining cross-covariances while reducing P(1,1) can violate PSD.
% Raise only the permitted scalar variance to its Schur-complement floor.
P_rest = (P_before(2:6,2:6)+P_before(2:6,2:6)')/2;
cross = P_before(2:6,1);
[V_rest,D_rest] = eig(P_rest);
lambda_rest = real(diag(D_rest));
if min(lambda_rest) < -1e-10
    error('Passive frozen covariance block is not PSD.');
end
positive = lambda_rest > max(size(P_rest))*eps(max(1,max(abs(lambda_rest))));
null_component = V_rest(:,~positive)'*cross;
if norm(null_component) > 1e-10*max(1,norm(cross))
    error('Passive Schur condition failed: cross-covariance is outside range(P_rest).');
end
projected = V_rest(:,positive)'*cross;
schur_floor = sum((projected.^2)./lambda_rest(positive));
roundoff_margin = cfg.passive_schur_margin_ulps * eps(max(1,abs(schur_floor)));
filter.P(1,1) = max(p11_post, schur_floor + roundoff_margin);
filter.P = (filter.P+filter.P')/2;
filter.control_deg = [0;0];

out.x_prior = x_before; out.P_prior = P_before;
out.y_hat = h; out.innovation = innovation; out.C = C;
out.K = [K1; zeros(5,1)]; out.control_deg = [0;0]; out.mode = 'Passive';
end

function u = control_from_state(x, filter, cfg)
% In this state convention e is the LOS offset from the current boresight.
% The Python THz plant adds the positive PI correction to the boresight.
% Keep that sign here; the older LED MATLAB plant used the opposite error
% convention and therefore a negative command.
feedback = cfg.kp*x(3:4) + cfg.ki*filter.integral(3:4);
if cfg.enable_velocity_feedforward
    feedforward = x(5:6) * cfg.prediction_delay_s;
else
    feedforward = zeros(2,1);
end
u = feedback + feedforward;
u = max(-cfg.max_command_deg, min(cfg.max_command_deg, u));
end
