function [z_hat, C] = thz_pattern_measurement(x, scan_offset_deg, age_s, pattern)
% Six-state THz measurement:
% z = q - q_dot*tau - L_H(e_H-omega_H*tau-s_H)
%                         - L_V(e_E-omega_E*tau-s_E) + v.
% Since L is a measured loss in dB, the relative gain is -L.

q = x(1); q_dot = x(2);
e_h = x(3); e_e = x(4);
w_h = x(5); w_e = x(6);
residual = [e_h - w_h*age_s - scan_offset_deg(1), ...
            e_e - w_e*age_s - scan_offset_deg(2)];
[loss_h, dloss_h] = axis_value(residual(1), pattern.h_angle_deg, pattern.h_interp);
[loss_e, dloss_e] = axis_value(residual(2), pattern.v_angle_deg, pattern.v_interp);
z_hat = q - q_dot*age_s - loss_h - loss_e;

% d(-loss)/d state angle is -dloss.  The velocity derivatives include
% the -age factor from the state transform, matching the Python IEKF.
g_h = -dloss_h; g_e = -dloss_e;
C = [1, -age_s, g_h, g_e, -age_s*g_h, -age_s*g_e];
end

function [loss, slope] = axis_value(angle, support, interp)
clamped = min(max(angle, support(1)), support(end));
loss = max(interp(clamped), 0);
if clamped ~= angle || loss <= 0 || clamped == support(1) || clamped == support(end)
    slope = 0;
else
    % Differentiate the same PCHIP loss curve used by the measurement model.
    % This avoids interpolating a separately sampled gradient, which can
    % produce a Jacobian inconsistent with the actual antenna response.
    h = min(1e-5, 0.25*min(clamped-support(1), support(end)-clamped));
    left = max(support(1), clamped-h);
    right = min(support(end), clamped+h);
    slope = (interp(right)-interp(left))/(right-left);
end
end
