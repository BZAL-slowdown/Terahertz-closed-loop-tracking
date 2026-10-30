function pattern = load_thz_pattern(cfg)
% Load the four numeric columns of the measured workbook:
% H gain, H angle, V gain, V angle.  No header spelling is assumed because
% the original Chinese headers can be decoded differently by Excel/MATLAB.
raw = readmatrix(cfg.pattern_file);
raw = raw(all(isfinite(raw(:,1:4)),2),1:4);
assert(size(raw,1) >= 5, 'Antenna workbook has too few numeric rows.');

[pattern.h_angle_deg, pattern.h_loss_db] = ...
    make_axis(raw(:,2), raw(:,1));
[pattern.v_angle_deg, pattern.v_loss_db] = ...
    make_axis(raw(:,4), raw(:,3));
pattern.h_peak_dbi = max(raw(:,1));
pattern.v_peak_dbi = max(raw(:,3));
pattern.h_hpbw_deg = hpbw(pattern.h_angle_deg, pattern.h_loss_db);
pattern.v_hpbw_deg = hpbw(pattern.v_angle_deg, pattern.v_loss_db);
pattern.h_interp = griddedInterpolant(pattern.h_angle_deg, pattern.h_loss_db, 'pchip', 'nearest');
pattern.v_interp = griddedInterpolant(pattern.v_angle_deg, pattern.v_loss_db, 'pchip', 'nearest');
end

function [angle, loss] = make_axis(angle_raw, gain_raw)
[peak, idx] = max(gain_raw);
angle = angle_raw - angle_raw(idx);
[angle, order] = sort(angle(:));
loss = peak - gain_raw(order);
[angle, unique_idx] = unique(angle, 'stable');
loss = max(loss(unique_idx), 0);
end

function width = hpbw(angle, loss)
dense = linspace(angle(1), angle(end), 20001)';
curve = interp1(angle, loss, dense, 'pchip');
valid = dense(curve <= 3);
width = valid(end) - valid(1);
end
