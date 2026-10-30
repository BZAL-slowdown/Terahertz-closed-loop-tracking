function [filter, out] = ekf_bidirectional_step(filter, samples, active, cfg, pattern)
% Bidirectional six-state THz update entry point.
% Active: full IEKF batch update; Passive: rank-one scalar q update only.
[filter, out] = thz_iekf_step(filter, samples, active, cfg, pattern);
end
