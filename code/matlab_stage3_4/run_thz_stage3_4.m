%% 220 GHz THz Stages 3-4: paper-radius vs pattern-adapted radius
% Log rows are post-motion states at t=0..100 s (initial state included).
clear; close all; clc;
root=fileparts(mfilename('fullpath')); addpath(root);
cfg=thz_stage34_config(root); pattern=load_thz_pattern(cfg);
out_dir=cfg.output_dir;
fig_dir=fullfile(out_dir,'figures');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

paper_static=simulate_thz_stage34(cfg,0,cfg.stage3_steps,'static','paper');
thz_static=simulate_thz_stage34(cfg,0,cfg.stage3_steps,'static','thz');
paper_dynamic_noff=simulate_thz_stage34(cfg,0.2,cfg.steps,'dynamic','paper',false);
paper_dynamic=simulate_thz_stage34(cfg,0.2,cfg.steps,'dynamic','paper',true);
thz_dynamic_noff=simulate_thz_stage34(cfg,0.2,cfg.steps,'dynamic','thz',false);
thz_dynamic=simulate_thz_stage34(cfg,0.2,cfg.steps,'dynamic','thz',true);
static_runs={paper_static,thz_static}; dynamic_runs={paper_dynamic,thz_dynamic};
for i=1:2
    s=static_runs{i};
    assert(s.passive_freeze_max_x<cfg.passive_assert_tol,'Passive state freeze failed.');
    assert(s.passive_freeze_max_P<cfg.passive_assert_tol,'Passive covariance freeze failed.');
    assert(all(s.active1(2:end)+s.active2(2:end)==1),'Active schedule is not mutually exclusive.');
    assert(all(~(s.r1_deg>0 & s.r2_deg>0)),'Both nodes scanned in one slot.');
    assert(all(s.p1_min_eig>-1e-9) && all(s.p2_min_eig>-1e-9),'Static covariance lost PSD.');
end
assert(abs(paper_static.r1_deg(2)-4)<1e-12 && abs(thz_static.r1_deg(2)-thz_static.radius_max_deg(2))<1e-12, ...
    'Initial scan-radius policy mismatch.');
for i=1:2
    d=dynamic_runs{i};
    assert(all(isfinite(d.distance_m)) && all(isfinite(d.e1_deg)) && all(isfinite(d.e2_deg)), ...
        'Dynamic run contains NaN/Inf.');
    assert(all(d.p1_min_eig>-1e-9) && all(d.p2_min_eig>-1e-9),'Dynamic covariance lost PSD.');
end
assert(abs(cfg.prediction_delay_s-2*cfg.ns*cfg.dt_s)<eps, ...
    'Feedforward delay must match the 8 s TDD span.');
archive=load_author_archives(cfg.author_data_dir);

write_timeseries(paper_static,fullfile(out_dir,'paper_radius_stage3_static.csv'));
write_timeseries(thz_static,fullfile(out_dir,'thz_radius_stage3_static.csv'));
write_timeseries(paper_dynamic,fullfile(out_dir,'paper_radius_stage4_dynamic.csv'));
write_timeseries(thz_dynamic,fullfile(out_dir,'thz_radius_stage4_dynamic.csv'));
write_timeseries(paper_dynamic_noff,fullfile(out_dir,'paper_radius_stage4_no_feedforward.csv'));
write_timeseries(thz_dynamic_noff,fullfile(out_dir,'thz_radius_stage4_no_feedforward.csv'));
writetable(struct2table(archive.fig8),fullfile(out_dir,'author_fig8_archive.csv'));
writetable(struct2table(archive.fig10),fullfile(out_dir,'author_fig10_archive.csv'));
writetable(table({'paper';'thz'}, ...
    [paper_static.passive_freeze_max_x;thz_static.passive_freeze_max_x], ...
    [paper_static.passive_freeze_max_P;thz_static.passive_freeze_max_P], ...
    [paper_dynamic.e1_deg(end);thz_dynamic.e1_deg(end)], ...
    [paper_dynamic.e2_deg(end);thz_dynamic.e2_deg(end)], ...
    [paper_dynamic.distance_m(1);thz_dynamic.distance_m(1)], ...
    [paper_dynamic.distance_m(end);thz_dynamic.distance_m(end)], ...
    'VariableNames',{'radius_mode','freeze_x_max','freeze_P_max','rover_final_error_deg', ...
    'elevator_final_error_deg','distance_start_m','distance_end_m'}), ...
    fullfile(out_dir,'radius_mode_summary.csv'));
writetable(table({'paper';'paper';'thz';'thz'},[0;1;0;1], ...
    [paper_dynamic_noff.e1_deg(end);paper_dynamic.e1_deg(end); ...
     thz_dynamic_noff.e1_deg(end);thz_dynamic.e1_deg(end)], ...
    [paper_dynamic_noff.e2_deg(end);paper_dynamic.e2_deg(end); ...
     thz_dynamic_noff.e2_deg(end);thz_dynamic.e2_deg(end)], ...
    [paper_dynamic_noff.distance_m(end);paper_dynamic.distance_m(end); ...
     thz_dynamic_noff.distance_m(end);thz_dynamic.distance_m(end)], ...
    'VariableNames',{'radius_mode','feedforward_enabled','rover_final_error_deg', ...
    'elevator_final_error_deg','distance_end_m'}), ...
    fullfile(out_dir,'feedforward_ablation_summary.csv'));

% 1. Measured cuts, main-lobe width and requested radius bands.
f=figure('Color','w','Position',[80 80 1100 760]); tiledlayout(2,1,'TileSpacing','compact');
nexttile; plot(pattern.h_angle_deg,pattern.h_peak_dbi-pattern.h_loss_db,'LineWidth',1.2); hold on;
plot(pattern.v_angle_deg,pattern.v_peak_dbi-pattern.v_loss_db,'LineWidth',1.2);
yline(pattern.h_peak_dbi-3,'k:'); xline(-pattern.h_hpbw_deg/2,'k:'); xline(pattern.h_hpbw_deg/2,'k:');
xlabel('Angle from measured peak (deg)'); ylabel('Gain (dBi)'); grid on;
legend('H cut','V cut','H peak - 3 dB','Location','best'); title('Measured 220 GHz antenna cuts');
nexttile; bar([cfg.delta_l_deg cfg.delta_h_deg; ...
    min(pattern.h_hpbw_deg,pattern.v_hpbw_deg)*cfg.thz_radius_min_hpbw_fraction, ...
    min(pattern.h_hpbw_deg,pattern.v_hpbw_deg)*cfg.thz_radius_max_hpbw_fraction]);
set(gca,'XTickLabel',{'Paper','THz adapted'}); ylabel('Radius bounds (deg)'); grid on;
exportgraphics(f,fullfile(fig_dir,'fig01_measured_pattern_radius_scales.png'),'Resolution',200); close(f);

% 2. Complementary static scheduler and radius traces for both policies.
f=figure('Color','w','Position',[80 80 1120 900]); tiledlayout(3,1,'TileSpacing','compact');
nexttile; stairs(paper_static.time_s,paper_static.r1_deg,'LineWidth',1.2); hold on;
stairs(paper_static.time_s,paper_static.r2_deg,'LineWidth',1.2);
stairs(thz_static.time_s,thz_static.r1_deg,'--','LineWidth',1.15);
stairs(thz_static.time_s,thz_static.r2_deg,'--','LineWidth',1.15);
xlabel('Time (s)'); ylabel('Applied scan radius (deg)'); grid on;
legend('R1 paper','R2 paper','R1 THz','R2 THz','Location','best'); title('Stage 3 radius comparison');
nexttile; stairs(paper_static.time_s,paper_static.active1,'LineWidth',1.1); hold on;
stairs(paper_static.time_s,paper_static.active2,'LineWidth',1.1); ylim([-0.1 1.1]);
xlabel('Time (s)'); ylabel('Active flag'); grid on; legend('R1','R2');
nexttile; semilogy(paper_static.time_s,max(paper_static.p1_ang_max_deg2,eps),'LineWidth',1.1); hold on;
semilogy(thz_static.time_s,max(thz_static.p1_ang_max_deg2,eps),'LineWidth',1.1);
semilogy(paper_static.time_s,max(paper_static.p2_ang_max_deg2,eps),'--','LineWidth',1.1);
semilogy(thz_static.time_s,max(thz_static.p2_ang_max_deg2,eps),'--','LineWidth',1.1);
xlabel('Time (s)'); ylabel('lambda max(Pang) (deg^2)'); grid on;
legend('R1 paper','R1 THz','R2 paper','R2 THz','Location','best');
exportgraphics(f,fullfile(fig_dir,'fig02_stage3_radius_scheduler_covariance.png'),'Resolution',200); close(f);

% 3. Static true and estimated pointing errors, by axis.
f=figure('Color','w','Position',[80 80 1100 760]); tiledlayout(2,2,'TileSpacing','compact');
plot_error_pair(nexttile,paper_static,thz_static,'e1_h_deg','e1_est_h_deg','Rover H error');
plot_error_pair(nexttile,paper_static,thz_static,'e1_e_deg','e1_est_e_deg','Rover E error');
plot_error_pair(nexttile,paper_static,thz_static,'e2_h_deg','e2_est_h_deg','Elevator H error');
plot_error_pair(nexttile,paper_static,thz_static,'e2_e_deg','e2_est_e_deg','Elevator E error');
exportgraphics(f,fullfile(fig_dir,'fig03_stage3_static_pointing_estimates.png'),'Resolution',200); close(f);

% 4. Passive invariant diagnostics and covariance minimum eigenvalues.
f=figure('Color','w','Position',[80 80 1080 740]); tiledlayout(2,1,'TileSpacing','compact');
nexttile; semilogy(paper_static.time_s,max(paper_static.passive_freeze_x,realmin),'LineWidth',1.2); hold on;
semilogy(paper_static.time_s,max(paper_static.passive_freeze_P,realmin),'LineWidth',1.2);
yline(cfg.passive_assert_tol,'k:'); xlabel('Time (s)'); ylabel('Maximum frozen-entry difference');
legend('x(2:6)','P protected rows/columns','1e-12 tolerance','Location','best'); grid on;
nexttile; plot(paper_static.time_s,paper_static.p1_min_eig,'LineWidth',1.1); hold on;
plot(paper_static.time_s,paper_static.p2_min_eig,'LineWidth',1.1);
yline(0,'k:'); xlabel('Time (s)'); ylabel('Minimum eigenvalue of P'); grid on; legend('R1','R2');
exportgraphics(f,fullfile(fig_dir,'fig04_stage3_passive_invariants_psd.png'),'Resolution',200); close(f);

% 5. Dynamic run comparison: true errors, distance and applied radii.
f=figure('Color','w','Position',[80 80 1120 920]); tiledlayout(4,1,'TileSpacing','compact');
nexttile; plot(paper_dynamic.time_s,paper_dynamic.e1_deg,'LineWidth',1.2); hold on;
plot(paper_dynamic.time_s,paper_dynamic.e2_deg,'LineWidth',1.2);
plot(thz_dynamic.time_s,thz_dynamic.e1_deg,'--','LineWidth',1.2);
plot(thz_dynamic.time_s,thz_dynamic.e2_deg,'--','LineWidth',1.2);
ylabel('LOS error (deg)'); grid on; legend('Rover paper','Elevator paper','Rover THz','Elevator THz','Location','best');
title('Stage 4 orthogonal motion: matched initial state and noise seed');
nexttile; plot(paper_dynamic.time_s,paper_dynamic.distance_m,'LineWidth',1.2); hold on;
plot(thz_dynamic.time_s,thz_dynamic.distance_m,'--','LineWidth',1.2);
ylabel('Distance (m)'); grid on; legend('Paper radius','THz radius','Location','best');
nexttile; stairs(paper_dynamic.time_s,paper_dynamic.r1_deg,'LineWidth',1.1); hold on;
stairs(thz_dynamic.time_s,thz_dynamic.r1_deg,'--','LineWidth',1.1);
stairs(paper_dynamic.time_s,paper_dynamic.r2_deg,'LineWidth',1.1);
stairs(thz_dynamic.time_s,thz_dynamic.r2_deg,'--','LineWidth',1.1);
ylabel('Radius (deg)'); grid on; legend('R1 paper','R1 THz','R2 paper','R2 THz','Location','best');
nexttile; plot(paper_dynamic.time_s,paper_dynamic.los1_h_rate_deg_s,'LineWidth',1.05); hold on;
plot(paper_dynamic.time_s,paper_dynamic.los2_e_rate_deg_s,'LineWidth',1.05);
xlabel('Time (s)'); ylabel('LOS rate (deg/s)'); grid on; legend('Rover H','Elevator E','Location','best');
exportgraphics(f,fullfile(fig_dir,'fig05_stage4_dynamic_mode_comparison.png'),'Resolution',200); close(f);

% 6. Dynamic filter, control, received-power and covariance diagnostics.
f=figure('Color','w','Position',[80 80 1120 1000]); tiledlayout(4,2,'TileSpacing','compact');
plot_series(nexttile,paper_dynamic,thz_dynamic,'e1_est_deg','Estimated Rover error (deg)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'e2_est_deg','Estimated Elevator error (deg)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'u1_h_deg','Rover H control (deg)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'u2_h_deg','Elevator H control (deg)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'q1_dbm','Rover q estimate (dBm)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'q2_dbm','Elevator q estimate (dBm)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'innovation1_db','Rover mean innovation (dB)');
plot_series(nexttile,paper_dynamic,thz_dynamic,'p1_ang_max_deg2','Rover lambda max(Pang) (deg^2)');
exportgraphics(f,fullfile(fig_dir,'fig06_stage4_filter_control_link_diagnostics.png'),'Resolution',200); close(f);

% 7. Horizontal and elevation components expose axis-specific failure modes.
f=figure('Color','w','Position',[80 80 1120 760]); tiledlayout(2,2,'TileSpacing','compact');
plot_series_field(nexttile,paper_dynamic,thz_dynamic,'e1_h_deg','Rover horizontal error (deg)');
plot_series_field(nexttile,paper_dynamic,thz_dynamic,'e1_e_deg','Rover elevation error (deg)');
plot_series_field(nexttile,paper_dynamic,thz_dynamic,'e2_h_deg','Elevator horizontal error (deg)');
plot_series_field(nexttile,paper_dynamic,thz_dynamic,'e2_e_deg','Elevator elevation error (deg)');
exportgraphics(f,fullfile(fig_dir,'fig07_stage4_axis_error_components.png'),'Resolution',200); close(f);

% 8. Orthogonal rover/elevator tracks in world coordinates and link range.
f=figure('Color','w','Position',[80 80 1100 480]); tiledlayout(1,2,'TileSpacing','compact');
nexttile; plot3(paper_dynamic.r1_x_m,paper_dynamic.r1_y_m,paper_dynamic.r1_z_m,'-', 'LineWidth',1.5); hold on;
plot3(paper_dynamic.r2_x_m,paper_dynamic.r2_y_m,paper_dynamic.r2_z_m,'-', 'LineWidth',1.5);
plot3(paper_dynamic.r1_x_m(1),paper_dynamic.r1_y_m(1),paper_dynamic.r1_z_m(1),'ko','MarkerFaceColor','w');
plot3(paper_dynamic.r1_x_m(end),paper_dynamic.r1_y_m(end),paper_dynamic.r1_z_m(end),'ks','MarkerFaceColor','w');
plot3(paper_dynamic.r2_x_m(1),paper_dynamic.r2_y_m(1),paper_dynamic.r2_z_m(1),'ko','MarkerFaceColor','w');
plot3(paper_dynamic.r2_x_m(end),paper_dynamic.r2_y_m(end),paper_dynamic.r2_z_m(end),'ks','MarkerFaceColor','w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); grid on; axis equal; view(35,24);
legend('Rover','Elevator','Start','End','Location','best'); title('Fig. 9 orthogonal trajectory');
nexttile; plot(paper_dynamic.time_s,paper_dynamic.distance_m,'LineWidth',1.3); hold on;
plot(thz_dynamic.time_s,thz_dynamic.distance_m,'--','LineWidth',1.3);
xlabel('Time (s)'); ylabel('Range (m)'); grid on; legend('Paper radius','THz adapted');
title('Inter-terminal distance');
exportgraphics(f,fullfile(fig_dir,'fig08_stage4_orthogonal_trajectory.png'),'Resolution',200); close(f);

% 9-10. Author archive redraws with provenance in titles and labels.
f=figure('Color','w','Position',[80 80 1050 740]); tiledlayout(2,1,'TileSpacing','compact');
nexttile; errorbar(archive.fig8.distance_m,archive.fig8.ekf_intensity_mean,archive.fig8.ekf_intensity_std,'o-'); hold on;
errorbar(archive.fig8.distance_m,archive.fig8.es_intensity_mean,archive.fig8.es_intensity_std,'s--');
xlabel('Distance (m)'); ylabel('Mean intensity (V)'); grid on; legend('EKF archive','ES archive'); title('Author Fig. 8 archive: distance sweep');
nexttile; errorbar(archive.fig8.distance_m,archive.fig8.ekf_error_mean_deg,archive.fig8.ekf_error_std,'o-'); hold on;
errorbar(archive.fig8.distance_m,archive.fig8.es_error_mean_deg,archive.fig8.es_error_std,'s--');
xlabel('Distance (m)'); ylabel('Mean pointing error (deg)'); grid on; legend('EKF archive','ES archive');
exportgraphics(f,fullfile(fig_dir,'fig09_author_fig8_distance_archive.png'),'Resolution',200); close(f);
f=figure('Color','w','Position',[80 80 980 580]);
errorbar(archive.fig10.speed_factor,archive.fig10.ekf_error_mean_deg,archive.fig10.ekf_error_std,'o-','LineWidth',1.1); hold on;
errorbar(archive.fig10.speed_factor,archive.fig10.es_error_mean_deg,archive.fig10.es_error_std,'s--','LineWidth',1.1);
xlabel('Author relative-speed factor'); ylabel('Mean pointing error (deg)'); grid on;
legend('EKF archive','ES archive','Location','northwest'); title('Author Fig. 10 archive: speed sweep');
exportgraphics(f,fullfile(fig_dir,'fig10_author_fig10_speed_archive.png'),'Resolution',200); close(f);

% 9. Passive covariance and parameter audit table.
writetable(table({'Paper';'THz adapted'},[paper_static.passive_freeze_max_x;thz_static.passive_freeze_max_x], ...
    [paper_static.passive_freeze_max_P;thz_static.passive_freeze_max_P], ...
    [paper_dynamic.e1_deg(end);thz_dynamic.e1_deg(end)],[paper_dynamic.e2_deg(end);thz_dynamic.e2_deg(end)], ...
    'VariableNames',{'mode','max_frozen_state_delta','max_frozen_covariance_delta','rover_final_error_deg','elevator_final_error_deg'}), ...
    fullfile(out_dir,'radius_mode_summary.csv'));

fprintf('\nRadius comparison complete.\n');
fprintf('Pattern HPBW H/V %.4f / %.4f deg; THz limits %.4f..%.4f deg.\n', ...
    pattern.h_hpbw_deg,pattern.v_hpbw_deg,thz_static.radius_min_deg(1),thz_static.radius_max_deg(1));
fprintf('Static freeze max: paper x %.3e P %.3e; THz x %.3e P %.3e.\n', ...
    paper_static.passive_freeze_max_x,paper_static.passive_freeze_max_P, ...
    thz_static.passive_freeze_max_x,thz_static.passive_freeze_max_P);
fprintf('Paper dynamic: distance %.4f -> %.4f m; end errors %.4f / %.4f deg.\n', ...
    paper_dynamic.distance_m(1),paper_dynamic.distance_m(end),paper_dynamic.e1_deg(end),paper_dynamic.e2_deg(end));
fprintf('THz dynamic: distance %.4f -> %.4f m; end errors %.4f / %.4f deg.\n', ...
    thz_dynamic.distance_m(1),thz_dynamic.distance_m(end),thz_dynamic.e1_deg(end),thz_dynamic.e2_deg(end));
fprintf('Paper radius no-FF: end errors %.4f / %.4f deg; FF: %.4f / %.4f deg.\n', ...
    paper_dynamic_noff.e1_deg(end),paper_dynamic_noff.e2_deg(end),paper_dynamic.e1_deg(end),paper_dynamic.e2_deg(end));
fprintf('THz radius no-FF: end errors %.4f / %.4f deg; FF: %.4f / %.4f deg.\n', ...
    thz_dynamic_noff.e1_deg(end),thz_dynamic_noff.e2_deg(end),thz_dynamic.e1_deg(end),thz_dynamic.e2_deg(end));
fprintf('Outputs: %s\n',out_dir);

function plot_error_pair(ax,paper,thz,true_name,estimate_name,caption)
axes(ax); plot(paper.time_s,paper.(true_name),'LineWidth',1.1); hold on;
plot(paper.time_s,paper.(estimate_name),'--','LineWidth',1.1);
plot(thz.time_s,thz.(true_name),'LineWidth',1.1); plot(thz.time_s,thz.(estimate_name),'--','LineWidth',1.1);
xlabel('Time (s)'); ylabel('deg'); title(caption); grid on;
legend('Paper true','Paper estimate','THz true','THz estimate','Location','best');
end
function plot_series(ax,paper,thz,name,caption)
axes(ax); plot(paper.time_s,paper.(name),'LineWidth',1.0); hold on;
plot(thz.time_s,thz.(name),'--','LineWidth',1.0); title(caption); xlabel('Time (s)'); grid on;
legend('Paper radius','THz radius','Location','best');
end
function plot_series_field(ax,paper,thz,name,caption)
axes(ax); plot(paper.time_s,paper.(name),'LineWidth',1.0); hold on;
plot(thz.time_s,thz.(name),'--','LineWidth',1.0); title(caption); xlabel('Time (s)'); ylabel('deg'); grid on;
legend('Paper radius','THz radius','Location','best');
end
function write_timeseries(data,path)
names=fieldnames(data); columns={}; labels={}; n=numel(data.time_s);
for i=1:numel(names)
    value=data.(names{i});
    if isnumeric(value) && isvector(value) && numel(value)==n
        columns{end+1}=value(:); labels{end+1}=names{i}; %#ok<AGROW>
    end
end
writetable(table(columns{:},'VariableNames',labels),path);
end
