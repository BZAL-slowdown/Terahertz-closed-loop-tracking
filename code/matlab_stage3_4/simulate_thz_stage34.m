function data = simulate_thz_stage34(cfg, speed_factor, n_steps, mode, radius_mode, feedforward_enabled)
% Shared Stage 3/4 loop with comparable paper and THz radius policies.
% Logged rows are post-motion states at t=0..n_steps*dt (initial row included).
if nargin < 5, radius_mode = 'paper'; end
if nargin < 6, feedforward_enabled = cfg.enable_velocity_feedforward; end
cfg.enable_velocity_feedforward = feedforward_enabled;
assert(any(strcmp(radius_mode, {'paper','thz'})), 'Unknown radius mode.');
rng(cfg.seed + round(100*speed_factor) + strcmp(mode,'dynamic'), 'twister');
pattern = load_thz_pattern(cfg);
robots = initialize_robots(cfg, pattern, mode, radius_mode);

n = n_steps;
names = {'time_s','distance_m','r1_deg','r2_deg','active1','active2', ...
    'r1_x_m','r1_y_m','r1_z_m','r2_x_m','r2_y_m','r2_z_m', ...
    'e1_deg','e2_deg','e1_h_deg','e1_e_deg','e2_h_deg','e2_e_deg', ...
    'e1_est_deg','e2_est_deg','e1_est_h_deg','e1_est_e_deg','e2_est_h_deg','e2_est_e_deg', ...
    'q1_dbm','q2_dbm','p1_ang_max_deg2','p2_ang_max_deg2', ...
    'y1_dbm','y2_dbm','innovation1_db','innovation2_db','u1_h_deg','u1_e_deg', ...
    'u2_h_deg','u2_e_deg','gain1','gain2','p1_min_eig','p2_min_eig', ...
    'passive_freeze_x','passive_freeze_P','los1_h_rate_deg_s','los1_e_rate_deg_s', ...
    'los2_h_rate_deg_s','los2_e_rate_deg_s','radius_min_deg','radius_max_deg', ...
    'ff1_h_deg','ff1_e_deg','ff2_h_deg','ff2_e_deg'};
for j=1:numel(names), data.(names{j})=zeros(n+1,1); end
data.radius_mode = radius_mode;
data.radius_rule = radius_rule_text(radius_mode);
data.speed_factor = speed_factor;
data.state_order = '[q,q_dot,e_H,e_E,omega_H,omega_E]';

freeze_x_max = 0; freeze_P_max = 0;
positions={robots(1).position,robots(2).position};
initial_desired=zeros(2,2);
for r=1:2
    los=positions{3-r}-positions{r};
    initial_desired(r,:)=[atan2d(los(2),los(1)),atan2d(los(3),hypot(los(1),los(2)))];
end
previous_los=initial_desired;
data.time_s(1)=0; data.distance_m(1)=norm(robots(2).position-robots(1).position);
for r=1:2
    data.(['r' num2str(r) '_x_m'])(1)=robots(r).position(1);
    data.(['r' num2str(r) '_y_m'])(1)=robots(r).position(2);
    data.(['r' num2str(r) '_z_m'])(1)=robots(r).position(3);
end
data.radius_min_deg(1)=radius_bounds(cfg,pattern,radius_mode,1);
data.radius_max_deg(1)=radius_bounds(cfg,pattern,radius_mode,2);
for r=1:2
    data.(['e' num2str(r) '_h_deg'])(1)=0.3;
    data.(['e' num2str(r) '_deg'])(1)=0.3;
    data.(['e' num2str(r) '_est_h_deg'])(1)=robots(r).filter.x(3);
    data.(['e' num2str(r) '_est_e_deg'])(1)=robots(r).filter.x(4);
    data.(['q' num2str(r) '_dbm'])(1)=robots(r).filter.x(1);
    data.(['p' num2str(r) '_ang_max_deg2'])(1)=max(eig(robots(r).filter.P(3:4,3:4)));
    data.(['p' num2str(r) '_min_eig'])(1)=min(eig(robots(r).filter.P));
end
for k=1:n
    positions = {robots(1).position, robots(2).position};
    desired = zeros(2,2); errors = zeros(2,2); distances=zeros(2,1);
    for r=1:2
        other=3-r;
        los=positions{other}-positions{r}; distances(r)=norm(los);
        desired(r,:)=[atan2d(los(2),los(1)), atan2d(los(3),hypot(los(1),los(2)))];
        errors(r,:)=desired(r,:)-robots(r).heading_deg;
        errors(r,1)=wrap180(errors(r,1));
    end
    active_id=1+ (mod(k-1,2*cfg.ns)>=cfg.ns);
    sample_id=mod(k-1,cfg.ns);
    active=[active_id==1, active_id==2];
    z=zeros(1,2); innov=zeros(1,2); command=zeros(2,2); radius_used=zeros(1,2);
    filter_x_before=cell(1,2); filter_P_before=cell(1,2);
    for r=1:2
        other=3-r; radius=0;
        if active(r), radius=robots(r).radius_deg; end
        radius_used(r)=radius;
        scan=radius*[cosd(sample_id*cfg.scan_step_deg), sind(sample_id*cfg.scan_step_deg)];
        tx_loss=pattern_loss(errors(other,:),pattern);
        rx_loss=pattern_loss(errors(r,:)-scan,pattern);
        q_true=link_q0(cfg,distances(r))-tx_loss;
        z(r)=q_true-rx_loss+cfg.rng_measurement_sigma_db*randn;
        filter_x_before{r}=robots(r).filter.x;
        filter_P_before{r}=robots(r).filter.P;
        sample.scan=scan; sample.age=0; sample.z=z(r);
        if active(r)
            robots(r).pending_samples(end+1)=sample;
            robots(r).last_control=[0;0];
            if sample_id==cfg.ns-1
                for j=1:numel(robots(r).pending_samples)
                    robots(r).pending_samples(j).age=(numel(robots(r).pending_samples)-j)*cfg.dt_s;
                end
                cfg_step=cfg;
                update_time=(k-1)*cfg.dt_s;
                cfg_step.active_interval_s=update_time-robots(r).last_active_update_s;
                [robots(r).filter,out]=ekf_bidirectional_step(robots(r).filter, ...
                    robots(r).pending_samples,true,cfg_step,pattern);
                robots(r).last_control=out.control_deg;
                robots(r).last_innovation=mean(out.innovation);
                robots(r).heading_deg=robots(r).heading_deg+out.control_deg';
                robots(r).pending_samples=robots(r).pending_samples([]);
                robots(r).last_active_update_s=update_time;
                robots(r).eig_history(end+1)=angle_eigenvalue(robots(r).filter.P);
                robots(r).radius_deg=adapt_radius(robots(r).eig_history,cfg,pattern,radius_mode);
            else
                robots(r).eig_history(end+1)=angle_eigenvalue(robots(r).filter.P);
            end
        else
            [robots(r).filter,out]=ekf_bidirectional_step(robots(r).filter,sample,false,cfg,pattern);
            robots(r).last_control=[0;0];
            robots(r).last_innovation=out.innovation;
        end
        innov(r)=robots(r).last_innovation;
        command(:,r)=robots(r).last_control;
        if active(r) && sample_id==cfg.ns-1 && feedforward_enabled
            ff=robots(r).filter.x(5:6)*cfg.prediction_delay_s;
        else
            ff=[0;0];
        end
        data.(['ff' num2str(r) '_h_deg'])(k+1)=ff(1);
        data.(['ff' num2str(r) '_e_deg'])(k+1)=ff(2);

        if ~active(r)
            dx=norm(robots(r).filter.x(2:6)-filter_x_before{r}(2:6),inf);
            dp=max(abs(robots(r).filter.P(2:6,:)-filter_P_before{r}(2:6,:)),[],'all');
            dp=max(dp,max(abs(robots(r).filter.P(:,2:6)-filter_P_before{r}(:,2:6)),[],'all'));
            assert(dx<cfg.passive_assert_tol,'Passive x(2:6) drifted.');
            assert(dp<cfg.passive_assert_tol,'Passive covariance rows/columns drifted.');
            freeze_x_max=max(freeze_x_max,dx); freeze_P_max=max(freeze_P_max,dp);
        end
    end

    % Move once after the simultaneous measurement/control epoch; logs below
    % are then recomputed from the new geometry and labeled with t_{k+1}.
    if speed_factor~=0
        step=0.005*speed_factor/sqrt(2);
        robots(1).position(2)=robots(1).position(2)+step;
        robots(2).position(3)=robots(2).position(3)+step;
    end
    positions={robots(1).position,robots(2).position};
    post_err=zeros(2,2); post_desired=zeros(2,2); post_dist=zeros(2,1);
    for r=1:2
        los=positions{3-r}-positions{r}; post_dist(r)=norm(los);
        post_desired(r,:)=[atan2d(los(2),los(1)),atan2d(los(3),hypot(los(1),los(2)))];
        post_err(r,:)=post_desired(r,:)-robots(r).heading_deg;
        post_err(r,1)=wrap180(post_err(r,1));
    end
    los_rate=(post_desired-desired)/cfg.dt_s;
    los_rate(:,1)=wrap180(los_rate(:,1));
    previous_los=post_desired; %#ok<NASGU>
    row=k+1; data.time_s(row)=k*cfg.dt_s;
    data.distance_m(row)=norm(robots(2).position-robots(1).position);
    for r=1:2
        data.(['r' num2str(r) '_x_m'])(row)=robots(r).position(1);
        data.(['r' num2str(r) '_y_m'])(row)=robots(r).position(2);
        data.(['r' num2str(r) '_z_m'])(row)=robots(r).position(3);
    end
    data.active1(row)=active(1); data.active2(row)=active(2);
    data.(['r1_deg'])(row)=radius_used(1);
    data.(['r2_deg'])(row)=radius_used(2);
    for r=1:2
        data.(['e' num2str(r) '_deg'])(row)=norm(post_err(r,:));
        data.(['e' num2str(r) '_h_deg'])(row)=post_err(r,1);
        data.(['e' num2str(r) '_e_deg'])(row)=post_err(r,2);
        data.(['e' num2str(r) '_est_deg'])(row)=norm(robots(r).filter.x(3:4));
        data.(['e' num2str(r) '_est_h_deg'])(row)=robots(r).filter.x(3);
        data.(['e' num2str(r) '_est_e_deg'])(row)=robots(r).filter.x(4);
        data.(['q' num2str(r) '_dbm'])(row)=robots(r).filter.x(1);
        data.(['p' num2str(r) '_ang_max_deg2'])(row)=max(eig((robots(r).filter.P(3:4,3:4)+robots(r).filter.P(3:4,3:4)')/2));
        data.(['y' num2str(r) '_dbm'])(row)=z(r);
        data.(['innovation' num2str(r) '_db'])(row)=innov(r);
        data.(['u' num2str(r) '_h_deg'])(row)=command(1,r);
        data.(['u' num2str(r) '_e_deg'])(row)=command(2,r);
        data.(['gain' num2str(r)])(row)=active(r)*cfg.kp;
        data.(['p' num2str(r) '_min_eig'])(row)=min(eig((robots(r).filter.P+robots(r).filter.P')/2));
        data.(['los' num2str(r) '_h_rate_deg_s'])(row)=los_rate(r,1);
        data.(['los' num2str(r) '_e_rate_deg_s'])(row)=los_rate(r,2);
    end
    data.radius_min_deg(row)=radius_bounds(cfg,pattern,radius_mode,1);
    data.radius_max_deg(row)=radius_bounds(cfg,pattern,radius_mode,2);
    data.passive_freeze_x(row)=freeze_x_max; data.passive_freeze_P(row)=freeze_P_max;
end
data.passive_freeze_max_x=freeze_x_max; data.passive_freeze_max_P=freeze_P_max;
data.pattern_h_hpbw_deg=pattern.h_hpbw_deg; data.pattern_v_h_hpbw_deg=pattern.v_hpbw_deg;
data.radius_mode=radius_mode; data.radius_rule=radius_rule_text(radius_mode);
end

function robots=initialize_robots(cfg,pattern,mode,radius_mode)
robots(1).position=[1.5;-1.37;0.25]; robots(2).position=[-0.8;-1.37;0.25];
local_bias_deg=0.3;
los1=robots(2).position-robots(1).position; los2=-los1;
desired1=[atan2d(los1(2),los1(1)),atan2d(los1(3),hypot(los1(1),los1(2)))];
desired2=[atan2d(los2(2),los2(1)),atan2d(los2(3),hypot(los2(1),los2(2)))];
robots(1).heading_deg=desired1-[local_bias_deg,0]; robots(2).heading_deg=desired2-[local_bias_deg,0];
for r=1:2
    remote_loss=pattern_loss([local_bias_deg,0],pattern);
    robots(r).filter.x=cfg.initial_x;
    robots(r).filter.x(1)=cfg.initial_x(1)-remote_loss;
    robots(r).filter.P=cfg.initial_P;
    robots(r).filter.integral=zeros(6,1); robots(r).last_control=[0;0];
    robots(r).last_innovation=0; robots(r).eig_history=[]; robots(r).last_active_update_s=0;
    if strcmp(radius_mode,'paper'), robots(r).radius_deg=cfg.scan_radius_initial_deg;
    else
        initial_lambda=angle_eigenvalue(robots(r).filter.P);
        robots(r).radius_deg=adapt_radius(repmat(initial_lambda,cfg.ns,1),cfg,pattern,radius_mode);
    end
    robots(r).pending_samples=struct('scan',{},'age',{},'z',{});
end
end

function radius=adapt_radius(eig_history,cfg,pattern,mode)
window=eig_history(max(1,numel(eig_history)-cfg.ns+1):end);
if strcmp(mode,'paper')
    % Match NodeEKF: floor(mean(lambda)/7000), with mean converted to a sum.
    radius=max(cfg.delta_l_deg,min(floor(cfg.K_delta*sum(window)),cfg.delta_h_deg));
else
    % Engineering law: map angular 1-sigma uncertainty into measured HPBW.
    % At sigma=0.5 HPBW, command the 0.5 HPBW ceiling; clip at 0.1 HPBW.
    hpbw=min(pattern.h_hpbw_deg,pattern.v_hpbw_deg);
    lo=cfg.thz_radius_min_hpbw_fraction*hpbw;
    hi=cfg.thz_radius_max_hpbw_fraction*hpbw;
    sigma=sqrt(max(mean(window),0));
    radius=lo+(hi-lo)*min(sigma/(cfg.thz_uncertainty_saturation_fraction*hpbw),1);
end
end

function lambda=angle_eigenvalue(P)
Pang=(P(3:4,3:4)+P(3:4,3:4)')/2;
lambda=max(eig(Pang));
end

function text=radius_rule_text(mode)
if strcmp(mode,'paper'), text='Paper 2-5 deg clamp; K_delta=1/(7000*ns), window sum.';
else, text='Engineering: 0.1-0.5 x narrower measured HPBW, linear in sigma/HPBW.'; end
end
function value=radius_bounds(cfg,pattern,mode,index)
if strcmp(mode,'paper')
    values=[cfg.delta_l_deg,cfg.delta_h_deg];
else
    hpbw=min(pattern.h_hpbw_deg,pattern.v_hpbw_deg);
    values=hpbw*[cfg.thz_radius_min_hpbw_fraction,cfg.thz_radius_max_hpbw_fraction];
end
value=values(index);
end

function loss=pattern_loss(error_deg,pattern)
lh=axis_value(error_deg(1),pattern.h_angle_deg,pattern.h_interp);
lv=axis_value(error_deg(2),pattern.v_angle_deg,pattern.v_interp); loss=lh+lv;
end
function loss=axis_value(angle,support,interp)
clamped=min(max(angle,support(1)),support(end)); loss=max(interp(clamped),0);
end
function q0=link_q0(cfg,distance_m)
c=299792458; fspl=20*log10(4*pi*distance_m*cfg.frequency_hz/c);
gas=(cfg.oxygen_db_per_km+cfg.water_db_per_km)*distance_m/1000;
q0=cfg.tx_power_dbm+cfg.h_peak_dbi+cfg.v_peak_dbi-fspl-gas;
end
function a=wrap180(a), a=mod(a+180,360)-180; end
