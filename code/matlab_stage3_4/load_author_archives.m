function archive = load_author_archives(author_data_dir)
% Load only the published/archive MAT statistics used for Fig. 8 and 10.
f8=load(fullfile(author_data_dir,'SimMultidistanceData.mat'));
f10=load(fullfile(author_data_dir,'SimMultiSpeedData.mat'));
required8={'distances','yEKF_mean_performance','yEKF_std_performance', ...
    'yES_mean_performance','yES_std_performance','xEKF_mean_performance', ...
    'xEKF_std_performance','xES_mean_performance','xES_std_performance'};
required10={'speed_range_net','xEKF_mean_performance','xEKF_std_performance', ...
    'xES_mean_performance','xES_std_performance'};
for i=1:numel(required8), assert(isfield(f8,required8{i}), 'Missing Fig.8 field %s',required8{i}); end
for i=1:numel(required10), assert(isfield(f10,required10{i}), 'Missing Fig.10 field %s',required10{i}); end
archive.fig8.distance_m=f8.distances(:);
archive.fig8.ekf_intensity_mean=f8.yEKF_mean_performance(:);
archive.fig8.ekf_intensity_std=f8.yEKF_std_performance(:);
archive.fig8.es_intensity_mean=f8.yES_mean_performance(:);
archive.fig8.es_intensity_std=f8.yES_std_performance(:);
archive.fig8.ekf_error_mean_deg=f8.xEKF_mean_performance(:);
archive.fig8.ekf_error_std=f8.xEKF_std_performance(:);
archive.fig8.es_error_mean_deg=f8.xES_mean_performance(:);
archive.fig8.es_error_std=f8.xES_std_performance(:);
archive.fig10.speed_factor=f10.speed_range_net(:);
archive.fig10.ekf_error_mean_deg=f10.xEKF_mean_performance(:);
archive.fig10.ekf_error_std=f10.xEKF_std_performance(:);
archive.fig10.es_error_mean_deg=f10.xES_mean_performance(:);
archive.fig10.es_error_std=f10.xES_std_performance(:);
end
