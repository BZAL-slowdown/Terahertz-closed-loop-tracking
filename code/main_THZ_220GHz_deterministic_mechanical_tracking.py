"""220 GHz continuous-communication mechanical beam tracking.

Power samples feed recursive angle/rate estimation; optional motion feed-forward
compensates command latency. Scans and actuator motion preserve payload activity.
EnergyMAP with unit feedback is the paper configuration; IEKF/PI are explicit
legacy options. Empirical-horizon sweeps are retained as legacy experiments.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
import warnings
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Protocol

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, least_squares

# =============================================================================
# Simulation parameters
# =============================================================================

# Paper entry: continuous_prediction (EnergyMAP + unit feedback).
# Receiver samples are extracted during payload transmission. Mechanical scan,
# recenter and stationary phases all contribute to SNR-based communication metrics.
# Legacy IEKF/PI sweeps remain explicitly selectable for compatibility.
SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent
DEFAULT_ANTENNA_FILENAME = 'LWB_CRS-220-100MM-2406-2.xlsx'
DEFAULT_ANTENNA_PATH = WORKSPACE / DEFAULT_ANTENNA_FILENAME
DEFAULT_OUTPUT_DIRECTORY = 'IEEE_THZ_220GHz_EKF_Mechanical_Feasibility'
DEFAULT_RESULTS_DIRECTORY = 'results_THZ_220GHz'
DEFAULT_TWC_RESULTS_DIRECTORY = 'results_TWC'
C_MPS = 299792458.0
FREQUENCY_HZ = 220000000000.0
WAVELENGTH_M = C_MPS / FREQUENCY_HZ
ATMOSPHERIC_PRESSURE_HPA = 1013.25
ATMOSPHERIC_TEMPERATURE_K = 288.15
WATER_VAPOUR_DENSITY_G_M3 = 7.5
DEFAULT_TX_POWER_DBM = 20.0
DEFAULT_BANDWIDTH_HZ = 800000000.0
DEFAULT_NOISE_FIGURE_DB = 8.0
DEFAULT_TEMPERATURE_K = 290.0
DEFAULT_DURATION_S = 100.0
DEFAULT_DT_S = 0.01
DEFAULT_DISTANCE_M = 1000.0
DEFAULT_SCAN_LOSS_DB = 0.75
DEFAULT_QUANTIZATION_DEG = 0.1
DEFAULT_COMMAND_TIME_S = 0.1
GIMBAL_SLEW_LIMIT_DEG_S = 360.0
GIMBAL_EL_LIMIT_DEG = 90.0
INITIAL_ERROR_PHASE_A_DEG = 0.0
INITIAL_ERROR_PHASE_B_DEG = 120.0
EKF_PROCESS_SCALE = 1.0
EKF_INITIAL_Q_STD_DBM = 1.0
EKF_INITIAL_Q_DOT_STD_DB_PER_S = 0.25
EKF_Q_ACCEL_SIGMA_DB_PER_S2 = 0.05
EKF_INITIAL_ANGLE_STD_DEG = 0.05
EKF_INITIAL_VEL_STD_DEG_S = 0.1
EKF_ASSUMED_MEASUREMENT_SIGMA_DB = 0.25
IEKF_ITERATIONS = 8
INNOVATION_CONSISTENCY_DESIGN_THRESHOLD = 11.34486673
INNOVATION_INFLATION_CAP = 100.0
MAX_ACCEL_MPS2 = 2.0
ACCEL_PRIOR_SIGMA_DIVISOR = 3.0
PI_KP = 0.98
PI_KI = 0.2
PI_DEADBAND_DEG = 0.05
PI_LIMIT_DEG = 3.0
DEFAULT_PREDICTION_HORIZON_COMMANDS = 4.5


@dataclass(frozen=True)
class PowerObservationConfig:
    """Statistics of the receiver-side power observation in dB/dBm units.

    ``measurement_sigma_db`` is independent additive receiver/readout noise.
    ``model_error_sigma_db`` is a cycle-wise constant model-error term rather
    than an implausible white process at every numerical time step.  The
    simulation creates exactly one :class:`PowerObservationModel`; no global
    NumPy random state is used.
    """

    enabled: bool = True
    mode: Literal['gaussian_db', 'finite_sample_power'] = 'gaussian_db'
    measurement_sigma_db: float = 0.25
    model_error_sigma_db: float = 0.0
    model_error_refresh_cycles: int = 1
    servo_jitter_sigma_deg: float = 0.0
    seed: int = 1

    def validate(self) -> None:
        if self.measurement_sigma_db < 0.0:
            raise ValueError('measurement_sigma_db must be non-negative')
        if self.mode not in {'gaussian_db', 'finite_sample_power'}:
            raise ValueError("mode must be 'gaussian_db' or 'finite_sample_power'")
        if self.model_error_sigma_db < 0.0:
            raise ValueError('model_error_sigma_db must be non-negative')
        if self.servo_jitter_sigma_deg < 0.0:
            raise ValueError('servo_jitter_sigma_deg must be non-negative')
        if self.model_error_refresh_cycles <= 0:
            raise ValueError('model_error_refresh_cycles must be a positive integer')


@dataclass(frozen=True)
class PowerObservation:
    """One recorded stochastic observation; the physical truth is preserved."""

    true_pr_dbm: float
    observed_pr_dbm: float
    measurement_noise_db: float
    model_error_db: float


class PowerObservationModel:
    """Explicit RNG-owned power-observation channel for the three scan samples."""

    def __init__(self, config: PowerObservationConfig, rng: np.random.Generator) -> None:
        config.validate()
        self.config = config
        self.rng = rng
        self._cycle_index = -1
        self._model_error_db = 0.0

    def observe_dbm(self, true_power_dbm: float, tracking_cycle_index: int) -> PowerObservation:
        """Return ``P_r + X_model + n_P`` without changing the true link state."""
        if not self.config.enabled:
            return PowerObservation(float(true_power_dbm), float(true_power_dbm), 0.0, 0.0)
        if self.config.mode == 'finite_sample_power':
            raise NotImplementedError(
                'finite_sample_power is a reserved observation interface; '
                'implement the receiver power estimator before selecting it.'
            )
        refresh = tracking_cycle_index // self.config.model_error_refresh_cycles
        if refresh != self._cycle_index:
            self._cycle_index = refresh
            self._model_error_db = (
                float(self.rng.normal(0.0, self.config.model_error_sigma_db))
                if self.config.model_error_sigma_db > 0.0 else 0.0
            )
        noise_db = (
            float(self.rng.normal(0.0, self.config.measurement_sigma_db))
            if self.config.measurement_sigma_db > 0.0 else 0.0
        )
        observed = float(true_power_dbm) + self._model_error_db + noise_db
        return PowerObservation(float(true_power_dbm), observed, noise_db, self._model_error_db)


@dataclass(frozen=True)
class TrialRandomness:
    """Trial-level variables shared by every policy in a Monte Carlo trial."""

    seed: int
    initial_phase_a_rad: float
    initial_phase_b_rad: float

    @classmethod
    def from_seed(cls, seed: int) -> 'TrialRandomness':
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * math.pi, size=2)
        return cls(seed=int(seed), initial_phase_a_rad=float(phases[0]),
                   initial_phase_b_rad=float(phases[1]))

# =============================================================================
# Unified continuous-communication metric entry
# =============================================================================

def interval_weights(
    time_s: np.ndarray,
    window_start_s: float = 0.0,
    window_end_s: float | None = None,
) -> np.ndarray:
    """每个记录代表区间 ``(t_prev, t]`` 与评估窗口 ``[start, end]`` 的交集长度。

    首条记录是初始化快照, 不覆盖任何物理区间, 权重为 0 (指南 5.2:
    "初始化快照不能额外计作一个物理步")。暖机边界与结束边界跨越区间时,
    区间与窗口求交裁剪。这是全时段指标的唯一边界语义。
    """
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim == 0 or len(time_s) == 0:
        raise ValueError('interval_weights needs at least one sample')
    end = float(window_end_s if window_end_s is not None else time_s[-1])
    start = float(window_start_s)
    t_prev = np.empty_like(time_s)
    t_prev[0] = time_s[0]  # 初始化快照的区间为空, 权重 0
    if len(time_s) > 1:
        t_prev[1:] = time_s[:-1]
    return np.maximum(0.0, np.minimum(time_s, end) - np.maximum(t_prev, start))


def link_metrics_from_arrays(
    time_s: np.ndarray,
    snr_db: np.ndarray,
    support_valid: np.ndarray,
    snr_threshold_db: float,
    service_rate_bps_hz: float,
    dt_s: float,
    mechanical_active: np.ndarray | None = None,
    window_start_s: float = 0.0,
    window_end_s: float | None = None,
) -> dict[str, float]:
    """全时段连续通信指标的唯一数值实现。

    对评估时间区间的全部时刻积分 (指南 5.2):

      gamma_i   = 10^(snr_db_i / 10)
      se_i      = log2(1 + gamma_i) 若支持域有效, 否则 0
      available = 支持域有效 且 snr_db_i >= 门限
      mean_se   = sum(dt_i * se_i) / sum(dt_i)
      availability = sum(dt_i * available_i) / sum(dt_i)
      service_goodput = service_rate_bps_hz * availability

    每条记录代表 ``(t_prev, t]``; 权重为该区间与窗口 ``[window_start_s,
    window_end_s]`` 的交集长度, 初始化快照权重为 0。最长连续中断按实际区间
    长度累计, 不用记录个数乘固定 dt。机械活动比例只作诊断, 不再乘入 SE。
    """
    time_s = np.asarray(time_s, dtype=float)
    snr_db = np.asarray(snr_db, dtype=float)
    support_valid = np.asarray(support_valid, dtype=bool)
    if time_s.ndim == 0 or len(time_s) == 0:
        raise ValueError('link_metrics_from_arrays needs at least one sample')
    if mechanical_active is None:
        mechanical_active = np.zeros(len(time_s), dtype=bool)
    mechanical_active = np.asarray(mechanical_active, dtype=bool)
    dt = interval_weights(time_s, window_start_s, window_end_s)
    gamma = np.power(10.0, np.clip(snr_db, -200.0, 100.0) / 10.0)
    valid = support_valid
    se = np.where(valid, np.log2(1.0 + gamma), 0.0)
    available = valid & (snr_db >= snr_threshold_db)
    total = float(np.sum(dt))
    mean_se = float(np.sum(dt * se) / total) if total > 0.0 else 0.0
    availability = float(np.sum(dt * available) / total) if total > 0.0 else 0.0
    outage = ~available
    max_outage_s = 0.0
    run = 0.0
    for weight, bad in zip(dt, outage):
        if bad:
            run += weight
            max_outage_s = max(max_outage_s, run)
        else:
            run = 0.0
    scan = mechanical_active
    stationary = ~scan
    scan_total = float(np.sum(dt[scan])) if np.any(scan) else 0.0
    stationary_total = float(np.sum(dt[stationary])) if np.any(stationary) else 0.0
    scan_se = float(np.sum(dt[scan] * se[scan]) / scan_total) if scan_total > 0.0 else float('nan')
    stationary_se = (
        float(np.sum(dt[stationary] * se[stationary]) / stationary_total)
        if stationary_total > 0.0 else float('nan')
    )
    scan_availability = (
        float(np.sum(dt[scan] * available[scan]) / scan_total) if scan_total > 0.0 else float('nan')
    )
    return {
        'continuous_se': mean_se,
        'availability': availability,
        'service_goodput': float(service_rate_bps_hz * availability),
        'max_outage_s': float(max_outage_s),
        'total_outage_s': float(np.sum(dt * outage)),
        'total_weight_s': total,
        'scan_fraction': float(np.sum(dt[scan]) / total) if total > 0.0 else 0.0,
        'scan_se': scan_se,
        'stationary_se': stationary_se,
        'scan_availability': scan_availability,
        'full_support_exit': int(np.any(~valid)),
        'support_invalid_fraction': float(np.sum(dt * ~valid) / total) if total > 0.0 else 0.0,
        'mean_snr_db': float(np.sum(dt * snr_db) / total) if total > 0.0 else float('nan'),
        'beam_rmse_deg': float('nan'),  # filled by caller when beam errors are available
    }


class ContinuousLinkMetrics:
    """运行中的全时段指标累加器, 与 ``link_metrics_from_arrays`` 同口径。

    ``_record`` 每步调用 ``add``, ``get_snapshot``/GUI 实时读取, 而
    ``_summarize`` 用完整数组重算并通过同一语义的验收测试保证一致。
    累加器显式携带评估窗口 ``[window_start_s, window_end_s]``; 初始化快照
    (t=0) 的权重为 0, 跨越窗口边界的区间按交集裁剪。
    """

    def __init__(self, threshold_db: float, service_rate_bps_hz: float,
                 window_start_s: float = 0.0, window_end_s: float | None = None) -> None:
        self.threshold_db = float(threshold_db)
        self.service_rate_bps_hz = float(service_rate_bps_hz)
        self.window_start_s = float(window_start_s)
        self.window_end_s = window_end_s
        self._dt_se = 0.0
        self._dt_available = 0.0
        self._dt_total = 0.0
        self._dt_scan_se = 0.0
        self._dt_scan_available = 0.0
        self._dt_scan = 0.0
        self._max_outage_s = 0.0
        self._current_outage_s = 0.0
        self._support_exits = 0
        self._last_time_s: float | None = None

    def add(self, time_s: float, dt_s: float, snr_db: float, support_valid: bool,
            mechanical_active: bool, beam_error_deg: float) -> None:
        t = float(time_s)
        end = self.window_end_s if self.window_end_s is not None else t
        start = float(self.window_start_s)
        if self._last_time_s is not None:
            t_prev = self._last_time_s
        else:
            t_prev = t  # 初始化快照的区间为空, 权重 0
        weight = max(0.0, min(t, end) - max(t_prev, start))
        self._last_time_s = t
        gamma = 10.0 ** (float(snr_db) / 10.0)
        valid = bool(support_valid)
        se = math.log2(1.0 + gamma) if valid else 0.0
        available = valid and float(snr_db) >= self.threshold_db
        self._dt_total += weight
        self._dt_se += weight * se
        if available:
            self._dt_available += weight
        if mechanical_active:
            self._dt_scan += weight
            self._dt_scan_se += weight * se
            if available:
                self._dt_scan_available += weight
        if not available:
            self._current_outage_s += weight
            self._max_outage_s = max(self._max_outage_s, self._current_outage_s)
        else:
            self._current_outage_s = 0.0
        if not valid:
            self._support_exits += 1
        del beam_error_deg

    def snapshot(self) -> dict[str, float]:
        total = self._dt_total
        if total <= 0.0:
            return {
                'availability': float('nan'), 'continuous_se': float('nan'),
                'service_goodput': float('nan'), 'max_outage_s': float('nan'),
                'scan_fraction': float('nan'),
                'support_exits': self._support_exits,
            }
        return {
            'availability': float(self._dt_available / total),
            'continuous_se': float(self._dt_se / total),
            'service_goodput': float(self.service_rate_bps_hz * self._dt_available / total),
            'max_outage_s': float(self._max_outage_s),
            'scan_fraction': float(self._dt_scan / total),
            'support_exits': int(self._support_exits),
        }

# =============================================================================
# Geometry functions
# =============================================================================

def require_integer_multiple(value: float, base: float, name: str) -> int:
    """检查 value 是 base 的整数倍(固定时间网格可精确表示),返回倍数。

    例如 duration_s 必须是 dt_s 的整数倍、command_time_s 必须是 dt_s 的整数倍,
    否则步进循环无法精确对齐相位切换时刻。
    """
    ratio = value / base
    integer = round(ratio)
    if not math.isclose(ratio, integer, rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'{name}={value} must be an integer multiple of dt_s={base}')
    return int(integer)

def wrap_deg(angle_deg: float) -> float:
    """把角度折叠到 [-180, 180) 区间。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0

def angular_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    """两个单位方向向量之间的夹角 (deg)。"""
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def direction_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    """az/el (deg) → 单位方向向量。az 绕 +z 轴,el 向上为正。"""
    azimuth = math.radians(az_deg)
    elevation = math.radians(el_deg)
    return np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)], dtype=float)

def az_el_from_direction(direction: np.ndarray) -> tuple[float, float]:
    """单位方向向量 → (az, el) (deg)。是 direction_from_az_el 的逆运算。"""
    vector = np.asarray(direction, dtype=float)
    vector = vector / np.linalg.norm(vector)
    return (wrap_deg(math.degrees(math.atan2(vector[1], vector[0]))), math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0)))))

def tangent_basis(az_deg: float, el_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (视轴, 局部 H 轴, 局部 E 轴) 三个单位向量。

    H 轴 = 方位角增加方向(水平旋转),E 轴 = 俯仰角增加方向。
    局部指向误差 (h_deg, e_deg) 即目标方向在这两个轴上的投影角。
    """
    azimuth = math.radians(az_deg)
    elevation = math.radians(el_deg)
    boresight = direction_from_az_el(az_deg, el_deg)
    h_axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    e_axis = np.array([-math.sin(elevation) * math.cos(azimuth), -math.sin(elevation) * math.sin(azimuth), math.cos(elevation)])
    return (boresight, h_axis, e_axis)

def offset_direction(base_az_deg: float, base_el_deg: float, h_deg: float, e_deg: float) -> tuple[float, float]:
    """在 (base_az, base_el) 的局部切平面内施加 (h_deg, e_deg) 偏置,返回新 az/el。

    实现方式: 视轴向量 + h*H轴 + e*E轴 (小角近似) 后归一化。
    """
    boresight, h_axis, e_axis = tangent_basis(base_az_deg, base_el_deg)
    vector = boresight + math.radians(h_deg) * h_axis + math.radians(e_deg) * e_axis
    vector = vector / np.linalg.norm(vector)
    return az_el_from_direction(vector)

def local_error_deg(base_az_deg: float, base_el_deg: float, target_direction: np.ndarray) -> tuple[float, float]:
    """把目标方向投影到 (base_az, base_el) 的局部 H/E 切平面,返回 (h, e) 偏差 (deg)。"""
    boresight, h_axis, e_axis = tangent_basis(base_az_deg, base_el_deg)
    target = np.asarray(target_direction, dtype=float)
    denominator = max(float(np.dot(target, boresight)), 1e-12)
    return (math.degrees(math.atan2(float(np.dot(target, h_axis)), denominator)), math.degrees(math.atan2(float(np.dot(target, e_axis)), denominator)))

def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """单位向量球面线性插值,用于云台的非瞬时连续转动。"""
    start_unit = np.asarray(start, dtype=float)
    end_unit = np.asarray(end, dtype=float)
    start_unit /= np.linalg.norm(start_unit)
    end_unit /= np.linalg.norm(end_unit)
    dot = float(np.clip(np.dot(start_unit, end_unit), -1.0, 1.0))
    if dot > 0.999999:
        out = (1.0 - fraction) * start_unit + fraction * end_unit
        return out / np.linalg.norm(out)
    omega = math.acos(dot)
    return math.sin((1.0 - fraction) * omega) / math.sin(omega) * start_unit + math.sin(fraction * omega) / math.sin(omega) * end_unit

def itu_p676_specific_attenuation_db_per_km(
    pressure_hpa: float = ATMOSPHERIC_PRESSURE_HPA,
    temperature_k: float = ATMOSPHERIC_TEMPERATURE_K,
    water_vapour_density_g_m3: float = WATER_VAPOUR_DENSITY_G_M3,
    version: int = 12,
) -> tuple[float, float]:
    """按 ITU-R P.676 逐线计算 220 GHz 氧气和水汽比衰减 (dB/km)。

    ``itur`` 包提供 P.676-12 的精确逐线实现。其模型版本是进程级设置，
    因此本函数保存并恢复调用前版本，避免影响同进程中的其他实验。
    P.676-13 目前不在该包实现范围内，故默认明确锁定到已实现的 -12，
    而不是把预计算常数误称为 -13 计算结果。
    """
    if pressure_hpa <= 0.0 or temperature_k <= 0.0 or water_vapour_density_g_m3 < 0.0:
        raise ValueError('P.676 requires positive pressure/temperature and non-negative water vapour density')
    if version != 12:
        raise ValueError('The installed itur package supports the exact P.676-12 model only')
    try:
        import itur.models.itu676 as itu676
    except ImportError as error:
        raise ImportError("Install the THz simulation dependency with `pip install itur==0.4.0`") from error

    previous_version = itu676.get_version()
    try:
        itu676.change_version(version)
        oxygen = itu676.gamma0_exact(
            FREQUENCY_HZ / 1e9, pressure_hpa,
            water_vapour_density_g_m3, temperature_k,
        )
        water = itu676.gammaw_exact(
            FREQUENCY_HZ / 1e9, pressure_hpa,
            water_vapour_density_g_m3, temperature_k,
        )
        return float(oxygen.value), float(water.value)
    finally:
        itu676.change_version(previous_version)


def _axis_loss_and_derivative(interpolator: PchipInterpolator, derivative: PchipInterpolator, angle_deg: float) -> tuple[float, float]:
    """单轴 PCHIP 插值: 返回非负损耗及其对角度的一阶导数。

    注意: 原始增益切面中低于峰值 0 dB 的插值残差一律截断为 0,
    保证 L(H,E) >= 0 的物理单调语义,两个接口 (loss 与 gradient) 共用此实现。
    """
    raw_loss = float(interpolator(angle_deg))
    if not math.isfinite(raw_loss):
        raise ValueError('Non-finite antenna loss interpolation result')
    if raw_loss <= 0.0:
        return (0.0, 0.0)
    return (raw_loss, float(derivative(angle_deg)))

# =============================================================================
# Measured antenna pattern: separable PCHIP L(H,E) = L_H(H) + L_E(E)
# =============================================================================

@dataclass(frozen=True)
class PatternQA:
    """方向图质检摘要: 峰值增益、半功率波束宽度 (HPBW) 与测量支撑范围。"""
    h_peak_dbi: float
    e_peak_dbi: float
    common_peak_dbi: float
    peak_mismatch_db: float
    h_hpbw_deg: float
    e_hpbw_deg: float
    h_left_3db_deg: float
    h_right_3db_deg: float
    e_left_3db_deg: float
    e_right_3db_deg: float
    h_range_deg: tuple[float, float]
    e_range_deg: tuple[float, float]

class MeasuredAntennaPattern:
    """直接读取实测 H/V 切面的天线方向图模型。

    要点:
      * 不对峰值增益和角度轴做频率迁移 / HPBW 缩放 / 增益归一化;
      * 以各自切面的峰值为零角度,损耗 L = 峰值增益 - 实测增益;
      * 用 PCHIP (保形分段三次 Hermite) 插值,越界时 clamp 到支撑边界并打标志;
      * 二维模型保持可分离形式 L(H,E) = L_H(H) + L_E(E)。
    """

    def __init__(self, h_angle_deg: np.ndarray, h_gain_dbi: np.ndarray, e_angle_deg: np.ndarray, e_gain_dbi: np.ndarray, source_path: Optional[Path]=None) -> None:
        self.source_path = source_path.resolve() if source_path is not None else None
        h_angle, h_gain = self._sorted_unique(h_angle_deg, h_gain_dbi)
        e_angle, e_gain = self._sorted_unique(e_angle_deg, e_gain_dbi)
        self.h_peak_index = int(np.argmax(h_gain))
        self.e_peak_index = int(np.argmax(e_gain))
        self.h_peak_dbi = float(h_gain[self.h_peak_index])
        self.e_peak_dbi = float(e_gain[self.e_peak_index])
        self.common_peak_dbi = 0.5 * (self.h_peak_dbi + self.e_peak_dbi)
        self.peak_mismatch_db = abs(self.h_peak_dbi - self.e_peak_dbi)
        if self.peak_mismatch_db > 1.0:
            raise ValueError('H/E measured peak mismatch exceeds 1.0 dB; inspect workbook columns and cut definitions')
        self.h_angle = h_angle - h_angle[self.h_peak_index]
        self.e_angle = e_angle - e_angle[self.e_peak_index]
        self.h_loss = self.h_peak_dbi - h_gain
        self.e_loss = self.e_peak_dbi - e_gain
        self.h_angle, self.h_loss = self._sorted_unique(self.h_angle, self.h_loss)
        self.e_angle, self.e_loss = self._sorted_unique(self.e_angle, self.e_loss)
        if len(self.h_angle) < 5 or len(self.e_angle) < 5:
            raise ValueError('each measured cut requires at least five samples')
        self.h_interpolator = PchipInterpolator(self.h_angle, self.h_loss, extrapolate=False)
        self.e_interpolator = PchipInterpolator(self.e_angle, self.e_loss, extrapolate=False)
        self.h_derivative = self.h_interpolator.derivative()
        self.e_derivative = self.e_interpolator.derivative()
        self.qa = self._build_qa()

    @staticmethod
    def _sorted_unique(angle_deg: Iterable[float], gain_or_loss: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        """过滤非有限值、按角度排序、同一角度只保留第一个值。"""
        angles = np.asarray(list(angle_deg), dtype=float)
        values = np.asarray(list(gain_or_loss), dtype=float)
        finite = np.isfinite(angles) & np.isfinite(values)
        angles = angles[finite]
        values = values[finite]
        order = np.argsort(angles)
        angles = angles[order]
        values = values[order]
        unique_angles, first = np.unique(angles, return_index=True)
        return (unique_angles, values[first])

    @classmethod
    def from_excel(cls, path: Path) -> 'MeasuredAntennaPattern':
        """从天线 Excel 工作簿读取 H/E 实测切面 (自动识别中英文列名)。"""
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'antenna workbook was not found: {path}')
        frame = pd.read_excel(path, sheet_name=0)
        normalized = {str(column).strip().lower(): column for column in frame.columns}

        def choose(*tokens: str) -> Any:
            """在列名中按关键字匹配所需的列 (兼容中文/英文命名)。"""
            for name, original in normalized.items():
                if any((token in name for token in tokens)):
                    return original
            raise ValueError(f'cannot identify antenna workbook column from {tokens}; available columns={list(frame.columns)}')
        h_angle_column = choose('h扫描角度', 'h angle', 'h角度')
        h_gain_column = choose('h增益', 'h gain')
        e_angle_column = choose('v扫描角度', 'e扫描角度', 'v angle', 'e angle', 'v角度', 'e角度')
        e_gain_column = choose('v增益', 'e增益', 'v gain', 'e gain')
        h = frame[[h_angle_column, h_gain_column]].apply(pd.to_numeric, errors='coerce').dropna()
        e = frame[[e_angle_column, e_gain_column]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(h) < 5 or len(e) < 5:
            raise ValueError('workbook has fewer than five numeric samples per cut')
        return cls(h.iloc[:, 0].to_numpy(dtype=float), h.iloc[:, 1].to_numpy(dtype=float), e.iloc[:, 0].to_numpy(dtype=float), e.iloc[:, 1].to_numpy(dtype=float), source_path=path)

    @staticmethod
    def _three_db_crossings(angle_deg: np.ndarray, interpolator: PchipInterpolator) -> tuple[float, float]:
        """在峰值两侧分别求解损耗 = 3 dB 的角度交点 (brentq 求根)。

        返回 (左交点, 右交点),HPBW = 右 - 左。
        """
        left_candidates = angle_deg[angle_deg < 0.0]
        right_candidates = angle_deg[angle_deg > 0.0]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            raise ValueError('measured cut must include samples on both sides of peak')

        def root(left: float, right: float) -> float:
            return float(brentq(lambda value: float(interpolator(value)) - 3.0, left, right))
        left_grid = np.sort(left_candidates)
        right_grid = np.sort(right_candidates)
        left_crossings = [root(left, right) for left, right in zip(left_grid[:-1], left_grid[1:]) if (float(interpolator(left)) - 3.0) * (float(interpolator(right)) - 3.0) <= 0.0]
        right_crossings = [root(left, right) for left, right in zip(right_grid[:-1], right_grid[1:]) if (float(interpolator(left)) - 3.0) * (float(interpolator(right)) - 3.0) <= 0.0]
        if not left_crossings or not right_crossings:
            raise ValueError('could not locate both measured 3 dB crossings')
        return (max(left_crossings), min(right_crossings))

    def _build_qa(self) -> PatternQA:
        """构建方向图质检摘要 (峰值 / HPBW / 支撑范围)。"""
        h_left, h_right = self._three_db_crossings(self.h_angle, self.h_interpolator)
        e_left, e_right = self._three_db_crossings(self.e_angle, self.e_interpolator)
        return PatternQA(h_peak_dbi=self.h_peak_dbi, e_peak_dbi=self.e_peak_dbi, common_peak_dbi=self.common_peak_dbi, peak_mismatch_db=self.peak_mismatch_db, h_hpbw_deg=h_right - h_left, e_hpbw_deg=e_right - e_left, h_left_3db_deg=h_left, h_right_3db_deg=h_right, e_left_3db_deg=e_left, e_right_3db_deg=e_right, h_range_deg=(float(self.h_angle[0]), float(self.h_angle[-1])), e_range_deg=(float(self.e_angle[0]), float(self.e_angle[-1])))

    @staticmethod
    def _clamp(angle_deg: float, support: tuple[float, float]) -> tuple[float, bool]:
        """把角度 clamp 到测量支撑范围,返回 (实际使用的角度, 是否发生了越界)。"""
        low, high = support
        clamped = min(max(float(angle_deg), low), high)
        return (clamped, not math.isclose(clamped, angle_deg, abs_tol=1e-14))

    def _axis_loss(self, interpolator: PchipInterpolator, derivative: PchipInterpolator, angle_deg: float, support: tuple[float, float]) -> tuple[float, float, bool]:
        """单轴损耗查询: 先 clamp 到支撑范围,越界时导数置 0 (损耗沿边界平直)。"""
        used_angle, clamped = self._clamp(angle_deg, support)
        loss, slope = _axis_loss_and_derivative(interpolator, derivative, used_angle)
        return (loss, 0.0 if clamped else slope, clamped)

    def loss_and_gradient(self, h_deg: float, e_deg: float) -> tuple[float, np.ndarray, bool]:
        """二维可分离损耗 L(H,E) = L_H(H) + L_E(E) 及其梯度、越界标志。

        返回 (loss_db, dL/d[H,E], clamped)。
        注意: g(H,E) = -L(H,E),所以 g 的梯度 = -dL/d[H,E]。
        """
        h_loss, h_slope, h_clamped = self._axis_loss(self.h_interpolator, self.h_derivative, h_deg, self.qa.h_range_deg)
        e_loss, e_slope, e_clamped = self._axis_loss(self.e_interpolator, self.e_derivative, e_deg, self.qa.e_range_deg)
        return (h_loss + e_loss, np.array([-h_slope, -e_slope], dtype=float), h_clamped or e_clamped)

    def loss(self, h_deg: float, e_deg: float) -> float:
        """返回 clamp 后的可分离实测损耗,链路模型使用。"""
        loss, _, _ = self.loss_and_gradient(h_deg, e_deg)
        return loss

    def local_relative_gain(self, h_deg: float, e_deg: float) -> float:
        """返回 g(H,E) = -L(H,E): 相对峰值增益 (dB, <= 0)。"""
        return -self.loss(h_deg, e_deg)

    def local_gradient(self, h_deg: float, e_deg: float) -> np.ndarray:
        """返回 d g/d(H,E) = -dL/d(H,E),与 loss_and_gradient 使用同一插值。"""
        _, gradient, _ = self.loss_and_gradient(h_deg, e_deg)
        return gradient

    def gain_dbi(self, h_deg: float, e_deg: float) -> tuple[float, bool]:
        """返回 (绝对增益 dBi, 是否越界): gain = common_peak - L(H,E)。"""
        loss, _, clamped = self.loss_and_gradient(h_deg, e_deg)
        return (self.common_peak_dbi - loss, clamped)

    def scan_points(self, scan_loss_db: float = DEFAULT_SCAN_LOSS_DB) -> list[np.ndarray]:
        """Legacy fixed-order wrapper, numerically identical to the original."""
        return self.three_point_design(scan_loss_db, 0.0, (0, 1, 2))

    def three_point_design(self, scan_loss_db: float = DEFAULT_SCAN_LOSS_DB,
                           rotation_deg: float = 0.0,
                           permutation: tuple[int, int, int] = (0, 1, 2)) -> list[np.ndarray]:
        """构造确定性的 0° / 120° / 240° 三点扫描。

        在局部 H/E 平面内沿三个固定相位方向搜索半径 r,使每个点的损耗
        恰好等于 scan_loss_db (默认 0.75 dB)。半径受测量支撑边界限制
        (0.999 倍),若目标损耗超出实测支撑则报错。
        """
        if sorted(permutation) != [0, 1, 2] or not math.isfinite(rotation_deg):
            raise ValueError('invalid three-point rotation/permutation')
        if not math.isfinite(scan_loss_db) or scan_loss_db <= 0.0:
            raise ValueError('scan_loss_db must be positive')
        points: list[np.ndarray] = []
        for phase_deg in (0.0, 120.0, 240.0):
            phase = math.radians(phase_deg + rotation_deg)
            direction = np.array([math.cos(phase), math.sin(phase)])
            h_limit = self.qa.h_range_deg[1] / direction[0] if direction[0] > 0.0 else self.qa.h_range_deg[0] / direction[0]
            e_limit = (self.qa.e_range_deg[1] / direction[1] if direction[1] > 0.0 else self.qa.e_range_deg[0] / direction[1]) if abs(direction[1]) > 1e-12 else math.inf
            radius_limit = 0.999 * min(abs(h_limit), abs(e_limit))
            if not math.isfinite(radius_limit) or radius_limit <= 0.0:
                raise ValueError('measured support cannot form three-point scan')

            def residual(radius: float) -> float:
                return self.loss(radius * direction[0], radius * direction[1]) - scan_loss_db
            if residual(radius_limit) < 0.0:
                raise ValueError(f'scan loss {scan_loss_db} dB is outside measured support')
            radius = brentq(residual, 0.0, radius_limit)
            points.append(radius * direction)
        return [points[i] for i in permutation]


# =============================================================================
# 220-GHz link model
# =============================================================================

@dataclass(frozen=True)
class LinkBudget:
    """220 GHz 链路预算: FSPL + P.676 晴空气体吸收 + 接收功率 / 噪声功率 / SNR。"""
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB
    temperature_k: float = DEFAULT_TEMPERATURE_K

    @property
    def wavelength_m(self) -> float:
        """220 GHz 波长 (m)。"""
        return WAVELENGTH_M

    @property
    def noise_power_dbm(self) -> float:
        """接收机热噪声功率: N = k·T·B,加噪声系数 (dBm)。"""
        boltzmann = 1.380649e-23
        return 10.0 * math.log10(boltzmann * self.temperature_k * self.bandwidth_hz * 1000.0) + self.noise_figure_db

    @cached_property
    def gaseous_specific_attenuation_db_per_km(self) -> tuple[float, float]:
        """(氧气, 水汽) 比衰减 (dB/km),缓存避免重复谱线求和。"""
        return itu_p676_specific_attenuation_db_per_km()

    def fspl_db(self, distance_m: float) -> float:
        """自由空间路径损耗 FSPL = 20·log10(4πd/λ) (dB)。"""
        if distance_m <= 0.0:
            raise ValueError('FSPL requires a positive distance')
        return 20.0 * math.log10(4.0 * math.pi * distance_m / WAVELENGTH_M)

    def attenuation_components_db(self, distance_m: float) -> dict[str, float]:
        """按距离折算的气体吸收分量 (dB)。"""
        if distance_m <= 0.0:
            raise ValueError('atmospheric loss requires a positive distance')
        oxygen_per_km, water_per_km = self.gaseous_specific_attenuation_db_per_km
        scale = distance_m / 1000.0
        oxygen = oxygen_per_km * scale
        water = water_per_km * scale
        gaseous = oxygen + water
        return {'oxygen_attenuation_db': oxygen, 'water_vapour_attenuation_db': water, 'gaseous_attenuation_db': gaseous}

    def propagation_components_db(self, distance_m: float) -> dict[str, float]:
        """总传播损耗 = FSPL + 气体吸收,并返回各分量。"""
        components = self.attenuation_components_db(distance_m)
        fspl = self.fspl_db(distance_m)
        total = fspl + components['gaseous_attenuation_db']
        return {'fspl_db': fspl, **components, 'total_propagation_loss_db': total}

    def received_power_dbm(self, distance_m: float, tx_gain_dbi: float, rx_gain_dbi: float) -> float:
        """接收功率 (dBm) = Ptx + Gtx + Grx - 总传播损耗。"""
        return self.tx_power_dbm + tx_gain_dbi + rx_gain_dbi - self.propagation_components_db(distance_m)['total_propagation_loss_db']

    def snr_db(self, received_power_dbm: float) -> float:
        """SNR (dB) = 接收功率 - 噪声功率。"""
        return received_power_dbm - self.noise_power_dbm

# =============================================================================
# Simulation parameters, motion, and mechanical gimbal
# =============================================================================

@dataclass(frozen=True)
class ActiveScanConfig:
    """Stage-one radius search; geometry extensions are deliberately rejected."""
    enabled: bool = False
    loss_candidates_db: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    rotation_candidates_deg: tuple[float, ...] = (0.0,)
    optimize_order: bool = False
    alpha: float = 1.0
    planning_substeps: int = 12

    def validate(self) -> None:
        allowed = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
        if not self.loss_candidates_db or not set(self.loss_candidates_db) <= allowed:
            raise ValueError('stage-one scan losses must belong to the six declared radii')
        if len(set(self.loss_candidates_db)) != len(self.loss_candidates_db):
            raise ValueError('duplicate scan candidates')
        if tuple(self.rotation_candidates_deg) != (0.0,) or self.optimize_order:
            raise ValueError('stage one fixes rotation=0 and order=(0,1,2)')
        if not math.isfinite(self.alpha) or self.alpha < 0 or self.planning_substeps < 2:
            raise ValueError('invalid active scan score/integration configuration')


@dataclass(frozen=True)
class SimulationConfig:
    """One reproducible TWC simulation case and its explicit assumptions.

    Configuration of the measured-pattern link, mechanics and causal estimator.
    Mechanical phases organize commands; payload remains active throughout.
    """
    duration_s: float = DEFAULT_DURATION_S
    dt_s: float = DEFAULT_DT_S
    speed_mps: float = 0.0
    initial_distance_m: float = DEFAULT_DISTANCE_M
    initial_offset_deg: float = 0.0
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB
    temperature_k: float = DEFAULT_TEMPERATURE_K
    command_time_s: float = DEFAULT_COMMAND_TIME_S
    stationary_dwell_s: float = 0.0
    quantization_deg: float = DEFAULT_QUANTIZATION_DEG
    scan_loss_db: float = DEFAULT_SCAN_LOSS_DB
    active_scan: ActiveScanConfig = field(default_factory=ActiveScanConfig)
    service_horizon_enabled: bool = False
    quantization_aware_enabled: bool = False
    same_terminal_refresh_interval_s: float | None = None
    decision_diagnostics_enabled: bool = False
    enable_velocity_feedforward: bool = True
    prediction_enabled: bool = True
    actuation_delay_s: float | None = None
    prediction_horizon_commands: float | None = None
    estimator: Literal['iekf', 'energy_map'] = 'iekf'
    feedback: Literal['pi', 'unit'] = 'pi'
    command_limit_deg: float | None = None
    measurement_design_sigma_db: float = EKF_ASSUMED_MEASUREMENT_SIGMA_DB
    q_accel_sigma_db_per_s2: float = EKF_Q_ACCEL_SIGMA_DB_PER_S2
    observation: PowerObservationConfig = field(default_factory=PowerObservationConfig)
    snr_threshold_db: float = 6.0
    service_rate_bps_hz: float = 2.0
    warm_up_fraction: float = 0.25
    energy_samples: int = 256
    angular_psd: float = 0.0025
    q_psd: float = 0.25
    shadow_sigma_db: float = 0.0
    shadow_tau_s: float = 2.0
    pattern_scale: float = 1.0
    q_process_scale: float = 1.0
    r_measurement_scale: float = 1.0
    maneuver: str = 'orthogonal'
    sinusoidal_speed_delta_mps: float = 0.0
    sinusoidal_frequency_hz: float = 0.0
    convergence_lock_fraction: float = 0.95
    convergence_consecutive_cycles: int = 3
    randomize_initial_error_phase: bool = False
    benchmark_runtime: bool = False
    record_stride: int = 1

    @property
    def data_duration_s(self) -> float:
        """Compatibility alias: the stationary dwell without micro-scanning."""
        return self.stationary_dwell_s

    @data_duration_s.setter
    def data_duration_s(self, value: float) -> None:
        self.stationary_dwell_s = float(value)

    def validate(self) -> None:
        """参数合法性检查: 只检查真正会导致错误的内容。"""
        if self.duration_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError('duration_s and dt_s must be positive')
        require_integer_multiple(self.duration_s, self.dt_s, 'duration_s')
        require_integer_multiple(self.command_time_s, self.dt_s, 'command_time_s')
        require_integer_multiple(self.stationary_dwell_s, self.dt_s, 'stationary_dwell_s')
        if self.speed_mps < 0.0 or self.initial_distance_m <= 0.0:
            raise ValueError('speed_mps must be non-negative and distance positive')
        if self.initial_offset_deg < 0.0:
            raise ValueError('initial_offset_deg must be non-negative')
        if self.quantization_deg <= 0.0 or self.command_time_s <= 0.0:
            raise ValueError('mechanical quantization and command time must be positive')
        if self.stationary_dwell_s < 0.0:
            raise ValueError('stationary_dwell_s must be non-negative')
        if self.scan_loss_db <= 0.0:
            raise ValueError('scan_loss_db must be positive')
        if self.actuation_delay_s is not None and self.actuation_delay_s < 0.0:
            raise ValueError('actuation_delay_s must be non-negative')
        if self.prediction_horizon_commands is not None and self.prediction_horizon_commands < 0.0:
            raise ValueError('prediction_horizon_commands must be non-negative')
        if self.measurement_design_sigma_db <= 0.0:
            raise ValueError('measurement_design_sigma_db must be positive')
        if self.q_accel_sigma_db_per_s2 <= 0.0:
            raise ValueError('q_accel_sigma_db_per_s2 must be positive')
        if self.bandwidth_hz <= 0.0 or self.temperature_k <= 0.0:
            raise ValueError('receiver bandwidth and temperature must be positive')
        if self.noise_figure_db < 0.0:
            raise ValueError('receiver noise figure must be non-negative')
        if self.record_stride <= 0:
            raise ValueError('record_stride must be a positive integer')
        if self.q_process_scale <= 0.0 or self.r_measurement_scale <= 0.0:
            raise ValueError('Q/R scales must be positive')
        if self.estimator not in {'iekf', 'energy_map'}:
            raise ValueError("estimator must be 'iekf' or 'energy_map'")
        if self.feedback not in {'pi', 'unit'}:
            raise ValueError("feedback must be 'pi' or 'unit'")
        if self.command_limit_deg is not None and self.command_limit_deg <= 0.0:
            raise ValueError('command_limit_deg must be positive')
        if not 0.0 <= self.warm_up_fraction < 1.0:
            raise ValueError('warm_up_fraction must be in [0, 1)')
        if self.service_rate_bps_hz <= 0.0:
            raise ValueError('service_rate_bps_hz must be positive')
        if self.energy_samples <= 0:
            raise ValueError('energy_samples must be a positive integer')
        if self.angular_psd < 0.0 or self.q_psd < 0.0:
            raise ValueError('angular_psd and q_psd must be non-negative')
        if self.shadow_sigma_db < 0.0 or self.shadow_tau_s <= 0.0:
            raise ValueError('shadow parameters must be valid')
        if self.pattern_scale <= 0.0:
            raise ValueError('pattern_scale must be positive')
        if self.maneuver not in {'orthogonal', 'sinusoidal'}:
            raise ValueError("maneuver must be 'orthogonal' or 'sinusoidal'")
        if self.sinusoidal_speed_delta_mps < 0.0 or self.sinusoidal_frequency_hz < 0.0:
            raise ValueError('sinusoidal motion parameters must be non-negative')
        if not 0.0 < self.convergence_lock_fraction <= 1.0:
            raise ValueError('convergence_lock_fraction must be in (0, 1]')
        if self.convergence_consecutive_cycles <= 0:
            raise ValueError('convergence_consecutive_cycles must be positive')
        self.observation.validate()
        self.active_scan.validate()
        if self.same_terminal_refresh_interval_s is not None:
            if not math.isfinite(self.same_terminal_refresh_interval_s) or self.same_terminal_refresh_interval_s < self.effective_actuation_delay_s:
                raise ValueError('service refresh endpoint must be >= actuation delay')
        if self.quantization_aware_enabled and not self.service_horizon_enabled:
            raise ValueError('quantization-aware control requires service horizon')
        if (self.active_scan.enabled or self.service_horizon_enabled) and (self.estimator != 'energy_map' or self.feedback != 'unit'):
            raise ValueError('new policies require EnergyMAP and unit feedback')

    @property
    def steps(self) -> int:
        """仿真总步数 = duration_s / dt_s。"""
        return require_integer_multiple(self.duration_s, self.dt_s, 'duration_s')

    @property
    def metric_start_s(self) -> float:
        """稳态指标统计起点: 排除仿真开头的暖机时段。

        ``warm_up_fraction`` 默认 0.25, 与连续通信模型的 420 条记录口径一致。
        """
        return self.duration_s * self.warm_up_fraction

    @property
    def effective_actuation_delay_s(self) -> float:
        """状态参考时刻至命令生效的因果时延 (s)。

        基础模型使用标称回中命令时长 ``command_time_s``; 不再使用固定 0.45 s、
        4.5 倍命令时长或 A/B 数据窗中点。样本龄期已在估计器中对齐, 不重复计入。
        遗留字段 ``prediction_horizon_commands`` 仅在未显式给定 ``actuation_delay_s``
        时按旧口径折算 (仅用于旧实验兼容)。
        """
        if self.actuation_delay_s is not None:
            return float(self.actuation_delay_s)
        if self.prediction_horizon_commands is not None:
            return float(self.prediction_horizon_commands) * self.command_time_s
        return float(self.command_time_s)

    @property
    def prediction_horizon_s(self) -> float:
        """Compatibility: effective feed-forward delay when prediction is enabled."""
        return self.effective_actuation_delay_s if self.prediction_enabled else 0.0

@dataclass
class Gimbal:
    """两轴云台: 角度量化、命令转位时间、角速度限制、连续球面插值转动。

    一次 command() 把目标角量化为步进网格后,按最大角速度 GIMBAL_SLEW_LIMIT_DEG_S
    把行程切成 segments 段,每段耗时 command_time_s;advance(dt) 用 slerp 连续推进。
    """
    quantization_deg: float
    command_time_s: float
    az_deg: float
    el_deg: float
    start_vector: np.ndarray = field(init=False)
    target_vector: np.ndarray = field(init=False)
    elapsed_s: float = 0.0
    motion_duration_s: float = 0.0
    moving: bool = False
    segments: int = 1

    def __post_init__(self) -> None:
        vector = direction_from_az_el(self.az_deg, self.el_deg)
        self.start_vector = vector.copy()
        self.target_vector = vector.copy()

    @property
    def direction(self) -> np.ndarray:
        """当前指向的单位方向向量。"""
        return direction_from_az_el(self.az_deg, self.el_deg)

    def quantized_angles(self, az_deg: float, el_deg: float) -> tuple[float, float]:
        """把目标角量化到 quantization_deg 网格 (az 折叠,el 限位 ±90°)。"""
        azimuth = wrap_deg(self.quantization_deg * round(az_deg / self.quantization_deg))
        elevation = float(np.clip(self.quantization_deg * round(el_deg / self.quantization_deg), -GIMBAL_EL_LIMIT_DEG, GIMBAL_EL_LIMIT_DEG))
        return (azimuth, elevation)

    def command(self, az_deg: float, el_deg: float) -> float:
        """下达一次转动命令,返回本次转动总时长 (s)。

        行程不足一个步进量化距离时视为无转动 (moving=False),角度直接跳变到量化值。
        """
        azimuth, elevation = self.quantized_angles(az_deg, el_deg)
        target = direction_from_az_el(azimuth, elevation)
        distance = angular_distance_deg(self.direction, target)
        self.segments = max(1, int(math.ceil(distance / max(GIMBAL_SLEW_LIMIT_DEG_S * self.command_time_s, 1e-12))))
        self.motion_duration_s = self.segments * self.command_time_s
        self.start_vector = self.direction
        self.target_vector = target
        self.elapsed_s = 0.0
        self.moving = distance > 1e-12
        if not self.moving:
            self.az_deg, self.el_deg = (azimuth, elevation)
        return self.motion_duration_s

    def preview(self, az_deg: float, el_deg: float) -> 'MechanicalPreview':
        ghost = copy.deepcopy(self)
        duration = ghost.command(az_deg, el_deg)
        azimuth, elevation = ghost.quantized_angles(az_deg, el_deg)
        return MechanicalPreview(azimuth, elevation, duration, ghost.segments)

    def advance(self, dt_s: float) -> bool:
        """推进转动 dt_s 秒;返回 True 表示转动已完成。

        转动过程中用 slerp 连续插值,结束时精确落到量化目标角。
        """
        if not self.moving:
            return True
        self.elapsed_s += dt_s
        fraction = min(1.0, self.elapsed_s / self.motion_duration_s)
        direction = slerp(self.start_vector, self.target_vector, fraction)
        self.az_deg, self.el_deg = az_el_from_direction(direction)
        if fraction >= 1.0 - 1e-12:
            self.az_deg, self.el_deg = az_el_from_direction(self.target_vector)
            self.moving = False
            return True
        return False

@dataclass
class PaperOrthogonalMotion:
    """论文用的双端正交线性运动最小 benchmark。

    A 端沿 +x 运动,B 端沿 +z 运动(初始相距 d 沿 +y);
    因此两端的相对平移速度为 sqrt(2)·v,视距角速度随时间减小。
    """
    position: np.ndarray
    velocity: np.ndarray
    previous_position: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.previous_position = self.position.copy()

    def step(self, dt_s: float) -> None:
        """按匀速直线运动推进一个步长(记录旧位置用于计算视距角速度)。"""
        self.previous_position = self.position.copy()
        self.position = self.position + self.velocity * dt_s


@dataclass
class SinusoidalVelocityMotion:
    """Minimal maneuver stress test with explicit non-zero acceleration.

    The velocity magnitude along ``direction`` follows
    ``v(t)=v0+delta_v sin(2*pi*f*t)``.  It is deliberately not a random
    trajectory library: the parameters are visible in :class:`SimulationConfig`
    and the model retains the existing ``position/previous_position/step`` API.
    """

    position: np.ndarray
    direction: np.ndarray
    base_speed_mps: float
    speed_delta_mps: float
    frequency_hz: float
    time_s: float = 0.0
    previous_position: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, dtype=float)
        norm = float(np.linalg.norm(self.direction))
        if norm <= 0.0:
            raise ValueError('sinusoidal direction must be non-zero')
        self.direction = self.direction / norm
        self.previous_position = self.position.copy()

    def step(self, dt_s: float) -> None:
        self.previous_position = self.position.copy()
        midpoint_s = self.time_s + 0.5 * dt_s
        speed_mps = self.base_speed_mps + self.speed_delta_mps * math.sin(
            2.0 * math.pi * self.frequency_hz * midpoint_s
        )
        self.position = self.position + self.direction * speed_mps * dt_s
        self.time_s += dt_s

# =============================================================================
# PI + angular-rate prediction controller and six-state IEKF
# =============================================================================

@dataclass
class PIController:
    """两轴 PI 控制器: 死区 + 饱和限幅 + 条件积分抗饱和 (anti-windup)。

    控制律: u = KP·e + KI·∫e dt,输出 clip 到 ±PI_LIMIT_DEG;
    只有当未饱和 (|raw| <= limit) 的轴才累加积分,实现抗饱和。
    """
    integral: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def reset(self) -> None:
        self.integral[:] = 0.0

    def update(self, error_deg: np.ndarray, interval_s: float) -> np.ndarray:
        error = np.asarray(error_deg, dtype=float).copy()
        error[np.abs(error) < PI_DEADBAND_DEG] = 0.0
        proposal = self.integral + error * interval_s
        raw = PI_KP * error + PI_KI * proposal
        bounded = np.clip(raw, -PI_LIMIT_DEG, PI_LIMIT_DEG)
        self.integral[np.abs(raw) <= PI_LIMIT_DEG + 1e-12] = proposal[np.abs(raw) <= PI_LIMIT_DEG + 1e-12]
        return bounded

@dataclass(frozen=True)
class EKFResult:
    """IEKF 更新结果包 (确定性无噪声工况下仅作诊断参考)。"""
    accepted: bool
    normalized_innovation_score: float
    innovation_db: float
    measurement_scale: float

class IEKF:
    """六状态迭代扩展卡尔曼滤波器 (IEKF)。

    状态向量: x = [q, q_dot, e_H, e_E, omega_H, omega_E]^T
      * q       : 慢变链路功率基线 (dBm),包含发射功率、传播损耗、双方峰值增益
                  与远端相对增益,不含本端相对方向图增益;
      * q_dot   : q 的变化率 (dB/s);
      * e_H/e_E : 本端扫描中心在局部 H/E 切平面内的指向误差 (deg);
      * omega_H / omega_E : 视距角速度在局部 H/E 轴上的分量 (deg/s)。

    预测模型: 常数 q 变化率 + 常数角速率 (constant-q-rate / constant-angular-rate)。
    测量模型 (异步三点): z_i = q - q_dot·tau_i + g(e - omega·tau_i - s_i),
      tau_i 为第 i 个样本的真实 age,s_i 为该样本的扫描偏置;
    对应 Jacobian: [1, -tau, dg/dH, dg/dE, -tau·dg/dH, -tau·dg/dE]。
    更新: 迭代重线性化 (IEKF) + 测量设计协方差 + 创新协方差膨胀 + Joseph 协方差更新。
    """

    def __init__(self, pattern: MeasuredAntennaPattern, q0_dbm: float,
                 angular_accel_deg_s2: float,
                 measurement_design_sigma_db: float = EKF_ASSUMED_MEASUREMENT_SIGMA_DB,
                 q_accel_sigma_db_per_s2: float = EKF_Q_ACCEL_SIGMA_DB_PER_S2,
                 q_process_scale: float = 1.0,
                 r_measurement_scale: float = 1.0) -> None:
        self.pattern = pattern
        self.x = np.array([q0_dbm, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.diag([EKF_INITIAL_Q_STD_DBM ** 2, EKF_INITIAL_Q_DOT_STD_DB_PER_S ** 2, EKF_INITIAL_ANGLE_STD_DEG ** 2, EKF_INITIAL_ANGLE_STD_DEG ** 2, EKF_INITIAL_VEL_STD_DEG_S ** 2, EKF_INITIAL_VEL_STD_DEG_S ** 2])
        self.angular_accel_deg_s2 = max(angular_accel_deg_s2, 1e-08)
        self.measurement_design_sigma_db = measurement_design_sigma_db
        self.q_accel_sigma_db_per_s2 = q_accel_sigma_db_per_s2
        self.q_process_scale = q_process_scale
        self.r_measurement_scale = r_measurement_scale
        self.last_score = 0.0

    @property
    def q(self) -> float:
        return float(self.x[0])

    @property
    def q_dot(self) -> float:
        return float(self.x[1])

    @property
    def error(self) -> np.ndarray:
        """指向误差估计 [e_H, e_E] (deg)。"""
        return self.x[2:4].copy()

    @property
    def velocity(self) -> np.ndarray:
        """视距角速度估计 [omega_H, omega_E] (deg/s)。"""
        return self.x[4:6].copy()

    def predict(self, dt_s: float) -> None:
        """预测步: 常数 q 变化率 + 常数角速率模型。

        x ← Φ·x, 其中 Φ[0,1] = dt, Φ[2,4] = dt, Φ[3,5] = dt;
        P ← Φ·P·Φ^T + Q(离散设计协方差)。
        """
        transition = np.eye(6)
        transition[0, 1] = dt_s
        transition[2, 4] = dt_s
        transition[3, 5] = dt_s
        self.x = transition @ self.x
        process = np.zeros((6, 6), dtype=float)
        q_block = self.q_accel_sigma_db_per_s2 ** 2 * np.array([[dt_s ** 4 / 4.0, dt_s ** 3 / 2.0], [dt_s ** 3 / 2.0, dt_s ** 2]]) * EKF_PROCESS_SCALE * self.q_process_scale
        process[np.ix_([0, 1], [0, 1])] = q_block
        acceleration_squared = self.angular_accel_deg_s2 ** 2
        block = acceleration_squared * np.array([[dt_s ** 4 / 4.0, dt_s ** 3 / 2.0], [dt_s ** 3 / 2.0, dt_s ** 2]]) * EKF_PROCESS_SCALE * self.q_process_scale
        process[np.ix_([2, 4], [2, 4])] = block
        process[np.ix_([3, 5], [3, 5])] = block
        self.p = transition @ self.p @ transition.T + process
        self.p = 0.5 * (self.p + self.p.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError('IEKF prediction produced NaN or Inf')
        if self.minimum_eigenvalue() < -1e-09:
            raise FloatingPointError('IEKF prediction covariance is not PSD')

    def _state_transform(self, state: np.ndarray, old_base_az_deg: float, old_base_el_deg: float, new_base_az_deg: float, new_base_el_deg: float) -> np.ndarray:
        """把单个状态向量从旧基准切平面变换到新基准切平面 (q、q_dot 不变)。"""
        los_az, los_el = offset_direction(old_base_az_deg, old_base_el_deg, state[2], state[3])
        new_error = local_error_deg(new_base_az_deg, new_base_el_deg, direction_from_az_el(los_az, los_el))
        _, old_h, old_e = tangent_basis(old_base_az_deg, old_base_el_deg)
        _, new_h, new_e = tangent_basis(new_base_az_deg, new_base_el_deg)
        angular_velocity_world = math.radians(state[4]) * old_h + math.radians(state[5]) * old_e
        new_velocity = np.array([math.degrees(float(np.dot(angular_velocity_world, new_h))), math.degrees(float(np.dot(angular_velocity_world, new_e)))])
        return np.array([state[0], state[1], new_error[0], new_error[1], new_velocity[0], new_velocity[1]], dtype=float)

    def transform_reference_frame(self, old_base_az_deg: float, old_base_el_deg: float, new_base_az_deg: float, new_base_el_deg: float) -> None:
        """recenter 时把状态与协方差变换到新的扫描中心切平面。

        状态用解析变换;协方差用数值有限差分 Jacobian 传播 P ← J·P·J^T。
        """
        old_state = self.x.copy()
        transformed = self._state_transform(old_state, old_base_az_deg, old_base_el_deg, new_base_az_deg, new_base_el_deg)
        jacobian = np.zeros((6, 6), dtype=float)
        for index in range(6):
            step = 1e-05 if index in (0, 1) else 1e-06
            plus = old_state.copy()
            minus = old_state.copy()
            plus[index] += step
            minus[index] -= step
            forward = self._state_transform(plus, old_base_az_deg, old_base_el_deg, new_base_az_deg, new_base_el_deg)
            backward = self._state_transform(minus, old_base_az_deg, old_base_el_deg, new_base_az_deg, new_base_el_deg)
            jacobian[:, index] = (forward - backward) / (2.0 * step)
        self.x = transformed
        self.p = jacobian @ self.p @ jacobian.T
        self.p = 0.5 * (self.p + self.p.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise FloatingPointError('reference-frame transform produced NaN or Inf')
        if float(np.linalg.eigvalsh(self.p).min()) < -1e-09:
            raise ValueError('reference-frame covariance lost positive semidefiniteness')

    def recenter_transform(self, old_base_az_deg: float, old_base_el_deg: float,
                           new_base_az_deg: float, new_base_el_deg: float) -> None:
        """Public IEKF reference-frame transform used after a mechanical recenter."""
        self.transform_reference_frame(old_base_az_deg, old_base_el_deg,
                                       new_base_az_deg, new_base_el_deg)

    def measurement_model(self, scan_offset_deg: np.ndarray, age_s: float) -> tuple[float, np.ndarray]:
        """异步三点测量模型: 返回 (预测测量值, 6 维 Jacobian)。

        重建样本 age=tau 时刻的历史状态:
          历史指向误差 = e - omega·tau - s      (s = 扫描偏置)
          测量 z = q - q_dot·tau + g(历史指向误差)
        Jacobian = [1, -tau, dg/dH, dg/dE, -tau·dg/dH, -tau·dg/dE]
        (g 为该点实测方向图的局部相对增益,梯度由 PCHIP 导数给出)
        """
        return pointing_measurement(self.x, self.pattern, scan_offset_deg, age_s)

    def measurement_covariance(self, sample_count: int) -> np.ndarray:
        """Return the design R with the documented covariance-scale convention."""
        if sample_count <= 0:
            raise ValueError('sample_count must be positive')
        return (
            np.eye(sample_count)
            * self.measurement_design_sigma_db ** 2
            * self.r_measurement_scale
        )

    def update(self, samples: list[tuple[np.ndarray, float, float]]) -> EKFResult:
        """Update from the three real asynchronous scan samples ``(s, z, age)``."""
        return self.batch_update(samples)

    def batch_update(self, samples: list[tuple[np.ndarray, float, float]]) -> EKFResult:
        """三样本 IEKF 批量更新 (设计协方差 + 创新膨胀 + Joseph 协方差更新)。

        samples: [(scan_offset_deg, measurement_dbm, age_s), ...]
        步骤:
          1. 在迭代前的先验状态处计算三样本 Jacobian 与预测,求归一化新息平方 score;
          2. 若 score 超过设计阈值,则按 score/阈值 (封顶 INNOVATION_INFLATION_CAP)
             膨胀测量协方差 R;
          3. 迭代 IEKF_ITERATIONS 次重线性化更新状态;
          4. Joseph 形式更新协方差 P = (I-KH)P(I-KH)^T + K·R·K^T;
          5. 任何数值失败 (非有限 / 非半正定) 都回滚到先验并返回 rejected。
        """
        if not samples:
            return EKFResult(False, math.inf, 0.0, 1.0)
        prior_x = self.x.copy()
        prior_p = self.p.copy()
        observations = np.asarray([sample[1] for sample in samples], dtype=float)
        self.x = prior_x.copy()
        predictions: list[float] = []
        jacobians: list[np.ndarray] = []
        for scan_offset, _, age_s in samples:
            predicted, jacobian = self.measurement_model(scan_offset, age_s)
            predictions.append(predicted)
            jacobians.append(jacobian)
        matrix = np.vstack(jacobians)
        residual = observations - np.asarray(predictions)
        # ``r_measurement_scale`` scales the covariance itself: R' = r_s R.
        # This keeps the experiment label mathematically aligned with the
        # usual Q/R-sensitivity convention rather than scaling the standard
        # deviation twice.
        base_r = self.measurement_covariance(len(samples))
        innovation_covariance = matrix @ prior_p @ matrix.T + base_r
        try:
            score = float(residual @ np.linalg.solve(innovation_covariance, residual))
        except np.linalg.LinAlgError:
            self.x, self.p = (prior_x, prior_p)
            return EKFResult(False, math.inf, float(np.max(np.abs(residual))), 1.0)
        if not math.isfinite(score):
            self.x, self.p = (prior_x, prior_p)
            return EKFResult(False, score, float(np.max(np.abs(residual))), 1.0)
        scale = min(INNOVATION_INFLATION_CAP, max(1.0, score / INNOVATION_CONSISTENCY_DESIGN_THRESHOLD))
        effective_r = base_r * scale
        current_x = prior_x.copy()
        last_matrix = matrix
        for _ in range(IEKF_ITERATIONS):
            self.x = current_x
            iteration_predictions: list[float] = []
            iteration_jacobians: list[np.ndarray] = []
            for scan_offset, _, age_s in samples:
                predicted, jacobian = self.measurement_model(scan_offset, age_s)
                iteration_predictions.append(predicted)
                iteration_jacobians.append(jacobian)
            matrix = np.vstack(iteration_jacobians)
            innovation = observations - np.asarray(iteration_predictions) + matrix @ (current_x - prior_x)
            covariance = matrix @ prior_p @ matrix.T + effective_r
            try:
                gain = np.linalg.solve(covariance, matrix @ prior_p).T
            except np.linalg.LinAlgError:
                self.x, self.p = (prior_x, prior_p)
                return EKFResult(False, score, float(np.max(np.abs(innovation))), scale)
            updated = prior_x + gain @ innovation
            last_matrix = matrix
            if np.linalg.norm(updated - current_x) < 1e-07:
                current_x = updated
                break
            current_x = updated
        self.x = current_x
        covariance = last_matrix @ prior_p @ last_matrix.T + effective_r
        gain = np.linalg.solve(covariance, last_matrix @ prior_p).T
        identity = np.eye(6)
        kh = gain @ last_matrix
        self.p = (identity - kh) @ prior_p @ (identity - kh).T + gain @ effective_r @ gain.T
        self.p = 0.5 * (self.p + self.p.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            self.x, self.p = (prior_x, prior_p)
            return EKFResult(False, score, float(np.max(np.abs(residual))), scale)
        if self.minimum_eigenvalue() < -1e-09:
            self.x, self.p = (prior_x, prior_p)
            return EKFResult(False, score, float(np.max(np.abs(residual))), scale)
        self.last_score = score
        return EKFResult(True, score, float(np.max(np.abs(residual))), scale)

    def minimum_eigenvalue(self) -> float:
        """协方差矩阵的最小特征值 (PSD 检查用)。"""
        return float(np.linalg.eigvalsh(0.5 * (self.p + self.p.T)).min())


class EnergyMAP(IEKF):
    """Six-state posterior mode under a Gamma energy likelihood.

    With total energy ``E/noise ~ Gamma(M, (1+SNR)/M)`` the signed square root
    of the likelihood deviance is a least-squares residual.  Optimisation is
    damped and restricted to the declared local state domain.  This is the
    continuous-communication estimator option (guide section 7); the recursive
    MAP fuses the historical state prior with a new batch of power/energy
    observations.  ``q_dot`` is retained as a disturbance state and is not a
    headline performance contributor.
    """

    def configure(self, noise_dbm: float, angular_psd: float, q_psd: float,
                  five_state: bool = False) -> None:
        self.noise_dbm = noise_dbm
        self.angular_psd = angular_psd
        self.q_psd = q_psd
        self.five_state = five_state
        self.energy_batch: list[tuple[float, int]] = []
        self.cost_pairs: list[tuple[float, float]] = []
        self.nfev: list[int] = []
        bw = 0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg)
        self.x = np.array([noise_dbm + 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.diag([20.0 ** 2, 0.5 ** 2, (0.25 * bw) ** 2, (0.25 * bw) ** 2,
                          (0.3 * bw) ** 2, (0.3 * bw) ** 2])
        if five_state:
            self.p[1, 1] = 1e-10

    def predict(self, dt_s: float) -> None:
        """Causal state propagation shared by both prediction groups."""
        transition = np.eye(6)
        transition[0, 1] = dt_s
        transition[2, 4] = dt_s
        transition[3, 5] = dt_s
        process = np.zeros((6, 6), dtype=float)
        block = np.array([[dt_s ** 3 / 3.0, dt_s ** 2 / 2.0], [dt_s ** 2 / 2.0, dt_s]])
        for indices in ([2, 4], [3, 5]):
            process[np.ix_(indices, indices)] = self.angular_psd * block
        process[0, 0] = self.q_psd * dt_s
        if not self.five_state:
            process[np.ix_([0, 1], [0, 1])] += 0.01 * block
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process
        if self.five_state:
            self.x[1] = 0.0
            self.p[1, :] = 0.0
            self.p[:, 1] = 0.0
            self.p[1, 1] = 1e-10
        self.p = 0.5 * (self.p + self.p.T)

    def batch_update(self, samples: list[tuple[np.ndarray, float, float]]) -> EKFResult:
        """Batch posterior-mode update from the aligned energy samples.

        The energy likelihood uses the Gamma ``(M, (1+SNR)/M)`` deviance; the
        optimiser is damped and box-constrained to the local state domain.
        Rejected updates retain the causal prior and propagation, never the
        true angle.
        """
        start = time.perf_counter()
        prior = self.x.copy()
        cov = self.p.copy()
        energies = self.energy_batch[-len(samples):]
        if len(energies) != len(samples):
            raise ValueError('Energy/sample alignment failed')
        self.energy_batch = []
        energy = np.array([value[0] for value in energies])
        m = np.array([value[1] for value in energies])
        inv_l = solve_triangular(np.linalg.cholesky(cov + 1e-10 * np.eye(6)), np.eye(6), lower=True)

        def measurements(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.x = x
            values = [self.measurement_model(sample[0], sample[2]) for sample in samples]
            z = np.array([value[0] for value in values])
            h = np.array([value[1] for value in values])
            snr = 10.0 ** np.clip((z - self.noise_dbm) / 10.0, -30.0, 15.0)
            mu = 1.0 + snr
            r = energy / mu
            d = np.maximum(r - 1.0 - np.log(r), 0.0)
            residual = np.sign(1.0 - r) * np.sqrt(2.0 * m * d)
            factor = np.ones_like(r)
            mask = np.abs(r - 1.0) > 1e-5
            factor[mask] = np.abs(r[mask] - 1.0) / np.sqrt(2.0 * np.maximum(d[mask], 1e-30))
            jac = np.sqrt(m)[:, None] * factor[:, None] * (math.log(10.0) / 10.0 * snr / mu)[:, None] * h
            return residual, jac

        def fun(x: np.ndarray) -> np.ndarray:
            residual, _ = measurements(x)
            return np.r_[inv_l @ (x - prior), residual]

        def jac(x: np.ndarray) -> np.ndarray:
            _, j = measurements(x)
            return np.vstack([inv_l, j])

        bw = 0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg)
        lo = np.array([-180.0, -20.0, -2 * bw, -2 * bw, -4 * bw, -4 * bw])
        hi = np.array([60.0, 20.0, 2 * bw, 2 * bw, 4 * bw, 4 * bw])
        x0 = np.clip(prior, lo + 1e-8, hi - 1e-8)
        before = float(np.sum(fun(x0) ** 2))
        fit = self.solve_posterior(fun, jac, x0, (lo, hi))
        after = float(np.sum(fun(fit.x) ** 2))
        self.cost_pairs.append((before, after))
        self.nfev.append(fit.nfev)
        if not np.isfinite(after) or after > before + 1e-7:
            self.x = prior
            self.p = cov
            return EKFResult(False, before, 0.0, 1.0)
        self.x = fit.x
        # Expected Gamma information, avoiding data-dependent curvature.
        h = gamma_information_rows(self.measurement_model, samples, self.noise_dbm, m)
        k = np.linalg.solve(np.eye(len(samples)) + h @ cov @ h.T, h @ cov).T
        a = np.eye(6) - k @ h
        self.p = a @ cov @ a.T + k @ k.T
        self.p = 0.5 * (self.p + self.p.T)
        if self.five_state:
            self.x[1] = 0.0
        return EKFResult(True, after, float(np.max(np.abs(measurements(fit.x)[0]))), 1.0)

    def update(self, samples: list[tuple[np.ndarray, float, float]]) -> EKFResult:
        return self.batch_update(samples)

    def solve_posterior(self, fun, jac, x0, bounds):
        """Legacy optimizer, unchanged; optional research subclasses may globalize.

        The Gamma objective, prior, box, covariance and random streams remain
        owned by batch_update. Default execution preserves the original call.
        """
        return least_squares(fun, x0, jac=jac, bounds=bounds, max_nfev=24,
                             ftol=1e-6, xtol=1e-6, gtol=1e-6)


# =============================================================================
# Fair-comparison tracking policies
# =============================================================================


def pointing_measurement(x, pattern, scan_offset_deg, age_s):
    residual = x[2:4] - x[4:6] * age_s - scan_offset_deg
    gain = pattern.local_relative_gain(float(residual[0]), float(residual[1]))
    gradient = pattern.local_gradient(float(residual[0]), float(residual[1]))
    jacobian = np.zeros(6, dtype=float)
    jacobian[0] = 1.0
    jacobian[1] = -age_s
    jacobian[2:4] = gradient
    jacobian[4:6] = -age_s * gradient
    return (x[0] - x[1] * age_s + gain, jacobian)

def gamma_information_rows(measurement_model, samples, noise_dbm, sample_counts):
    """Single source for EnergyMAP covariance and candidate expected information.

    I_i = M_i (ln(10)/10 SNR_i/(1+SNR_i))^2 h_i h_i^T.
    Preserve the legacy floating-point operation order and SNR clipping.
    """
    hs: list[np.ndarray] = []
    ws: list[float] = []
    for offset, _, age_s in samples:
        z, h = measurement_model(offset, age_s)
        snr = 10.0 ** np.clip((z - noise_dbm) / 10.0, -30.0, 15.0)
        hs.append(h)
        ws.append(math.log(10.0) / 10.0 * snr / (1.0 + snr))
    return np.array(hs) * np.sqrt(sample_counts)[:, None] * np.array(ws)[:, None]


def expected_gamma_information(x_prior, p_prior, offsets_real, sample_ages,
                               pattern, noise_dbm, sample_count):
    samples = [(offset, 0.0, age) for offset, age in zip(offsets_real, sample_ages)]
    if len(samples) != len(offsets_real) or len(samples) != len(sample_ages) or not samples:
        raise ValueError('unaligned expected-information samples')
    counts = np.broadcast_to(np.asarray(sample_count), (len(samples),))
    rows = gamma_information_rows(
        lambda offset, age: pointing_measurement(x_prior, pattern, offset, age),
        samples, noise_dbm, counts)
    information = rows.T @ rows
    posterior = np.linalg.solve(np.linalg.solve(p_prior, np.eye(6)) + information, np.eye(6))
    return information, 0.5 * (posterior + posterior.T)


def pointing_projection(tau_s: float) -> np.ndarray:
    """G(tau) x = e + tau omega; q and q_dot are nuisance states."""
    matrix = np.zeros((2, 6))
    matrix[0, 2], matrix[1, 3] = 1.0, 1.0
    matrix[0, 4], matrix[1, 5] = tau_s, tau_s
    return matrix


def future_pointing_covariance(p, tau_s, hpbw_deg=None):
    matrix = pointing_projection(tau_s)
    covariance = matrix @ p @ matrix.T
    if hpbw_deg is None:
        return covariance
    scales = np.asarray(hpbw_deg, dtype=float)
    if scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError('two positive H/E beamwidths are required')
    return covariance, covariance / np.outer(scales, scales)


def service_horizon(actuation_delay_s, same_terminal_refresh_interval_s):
    if actuation_delay_s < 0 or same_terminal_refresh_interval_s < actuation_delay_s:
        raise ValueError('service interval must have 0 <= tau_a <= T_f')
    return 0.5 * (actuation_delay_s + same_terminal_refresh_interval_s)


def frozen_array(value):
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MechanicalPreview:
    az_deg: float
    el_deg: float
    duration_s: float
    segments: int


@dataclass(frozen=True)
class MechanicalState:
    """Encoder state only; excludes terminal motion and simulation references."""
    az_deg: float
    el_deg: float
    quantization_deg: float
    command_time_s: float

    def gimbal(self):
        return Gimbal(self.quantization_deg, self.command_time_s, self.az_deg, self.el_deg)


@dataclass(frozen=True)
class DecisionContext:
    x: np.ndarray
    p: np.ndarray
    base_az_el: tuple[float, float]
    mechanics: MechanicalState
    noise_dbm: float
    energy_samples: int
    dt_s: float
    actuation_delay_s: float
    same_terminal_refresh_interval_s: float
    angular_psd: float
    q_psd: float
    command_limit_deg: float
    active_scan: ActiveScanConfig

    def __post_init__(self):
        object.__setattr__(self, 'x', frozen_array(self.x))
        object.__setattr__(self, 'p', frozen_array(self.p))


@dataclass(frozen=True)
class ScanDesign:
    commanded_offsets_deg: tuple[np.ndarray, ...]
    realized_offsets_deg: tuple[np.ndarray, ...]
    scan_loss_db: float
    rotation_deg: float = 0.0
    permutation: tuple[int, ...] = (0, 1, 2)
    predicted_sample_ages_s: tuple[float, ...] = ()
    command_durations_s: tuple[float, ...] = ()
    expected_information: np.ndarray | None = None
    expected_posterior: np.ndarray | None = None
    predicted_future_covariance: np.ndarray | None = None
    normalized_future_covariance: np.ndarray | None = None
    predicted_scan_se: float = 0.0
    predicted_objective: float = 0.0

    def __post_init__(self):
        for name in ('commanded_offsets_deg', 'realized_offsets_deg'):
            object.__setattr__(self, name, tuple(frozen_array(x) for x in getattr(self, name)))
        for name in ('expected_information', 'expected_posterior', 'predicted_future_covariance',
                     'normalized_future_covariance'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, frozen_array(getattr(self, name)))


def preview_scan(context, offsets):
    """Run a private encoder clone on the scheduler clock, including slew segments.

    Sample ages are accumulated by the same dt additions as Tracker.predict.
    Returned path includes the continuous spherical motion at every physics tick.
    """
    ghost = context.mechanics.gimbal()
    realized, durations, ages, path = [], [], [], []
    elapsed = 0.0
    for offset in offsets:
        target = offset_direction(*context.base_az_el, *offset)
        duration = ghost.command(*ghost.quantized_angles(*target))
        durations.append(duration)
        phase_elapsed = 0.0
        while phase_elapsed + 1e-12 < max(duration, context.dt_s):
            ghost.advance(context.dt_s)
            ages = [age + context.dt_s for age in ages]
            phase_elapsed += context.dt_s
            elapsed += context.dt_s
            path.append((elapsed, np.asarray(local_error_deg(*context.base_az_el, ghost.direction))))
        realized.append(np.asarray(local_error_deg(*context.base_az_el, ghost.direction)))
        ages.append(0.0)
    return tuple(realized), tuple(durations), tuple(ages), path


def predicted_se(x, pattern, noise_dbm, tau, offset):
    residual = x[2:4] + tau * x[4:6] - offset
    loss, _, clamped = pattern.loss_and_gradient(*residual)
    if clamped:
        return 0.0
    snr_db = x[0] + tau * x[1] - loss - noise_dbm
    return float(np.logaddexp(0.0, snr_db * math.log(10.0) / 10.0) / math.log(2.0))


def predicted_integrated_se(context, pattern, realized_offset, start_s=None, end_s=None):
    """Deterministic midpoint integral; only estimated q/q_dot/e/omega are read."""
    start = context.actuation_delay_s if start_s is None else start_s
    end = context.same_terminal_refresh_interval_s if end_s is None else end_s
    if end <= start:
        return 0.0
    count = context.active_scan.planning_substeps
    step = (end - start) / count
    return step * sum(predicted_se(context.x, pattern, context.noise_dbm,
                                  start + (i + 0.5) * step, realized_offset)
                      for i in range(count))


def preview_recenter(context, correction):
    ghost = context.mechanics.gimbal()
    target = offset_direction(*context.base_az_el, *correction)
    return ghost.preview(*ghost.quantized_angles(*target))


def quantized_service_command(context, pattern, closed_form):
    """3x3 AZ/EL encoder neighbours, deduplicated at wrapping/elevation limits.

    The center is considered first so zero-duration ties preserve the latency
    target. Candidate-specific segment durations define their service start.
    """
    ghost = context.mechanics.gimbal()
    center = preview_recenter(context, closed_form)
    candidates, seen = [], set()
    for dh, de in ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        preview = ghost.preview(center.az_deg + dh * ghost.quantization_deg,
                                center.el_deg + de * ghost.quantization_deg)
        target = (preview.az_deg, preview.el_deg)
        if target in seen:
            continue
        seen.add(target)
        realized = np.asarray(local_error_deg(*context.base_az_el, direction_from_az_el(*target)))
        # A candidate is feasible only if a bounded local request can quantize
        # to this encoder target, including at wrap/elevation boundaries.
        request = np.clip(np.degrees(np.tan(np.radians(realized))),
                          -context.command_limit_deg, context.command_limit_deg)
        if ghost.quantized_angles(*offset_direction(*context.base_az_el, *request)) != target:
            continue
        start = max(context.actuation_delay_s, preview.duration_s)
        value = predicted_integrated_se(context, pattern, realized, start_s=start)
        candidates.append((value, target, realized, preview))
    if not candidates:
        raise ValueError('no mechanically feasible service candidates')
    return max(candidates, key=lambda item: item[0])


def prepare_scan_design(context, pattern, loss_db, offsets=None, score=True):
    offsets = pattern.three_point_design(loss_db, 0.0, (0, 1, 2)) if offsets is None else offsets
    realized, durations, ages, path = preview_scan(context, offsets)
    if not score:
        return ScanDesign(tuple(offsets), realized, loss_db,
                          predicted_sample_ages_s=ages, command_durations_s=durations)
    # Propagate a private EnergyMAP prior with the existing process model, once
    # per predicted sample-clock tick. Never run an update or sample an RNG.
    prior = EnergyMAP(pattern, context.x[0], 0.0)
    prior.configure(context.noise_dbm, context.angular_psd, context.q_psd)
    prior.x, prior.p = context.x.copy(), context.p.copy()
    for _ in path:
        prior.predict(context.dt_s)
    information, posterior = expected_gamma_information(
        prior.x, prior.p, realized, ages, pattern, context.noise_dbm, context.energy_samples)
    tau = service_horizon(context.actuation_delay_s, context.same_terminal_refresh_interval_s)
    covariance, normalized = future_pointing_covariance(
        posterior, tau, (pattern.qa.h_hpbw_deg, pattern.qa.e_hpbw_deg))
    scan_se = context.dt_s * sum(predicted_se(context.x, pattern, context.noise_dbm, t, offset)
                                 for t, offset in path)
    center_se = context.dt_s * sum(predicted_se(context.x, pattern, context.noise_dbm, t, np.zeros(2))
                                   for t, _ in path)
    # Dimensionless normalized tradeoff: fractional payload loss versus the
    # future H/E pointing variance in squared HPBW units. alpha is manifested.
    communication_cost = (center_se - scan_se) / max(center_se, 1e-12)
    objective = -communication_cost - context.active_scan.alpha * float(np.trace(normalized))
    return ScanDesign(tuple(offsets), realized, loss_db, predicted_sample_ages_s=ages,
                      command_durations_s=durations, expected_information=information,
                      expected_posterior=posterior, predicted_future_covariance=covariance,
                      normalized_future_covariance=normalized, predicted_scan_se=scan_se,
                      predicted_objective=objective)


def select_active_scan(context, pattern):
    designs = [prepare_scan_design(context, pattern, loss) for loss in context.active_scan.loss_candidates_db]
    return max(designs, key=lambda design: design.predicted_objective)


@dataclass(frozen=True)
class PolicyUpdate:
    """A policy decision after the identical three-sample measurement budget."""

    correction_deg: np.ndarray
    iekf_result: EKFResult | None
    iekf_update_time_ms: float = 0.0
    controller_time_ms: float = 0.0
    quantized_target_az_el: tuple[float, float] | None = None
    decision_diagnostics: dict[str, Any] | None = None


class TrackingPolicy(Protocol):
    """Small strategy interface used by all non-oracle comparisons."""

    name: str
    uses_tracking_scheduler: bool
    is_oracle_upper_bound: bool
    tracking_measurements_per_update: int
    uses_sample_age: bool
    estimates_angular_rate: bool
    uses_prediction: bool

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        ...

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        ...


class _AgeAwarePolicy:
    """Shared non-public mechanics for the three IEKF policies.

    The control law is unified across the prediction switch (guide section 6):

        feedback    = feedback_controller(estimated_error, interval_s)
        feedforward = estimated_angular_rate * actuation_delay_s if prediction else 0
        correction  = clip(feedback + feedforward, -command_limit_deg, +limit)

    Both groups share the same estimator, state propagation, angular-rate
    estimate, priors, observation noise, micro-scan, initialisation, clipping
    and quantisation.  ``prediction_delay_s`` is now the causal delay from the
    state reference time to command effect; sample ages are already aligned
    inside the estimator and are not added a second time.
    """

    name = 'age_aware'
    uses_tracking_scheduler = True
    is_oracle_upper_bound = False
    tracking_measurements_per_update = 3
    uses_sample_age = True
    estimates_angular_rate = True
    uses_prediction = True
    force_synchronous_ages = False
    feedback_kind = 'pi'
    command_limit_deg = PI_LIMIT_DEG

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        estimator_samples = samples
        if self.force_synchronous_ages:
            estimator_samples = [(offset, observed_dbm, 0.0) for offset, observed_dbm, _ in samples]
        update_started = time.perf_counter()
        update = tracker.filter.update(estimator_samples)
        update_time_ms = (time.perf_counter() - update_started) * 1e3
        controller_started = time.perf_counter()
        if self.feedback_kind == 'unit':
            feedback = np.asarray(tracker.estimate_error, dtype=float).copy()
        else:
            feedback = tracker.controller.update(tracker.estimate_error, interval_s)
        feedforward = (
            np.asarray(tracker.estimate_velocity, dtype=float) * prediction_delay_s
            if self.uses_prediction else np.zeros(2, dtype=float)
        )
        return PolicyUpdate(
            np.clip(feedback + feedforward, -self.command_limit_deg, self.command_limit_deg), update,
            update_time_ms, (time.perf_counter() - controller_started) * 1e3,
        )

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index
        return mechanical_direction


class ProposedAgeAwarePredictive(_AgeAwarePolicy):
    """Proposed age-aware IEKF with PI plus angular-rate prediction."""

    name = 'proposed_age_aware_predictive'
    uses_prediction = True


class AgeAwareNoPrediction(_AgeAwarePolicy):
    """Ablation retaining age-aware IEKF but removing angular-rate prediction."""

    name = 'age_aware_no_prediction'
    uses_prediction = False


class MotionPredictivePolicy(_AgeAwarePolicy):
    """Power-observation MAP (EnergyMAP) with unit feedback and motion prediction.

    This is the continuous-communication main method: the same estimator for
    both groups, the difference is only the angular-rate feed-forward term
    ``tau * omega_hat`` with ``tau`` equal to the nominal recenter command
    latency.  The per-command magnitude is limited to one measured beamwidth
    by default; ``command_limit_deg`` and ``feedback_kind`` are resolved once
    at simulation construction from the effective configuration and are never
    re-derived inside the update.
    """

    name = 'motion_prediction'
    feedback_kind = 'unit'
    uses_prediction = True
    command_limit_deg = None  # resolved to one beamwidth at construction

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        if self.command_limit_deg is None:
            self.command_limit_deg = 0.5 * (tracker.pattern.qa.h_hpbw_deg + tracker.pattern.qa.e_hpbw_deg)
        return super().on_three_samples(tracker, samples, interval_s, prediction_delay_s)


class MotionNoPredictionPolicy(MotionPredictivePolicy):
    """Paired ablation: identical MAP estimator and unit feedback, no feed-forward."""

    name = 'motion_no_prediction'
    uses_prediction = False


class ContextPointingPolicy(MotionPredictivePolicy):
    """Controller boundary contains no Tracker, Terminal, motion or truth object."""
    active_scan_enabled = False
    service_horizon = False

    def __init__(self, quantization_aware=False):
        self.quantization_aware = quantization_aware

    def on_three_samples(self, *args, **kwargs):
        raise RuntimeError('context policies require scheduler-supplied DecisionContext')

    def prepare_scan_plan(self, context, pattern, fixed_loss_db):
        if self.active_scan_enabled:
            return select_active_scan(context, pattern)
        return prepare_scan_design(context, pattern, fixed_loss_db)

    def decide(self, context: DecisionContext, pattern: MeasuredAntennaPattern) -> PolicyUpdate:
        started = time.perf_counter()
        latency = context.x[2:4] + context.actuation_delay_s * context.x[4:6]
        tau = service_horizon(context.actuation_delay_s, context.same_terminal_refresh_interval_s)
        service = context.x[2:4] + tau * context.x[4:6]
        correction = np.clip(service if self.service_horizon else latency,
                             -context.command_limit_deg, context.command_limit_deg)
        target = None
        preview = preview_recenter(context, correction)
        realized = np.asarray(local_error_deg(*context.base_az_el,
                              direction_from_az_el(preview.az_deg, preview.el_deg)))
        integrated = predicted_integrated_se(context, pattern, realized,
                                             start_s=max(context.actuation_delay_s, preview.duration_s))
        if self.quantization_aware:
            integrated, target, realized, preview = quantized_service_command(context, pattern, correction)
            # Invert offset_direction's tangent construction for the display
            # correction; actual execution uses the exact encoder target.
            correction = np.clip(np.degrees(np.tan(np.radians(realized))),
                                 -context.command_limit_deg, context.command_limit_deg)
        covariance, normalized = future_pointing_covariance(
            context.p, tau, (pattern.qa.h_hpbw_deg, pattern.qa.e_hpbw_deg))
        diagnostics = dict(service_horizon_s=tau, actuation_delay_s=context.actuation_delay_s,
                           same_terminal_refresh_interval_s=context.same_terminal_refresh_interval_s,
                           latency_target=latency.tolist(), service_target=service.tolist(),
                           selected_quantized_command=realized.tolist(),
                           selected_quantized_az_el=[preview.az_deg, preview.el_deg],
                           predicted_command_duration_s=preview.duration_s,
                           predicted_integrated_se=integrated,
                           future_covariance=covariance.tolist(), normalized_future_covariance=normalized.tolist())
        return PolicyUpdate(correction, None, controller_time_ms=(time.perf_counter() - started) * 1e3,
                            quantized_target_az_el=target, decision_diagnostics=diagnostics)


class FixedServiceHorizonPolicy(ContextPointingPolicy):
    """Fixed three-point scan, u=e+((tau_a+T_f)/2)omega."""
    name = 'fixed_service_horizon'
    service_horizon = True


class ActiveLatencyPolicy(ContextPointingPolicy):
    """Active radius selection with the unchanged unit latency controller."""
    name = 'active_latency'
    active_scan_enabled = True


class ActiveServiceHorizonPolicy(FixedServiceHorizonPolicy):
    """Joint active probing and optional quantization-aware service pointing."""
    name = 'active_service_horizon'
    active_scan_enabled = True

    def __init__(self, quantization_aware=True):
        super().__init__(quantization_aware)


class SynchronousThreePointBaseline(_AgeAwarePolicy):
    """Conventional estimator that incorrectly treats all three samples as tau=0."""

    name = 'synchronous_three_point'
    force_synchronous_ages = True
    uses_prediction = False
    uses_sample_age = False


class FixedHorizonPredictionBaseline(SynchronousThreePointBaseline):
    """Conventional synchronous IEKF with one non-adaptive prediction delay.

    It is used only in the prediction-horizon ablation.  Unlike the proposed
    method it discards asynchronous sample ages and keeps its controller
    horizon fixed while the proposed horizon is swept from the CSV condition.
    """

    name = 'synchronous_fixed_horizon_prediction'
    uses_prediction = True

    def __init__(self, fixed_prediction_horizon_s: float = 0.45) -> None:
        if fixed_prediction_horizon_s < 0.0:
            raise ValueError('fixed_prediction_horizon_s must be non-negative')
        self.fixed_prediction_horizon_s = float(fixed_prediction_horizon_s)

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        return super().on_three_samples(
            tracker, samples, interval_s, self.fixed_prediction_horizon_s
        )


class LocalThreePointNoEstimator:
    """Three-point local power search without an estimator or prediction.

    This baseline receives exactly the same three measured powers and consumes
    the same mechanical scan/recenter resources as the IEKF methods.  Its
    correction is only a power-weighted local centroid; it has no state,
    angular-rate estimate, or feed-forward term.
    """

    name = 'local_three_point_no_estimator'
    uses_tracking_scheduler = True
    is_oracle_upper_bound = False
    tracking_measurements_per_update = 3
    uses_sample_age = False
    estimates_angular_rate = False
    uses_prediction = False

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        del tracker, interval_s, prediction_delay_s
        offsets = np.asarray([sample[0] for sample in samples], dtype=float)
        powers_dbm = np.asarray([sample[1] for sample in samples], dtype=float)
        # Subtracting the maximum keeps the exponent numerically benign and
        # makes the weighted centroid insensitive to an unknown q offset.
        weights = 10.0 ** ((powers_dbm - float(np.max(powers_dbm))) / 10.0)
        correction = np.average(offsets, axis=0, weights=weights)
        return PolicyUpdate(np.clip(correction, -PI_LIMIT_DEG, PI_LIMIT_DEG), None)

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index, true_los_direction
        return mechanical_direction


class ExhaustiveScanningBaseline:
    """Periodic 3x3 measured-power search followed by max-power recentering."""

    name = 'exhaustive_scanning'
    uses_tracking_scheduler = True
    is_oracle_upper_bound = False
    tracking_measurements_per_update = 9
    uses_sample_age = False
    estimates_angular_rate = False
    uses_prediction = False

    def __init__(self, scan_interval_s: float = 1.0) -> None:
        if scan_interval_s <= 0.0:
            raise ValueError('scan_interval_s must be positive')
        self.scan_interval_s = float(scan_interval_s)

    def scan_points(self, pattern: MeasuredAntennaPattern, scan_loss_db: float) -> list[np.ndarray]:
        """Return a 3x3 local search grid at the measured-pattern scan radius."""
        reference = pattern.scan_points(scan_loss_db)
        radius_deg = float(np.linalg.norm(reference[0]))
        return [np.array([h_deg, e_deg], dtype=float)
                for h_deg in (-radius_deg, 0.0, radius_deg)
                for e_deg in (-radius_deg, 0.0, radius_deg)]

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        del tracker, interval_s, prediction_delay_s
        best = max(samples, key=lambda sample: sample[1])
        return PolicyUpdate(np.clip(np.asarray(best[0], dtype=float), -PI_LIMIT_DEG, PI_LIMIT_DEG), None)

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index, true_los_direction
        return mechanical_direction


class NoTrackingStaticBeam:
    """Physical no-tracking lower baseline: fixed mechanical beams, no scans."""

    name = 'no_tracking_static_beam'
    uses_tracking_scheduler = False
    is_oracle_upper_bound = False
    tracking_measurements_per_update = 0
    uses_sample_age = False
    estimates_angular_rate = False
    uses_prediction = False

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        del tracker, samples, interval_s, prediction_delay_s
        return PolicyUpdate(np.zeros(2, dtype=float), None)

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index, true_los_direction
        return mechanical_direction


class OraclePerfectPointingNoOverhead:
    """Perfect-pointing, zero-overhead rate upper bound (never a fair baseline)."""

    name = 'oracle_perfect_pointing_no_overhead'
    uses_tracking_scheduler = False
    is_oracle_upper_bound = True
    tracking_measurements_per_update = 0
    uses_sample_age = False
    estimates_angular_rate = False
    uses_prediction = False

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        del tracker, samples, interval_s, prediction_delay_s
        return PolicyUpdate(np.zeros(2, dtype=float), None)

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index, mechanical_direction
        return np.asarray(true_los_direction, dtype=float).copy()


class ExternalLiteratureBaseline(Protocol):
    """Reserved interface for a cited external method; no algorithm is fabricated here."""

    name: str

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        ...


class ExternalLiteratureBaselineAdapter:
    """Attach a selected, cited three-sample method without granting it truth access.

    The caller supplies an adapter that maps the same ``Tracker`` and three
    observed-power samples to a :class:`PolicyUpdate`.  Until a paper is
    selected, invoking it raises instead of silently inventing a baseline.
    """

    name = 'external_literature_adapter'
    uses_tracking_scheduler = True
    is_oracle_upper_bound = False
    tracking_measurements_per_update = 3
    uses_sample_age = False
    estimates_angular_rate = False
    uses_prediction = False

    def __init__(self, implementation: Callable[['Tracker', list[tuple[np.ndarray, float, float]], float, float], PolicyUpdate] | None = None,
                 name: str | None = None) -> None:
        self.implementation = implementation
        if name is not None:
            self.name = name

    @property
    def is_available(self) -> bool:
        return self.implementation is not None

    def on_three_samples(self, tracker: 'Tracker', samples: list[tuple[np.ndarray, float, float]],
                         interval_s: float, prediction_delay_s: float) -> PolicyUpdate:
        if self.implementation is None:
            raise NotImplementedError('Insert the selected cited literature method in ExternalLiteratureBaselineAdapter.')
        return self.implementation(tracker, samples, interval_s, prediction_delay_s)

    def data_beam_direction(self, terminal_index: int, mechanical_direction: np.ndarray,
                            true_los_direction: np.ndarray) -> np.ndarray:
        del terminal_index, true_los_direction
        return mechanical_direction

# =============================================================================
# Three-point measurement state and bidirectional A/B TDD simulation
# =============================================================================

@dataclass
class Tracker:
    """单个终端的跟踪器: 只保存核心字段。

    base_az/base_el 是当前扫描中心基准方向;三点扫描、IEKF 更新与 recenter
    都围绕该基准进行。pending_samples 保存"已采集、等待凑齐三点"的异步样本,
    每个样本的 age 随仿真步进持续增长,批量更新时按真实 age 使用。
    """
    name: str
    pattern: MeasuredAntennaPattern
    base_az_deg: float
    base_el_deg: float
    filter: IEKF
    controller: PIController = field(default_factory=PIController)
    pending_samples: list[tuple[np.ndarray, float, float]] = field(default_factory=list)
    last_correction_s: float = 0.0

    @property
    def estimate_error(self) -> np.ndarray:
        return self.filter.error

    @property
    def estimate_velocity(self) -> np.ndarray:
        return self.filter.velocity

    @property
    def estimate_q(self) -> float:
        return self.filter.q

    @property
    def estimate_q_dot(self) -> float:
        return self.filter.q_dot

    def predict(self, dt_s: float) -> None:
        """推进滤波器预测,并同步"老化"待批量更新的样本 age。"""
        self.filter.predict(dt_s)
        self.pending_samples = [(scan, measurement, age_s + dt_s) for scan, measurement, age_s in self.pending_samples]

    def set_base(self, az_deg: float, el_deg: float) -> None:
        """更新跟踪基准方向,并把 IEKF 状态/协方差变换到新基准的局部切平面。"""
        old_base = (self.base_az_deg, self.base_el_deg)
        new_base = (wrap_deg(az_deg), float(el_deg))
        self.filter.recenter_transform(old_base[0], old_base[1], new_base[0], new_base[1])
        self.base_az_deg, self.base_el_deg = new_base

@dataclass
class Terminal:
    """终端: 名称 + 运动 + 云台 + 跟踪器。"""
    name: str
    motion: Any
    gimbal: Gimbal
    tracker: Tracker

@dataclass
class SimulationResult:
    """仿真结果: 只保留 summary 与 timeseries。"""
    summary: dict[str, Any]
    timeseries: list[dict[str, Any]]

class BidirectionalMechanicalSimulation:
    """双终端交替扫描的确定性机械跟踪仿真。

    主循环 (每步 dt_s):
      1. 两端运动推进 + 两端 IEKF 预测 (并老化未批量更新的样本 age);
      2. 仅推进当前活动终端云台;
      3. 当前相位 (扫描停留 / recenter) 到时后触发 _complete_measurement:
         - 扫描停留结束 → 采集一个功率样本 (age=0);
         - 凑齐 3 个样本 → IEKF 批量更新 → PI+前馈修正 → recenter;
         - recenter 结束 → 切换活动终端,开始另一端的三点扫描;
      4. 记录当前链路状态 (内部缓存 + 可选 timeseries)。

    每个终端完整更新周期 = 3 个扫描命令 + 1 个 recenter 命令 = 8Tc (两端交替),
    该语义与论文原版一致。
    """

    def __init__(self, config: SimulationConfig, pattern: MeasuredAntennaPattern,
                 record_timeseries: bool = True,
                 policy: TrackingPolicy | None = None,
                 trial_randomness: TrialRandomness | None = None) -> None:
        config.validate()
        self.config = config
        self.pattern = pattern
        explicit_policy = policy is not None
        self.policy: TrackingPolicy = policy or self._default_policy(config)
        self._resolve_policy_controls(explicit_policy)
        self.effective_config = self._effective_config_for_policy()
        self.link = LinkBudget(tx_power_dbm=config.tx_power_dbm, bandwidth_hz=config.bandwidth_hz, noise_figure_db=config.noise_figure_db, temperature_k=config.temperature_k)
        self.trial_randomness = trial_randomness or TrialRandomness.from_seed(config.observation.seed)
        seed_sequence = np.random.SeedSequence(self.trial_randomness.seed)
        observation_sequence, servo_sequence = seed_sequence.spawn(2)
        self.observation_model = PowerObservationModel(
            config.observation, np.random.default_rng(observation_sequence)
        )
        self.servo_rng = np.random.default_rng(servo_sequence)
        self.energy_factors, self.shadow = self._make_energy_and_shadow_streams()
        self.energy_censored = 0
        self.uses_tracking_scheduler = self.policy.uses_tracking_scheduler
        self.record_timeseries = record_timeseries
        self.time_s = 0.0
        self.step_index = 0
        self.active_index = 0
        self.phase = 'initializing'
        self.phase_elapsed_s = 0.0
        self.phase_duration_s = config.dt_s
        self.scan_index = 0
        self.current_scan_plan: ScanDesign | None = None
        self.cycle_diagnostics: list[dict[str, Any]] = []
        self.estimator_diagnostics: list[dict[str, Any]] = []
        self._future_checks: list[dict[str, Any]] = []
        self._last_decision_display: dict[str, Any] = {}
        # Visualization-only record of the real scan cycle.  These values are
        # copied out by get_snapshot(); no algorithmic method reads them.
        self.scan_completed_mask = [False] * len(self.scan_points)
        self.scan_measured_powers_dbm: list[float | None] = [None] * len(self.scan_points)
        self.scan_true_powers_dbm: list[float | None] = [None] * len(self.scan_points)
        self.scan_measurement_noise_db: list[float | None] = [None] * len(self.scan_points)
        self.scan_measurement_ages_s: list[float | None] = [None] * len(self.scan_points)
        self.last_measurement_scan_index: int | None = None
        self.last_measurement_time_s: float | None = None
        self.measurement_event_id = 0
        self.last_iekf_update_time_s: float | None = None
        self.last_recenter_duration_s = 0.0
        self.last_correction_local: np.ndarray | None = None
        self.old_center_local: np.ndarray | None = None
        self.new_center_local: np.ndarray | None = None
        self.recenter_terminal_name: str | None = None
        # Causal timing stamps: state reference -> command issue -> expected
        # effect -> actual completion (guide section 6).
        self.estimate_reference_time_s: float | None = None
        self.command_issue_time_s: float | None = None
        self.expected_effective_time_s: float | None = None
        self.last_command_complete_time_s: float | None = None
        self.timeseries: list[dict[str, Any]] = []
        self.full: dict[str, list[Any]] = {
            'time_s': [], 'distance_m': [], 'los_rate_deg_s': [],
            'mean_beam_error': [], 'pointing_loss': [], 'true_pr_dbm': [],
            'mean_center_error': [], 'a_center_error': [], 'b_center_error': [],
            'snr_db': [], 'outage': [], 'spectral_efficiency_bps_hz': [],
            'lock': [], 'pattern_clamped': [], 'tracking_active': [],
            'data_active': [], 'data_snr_db': [], 'data_outage': [],
            'data_spectral_efficiency_bps_hz': [], 'data_beam_error': [],
            'mechanical_phase': [], 'payload_active': [],
            'pattern_support_valid': [], 'link_available': [],
        }
        self.diagnostics: dict[str, list[float]] = {
            'accepted_update': [], 'innovation_score': [], 'measurement_scale': [],
            'minimum_covariance_eigenvalue': [], 'sample_age_s': [],
            'iekf_update_time_ms': [], 'controller_time_ms': [],
            'power_sample_time_ms': [], 'measurement_batch_compute_time_ms': [],
            'tracking_cycle_compute_time_ms': [],
            'simulation_step_time_ms': [], 'tracking_frame_duration_s': [],
            'tracking_phase_duration_s': [], 'terminal_refresh_interval_s': [],
            'data_phase_duration_s': [],
            'estimate_reference_time_s': [], 'command_issue_time_s': [],
            'expected_effective_time_s': [], 'command_complete_time_s': [],
            'command_latency_s': [], 'expected_command_latency_s': [],
            'estimate_to_issue_s': [],
        }
        self._measurement_compute_accumulator_ms = {'A': 0.0, 'B': 0.0}
        self.tracking_cycle_index = 0
        self.last_observation = PowerObservation(0.0, 0.0, 0.0, 0.0)
        self._servo_jitter_he_deg = np.zeros((2, 2), dtype=float)
        self.terminals = self._make_terminals()
        self._refresh_servo_jitter()
        self.current_state = self._link_state()
        initial_true_power = float(self.current_state['true_pr_dbm'])
        self.last_observation = PowerObservation(initial_true_power, initial_true_power, 0.0, 0.0)
        self.last_measured_power_dbm = initial_true_power
        self.power_measurement_event = False
        self.frame_start_s = self.time_s
        self._phase_for_record = 'stationary' if not self.uses_tracking_scheduler else 'three_point_scan'
        # 全程与稳态两个显式评估窗口; 初始化快照权重为 0, 跨越窗口边界的
        # 区间按交集裁剪。稳态窗口与 `_summarize` 主指标完全一致。
        self.link_metrics = ContinuousLinkMetrics(
            config.snr_threshold_db, config.service_rate_bps_hz,
            window_start_s=0.0, window_end_s=config.duration_s)
        self.link_metrics_steady = ContinuousLinkMetrics(
            config.snr_threshold_db, config.service_rate_bps_hz,
            window_start_s=config.metric_start_s, window_end_s=config.duration_s)
        if self.uses_tracking_scheduler:
            self._begin_next_measurement()
        else:
            self.phase = 'stationary'
            self.phase_duration_s = math.inf
        self._record()

    @property
    def scan_points(self) -> list[np.ndarray]:
        return [] if self.current_scan_plan is None else list(self.current_scan_plan.commanded_offsets_deg)

    @staticmethod
    def _default_policy(config: SimulationConfig) -> TrackingPolicy:
        """Policy injection by constructor config; no shared global factory."""
        if config.service_horizon_enabled:
            if config.active_scan.enabled:
                return ActiveServiceHorizonPolicy(quantization_aware=config.quantization_aware_enabled)
            return FixedServiceHorizonPolicy(quantization_aware=config.quantization_aware_enabled)
        if config.active_scan.enabled:
            return ActiveLatencyPolicy()
        if config.estimator == 'energy_map':
            return MotionPredictivePolicy() if config.prediction_enabled else MotionNoPredictionPolicy()
        return ProposedAgeAwarePredictive() if config.prediction_enabled else AgeAwareNoPrediction()

    def _mean_hpbw_deg(self) -> float:
        return 0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg)

    def _resolve_policy_controls(self, explicit_policy: bool) -> None:
        """把反馈类型、限幅与预测开关解析为唯一一份有效控制配置。

        配置派生的策略 (``explicit_policy=False``) 必须执行配置中的
        ``feedback``、``command_limit_deg`` 与 ``prediction_enabled``;
        显式传入的策略以策略自身声明为准, 其实际生效值统一从策略属性导出,
        summary/snapshot/CSV 不再出现配置与执行不一致的元数据。
        """
        policy = self.policy
        if isinstance(policy, ContextPointingPolicy) and (self.config.estimator != 'energy_map' or self.config.feedback != 'unit'):
            raise ValueError('context policies require EnergyMAP/unit feedback')
        if explicit_policy:
            if isinstance(policy, _AgeAwarePolicy):
                if policy.command_limit_deg is None:
                    policy.command_limit_deg = (
                        PI_LIMIT_DEG if policy.feedback_kind == 'pi' else self._mean_hpbw_deg()
                    )
            return
        if isinstance(policy, _AgeAwarePolicy):
            policy.feedback_kind = self.config.feedback
            if self.config.command_limit_deg is not None:
                policy.command_limit_deg = float(self.config.command_limit_deg)
            elif policy.command_limit_deg is None:
                policy.command_limit_deg = (
                    PI_LIMIT_DEG if policy.feedback_kind == 'pi' else self._mean_hpbw_deg()
                )

    def _effective_config_for_policy(self) -> SimulationConfig:
        """实际生效配置: 由唯一来源 (策略实例) 解析, 用于导出元数据。

        配置中的 ``feedback``/``command_limit_deg``/``prediction_enabled``
        只作声明; 实际执行值以解析后的策略属性为准, 因此导出字段与真实
        控制律一致 (验收问题 4)。
        """
        policy = self.policy
        feedback = getattr(policy, 'feedback_kind', None)
        limit = getattr(policy, 'command_limit_deg', None)
        prediction = getattr(policy, 'uses_prediction', False)
        return replace(
            self.config,
            feedback=feedback if feedback in ('pi', 'unit') else self.config.feedback,
            command_limit_deg=float(limit) if limit is not None else self.config.command_limit_deg,
            prediction_enabled=bool(prediction),
            active_scan=replace(self.config.active_scan, enabled=bool(getattr(policy, 'active_scan_enabled', False))),
            service_horizon_enabled=bool(getattr(policy, 'service_horizon', False)),
            quantization_aware_enabled=bool(getattr(policy, 'quantization_aware', False)),
        )

    def _make_energy_and_shadow_streams(self) -> tuple[np.ndarray, np.ndarray]:
        """Pre-generated Gamma energy factors and slow channel shadowing.

        Both streams are owned by explicit RNGs derived from the trial seed and
        are identical for the prediction and no-prediction groups (paired
        common random numbers).  They only feed the ``energy_map`` observation
        mode and are frozen before any simulation step.
        """
        count = round(self.config.duration_s / self.config.dt_s) + 2
        energy_rng = np.random.default_rng(np.random.SeedSequence([self.trial_randomness.seed, 887]))
        energy_factors = energy_rng.gamma(self.config.energy_samples, 1.0 / self.config.energy_samples, count)
        channel_rng = np.random.default_rng(np.random.SeedSequence([self.trial_randomness.seed, 991]))
        alpha = math.exp(-self.config.dt_s / self.config.shadow_tau_s)
        innovations = channel_rng.normal(0.0, self.config.shadow_sigma_db * math.sqrt(1.0 - alpha ** 2), count)
        shadow = np.zeros(count, dtype=float)
        shadow[0] = channel_rng.normal(0.0, self.config.shadow_sigma_db)
        for index in range(1, count):
            shadow[index] = alpha * shadow[index - 1] + innovations[index]
        return energy_factors, shadow

    @staticmethod
    def _los_direction(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        """source 指向 destination 的视距单位方向。"""
        delta = np.asarray(destination, dtype=float) - np.asarray(source, dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm <= 0.0:
            raise ValueError('LOS is undefined at zero separation')
        return delta / norm

    def _target_direction(self, terminal_index: int) -> np.ndarray:
        """终端 terminal_index 看向对端的视距方向。"""
        return self._los_direction(self.terminals[terminal_index].motion.position, self.terminals[1 - terminal_index].motion.position)

    def _make_terminals(self) -> list[Terminal]:
        """初始化 A/B 两端: 正交线性运动、云台、跟踪器与初始指向。

        * A 端位于 [0,0,20],沿 +x 运动;B 端位于 [0,d,20],沿 +z 运动;
        * 初始指向 = 视距方向 + 初始误差偏置 (A 沿局部 H 轴 0° 相位,
          B 沿 120° 相位,幅值 = initial_offset_deg);
        * q 初值 = 已知初始距离处的对准链路预算 (局部跟踪初始化假设);
        * 角加速度设计量 = sqrt(2)·MAX_ACCEL / d / ACCEL_PRIOR_SIGMA_DIVISOR (deg/s^2)。
        """
        if self.config.maneuver == 'sinusoidal':
            motion_a: Any = SinusoidalVelocityMotion(
                position=np.array([0.0, 0.0, 20.0], dtype=float),
                direction=np.array([1.0, 0.0, 0.0], dtype=float),
                base_speed_mps=self.config.speed_mps,
                speed_delta_mps=self.config.sinusoidal_speed_delta_mps,
                frequency_hz=self.config.sinusoidal_frequency_hz,
            )
            motion_b: Any = SinusoidalVelocityMotion(
                position=np.array([0.0, self.config.initial_distance_m, 20.0], dtype=float),
                direction=np.array([0.0, 0.0, 1.0], dtype=float),
                base_speed_mps=self.config.speed_mps,
                speed_delta_mps=self.config.sinusoidal_speed_delta_mps,
                frequency_hz=self.config.sinusoidal_frequency_hz,
            )
        else:
            motion_a = PaperOrthogonalMotion(position=np.array([0.0, 0.0, 20.0], dtype=float), velocity=np.array([self.config.speed_mps, 0.0, 0.0], dtype=float))
            motion_b = PaperOrthogonalMotion(position=np.array([0.0, self.config.initial_distance_m, 20.0], dtype=float), velocity=np.array([0.0, 0.0, self.config.speed_mps], dtype=float))
        los_a = self._los_direction(motion_a.position, motion_b.position)
        los_b = -los_a
        az_a, el_a = az_el_from_direction(los_a)
        az_b, el_b = az_el_from_direction(los_b)
        if self.config.initial_offset_deg:
            phase_a = (
                self.trial_randomness.initial_phase_a_rad
                if self.config.randomize_initial_error_phase
                else math.radians(INITIAL_ERROR_PHASE_A_DEG)
            )
            phase_b = (
                self.trial_randomness.initial_phase_b_rad
                if self.config.randomize_initial_error_phase
                else math.radians(INITIAL_ERROR_PHASE_B_DEG)
            )
            az_a, el_a = offset_direction(az_a, el_a, self.config.initial_offset_deg * math.cos(phase_a), self.config.initial_offset_deg * math.sin(phase_a))
            az_b, el_b = offset_direction(az_b, el_b, self.config.initial_offset_deg * math.cos(phase_b), self.config.initial_offset_deg * math.sin(phase_b))
        distance = float(np.linalg.norm(motion_b.position - motion_a.position))
        initial_q = self.link.received_power_dbm(distance, self.pattern.common_peak_dbi, self.pattern.common_peak_dbi)
        angular_accel = math.degrees(math.sqrt(2.0) * MAX_ACCEL_MPS2 / distance) / ACCEL_PRIOR_SIGMA_DIVISOR
        tracker_a = Tracker(name='A', pattern=self.pattern, base_az_deg=az_a, base_el_deg=el_a, filter=self._make_filter(initial_q, angular_accel))
        tracker_b = Tracker(name='B', pattern=self.pattern, base_az_deg=az_b, base_el_deg=el_b, filter=self._make_filter(initial_q, angular_accel))
        return [Terminal('A', motion_a, Gimbal(self.config.quantization_deg, self.config.command_time_s, az_a, el_a), tracker_a), Terminal('B', motion_b, Gimbal(self.config.quantization_deg, self.config.command_time_s, az_b, el_b), tracker_b)]

    def _make_filter(self, initial_q_dbm: float, angular_accel_deg_s2: float) -> IEKF:
        """Estimator constructor injection: IEKF or recursive MAP (EnergyMAP)."""
        if self.config.estimator == 'energy_map':
            filter_ = EnergyMAP(self.pattern, initial_q_dbm, angular_accel_deg_s2)
            filter_.configure(self.link.noise_power_dbm, self.config.angular_psd,
                              self.config.q_psd, five_state=False)
            return filter_
        return IEKF(self.pattern, initial_q_dbm, angular_accel_deg_s2,
                    self.config.measurement_design_sigma_db,
                    self.config.q_accel_sigma_db_per_s2,
                    self.config.q_process_scale, self.config.r_measurement_scale)

    def _command(self, terminal_index: int, base_az_el: tuple[float, float], offset: np.ndarray) -> None:
        """下达一次机械命令: 目标 = 基准方向 + 局部 H/E 偏置,量化后转动。"""
        terminal = self.terminals[terminal_index]
        target_az_el = offset_direction(base_az_el[0], base_az_el[1], float(offset[0]), float(offset[1]))
        target_az_el = terminal.gimbal.quantized_angles(*target_az_el)
        duration = terminal.gimbal.command(*target_az_el)
        self.phase_elapsed_s = 0.0
        self.phase_duration_s = max(duration, self.config.dt_s)

    def _recenter_and_command(self, terminal_index: int, correction_deg: np.ndarray,
                              quantized_target_az_el: tuple[float, float] | None = None) -> None:
        """把跟踪基准 recenter 到"当前基准 + 修正量"的量化位置,并命令云台转过去。"""
        terminal = self.terminals[terminal_index]
        old_base_az_el = (terminal.tracker.base_az_deg, terminal.tracker.base_el_deg)
        requested_az_el = offset_direction(terminal.tracker.base_az_deg, terminal.tracker.base_el_deg, float(correction_deg[0]), float(correction_deg[1]))
        achieved_az_el = (terminal.gimbal.quantized_angles(*requested_az_el)
                          if quantized_target_az_el is None else quantized_target_az_el)
        # Express old and new centers in the new tracking-center frame.  The
        # display arrow therefore shows the actual quantised recenter geometry,
        # while last_correction_local retains the controller output itself.
        old_center_direction = direction_from_az_el(*old_base_az_el)
        self.old_center_local = np.asarray(
            local_error_deg(*achieved_az_el, old_center_direction), dtype=float
        )
        self.new_center_local = np.zeros(2, dtype=float)
        self.last_correction_local = np.asarray(correction_deg, dtype=float).copy()
        self.recenter_terminal_name = terminal.name
        terminal.tracker.set_base(*achieved_az_el)
        self.phase = 'recenter'
        self._command(terminal_index, achieved_az_el, np.zeros(2, dtype=float))
        self.last_recenter_duration_s = self.phase_duration_s

    def _current_scan_offset(self, terminal_index: int) -> np.ndarray:
        """当前云台实际指向相对跟踪基准的局部 H/E 偏置 (deg)。"""
        terminal = self.terminals[terminal_index]
        target = terminal.gimbal.direction
        h_deg, e_deg = local_error_deg(terminal.tracker.base_az_deg, terminal.tracker.base_el_deg, target)
        return np.array([h_deg, e_deg], dtype=float)

    def _refresh_servo_jitter(self) -> None:
        """Refresh the unobserved actual-beam perturbation once per physics step."""
        sigma_deg = self.config.observation.servo_jitter_sigma_deg
        if sigma_deg == 0.0 or not self.config.observation.enabled:
            self._servo_jitter_he_deg.fill(0.0)
        else:
            # This stream is independent of tracking-measurement calls, so
            # every policy sees the same time-indexed jitter realization.
            self._servo_jitter_he_deg = self.servo_rng.normal(0.0, sigma_deg, size=(2, 2))

    def _mechanical_beam_direction(self, terminal: Terminal, terminal_index: int) -> np.ndarray:
        """Actual gimbal direction including optional unobserved servo jitter."""
        jitter_h_deg, jitter_e_deg = self._servo_jitter_he_deg[terminal_index]
        az_deg, el_deg = offset_direction(
            terminal.gimbal.az_deg, terminal.gimbal.el_deg,
            float(jitter_h_deg), float(jitter_e_deg),
        )
        return direction_from_az_el(az_deg, el_deg)

    def _beam_direction(self, terminal: Terminal, terminal_index: int) -> np.ndarray:
        """Actual data-beam direction selected by the policy.

        The policy can only change the data-beam mapping.  In the normal
        policies this is the mechanically achieved beam; the oracle is the
        explicitly labelled upper bound that returns the true LoS direction.
        """
        mechanical = self._mechanical_beam_direction(terminal, terminal_index)
        if isinstance(self.policy, ContextPointingPolicy):
            return mechanical
        return self.policy.data_beam_direction(
            terminal_index, mechanical, self._target_direction(terminal_index),
        )

    @staticmethod
    def _beam_center_direction(terminal: Terminal) -> np.ndarray:
        """扫描中心方向 = 跟踪基准方向 (同样保留零偏置往返,与原版数值一致)。"""
        az_deg, el_deg = offset_direction(terminal.tracker.base_az_deg, terminal.tracker.base_el_deg, 0.0, 0.0)
        return direction_from_az_el(az_deg, el_deg)

    def _link_state(self) -> dict[str, Any]:
        """计算当前步的完整链路状态 (真实量,不含随机噪声)。"""
        a, b = self.terminals
        target_a = self._target_direction(0)
        target_b = self._target_direction(1)
        a_h, a_e = local_error_deg(*az_el_from_direction(self._beam_direction(a, 0)), target_a)
        b_h, b_e = local_error_deg(*az_el_from_direction(self._beam_direction(b, 1)), target_b)
        a_gain, a_clamped = self.pattern.gain_dbi(a_h, a_e)
        b_gain, b_clamped = self.pattern.gain_dbi(b_h, b_e)
        distance = float(np.linalg.norm(b.motion.position - a.motion.position))
        true_pr_dbm = self.link.received_power_dbm(distance, a_gain, b_gain)
        pointing_loss = 2.0 * self.pattern.common_peak_dbi - a_gain - b_gain
        if self.config.estimator == 'energy_map':
            # Continuous-communication observation chain: slow shadowing and,
            # outside the measured-pattern support, conservative censoring
            # instead of extrapolated antenna gain.
            excess = (self.config.pattern_scale - 1.0) * pointing_loss
            pointing_loss = pointing_loss * self.config.pattern_scale
            true_pr_dbm = true_pr_dbm + self.shadow[self.step_index] - excess
            if a_clamped or b_clamped:
                true_pr_dbm = self.link.noise_power_dbm - 100.0
        snr_db = self.link.snr_db(true_pr_dbm)
        center_a = local_error_deg(*az_el_from_direction(self._beam_center_direction(a)), target_a)
        center_b = local_error_deg(*az_el_from_direction(self._beam_center_direction(b)), target_b)
        previous_a = a.motion.previous_position
        previous_b = b.motion.previous_position
        previous_los = self._los_direction(previous_a, previous_b)
        los_rate = angular_distance_deg(previous_los, target_a) / self.config.dt_s
        return {'distance_m': distance, 'a_beam_h_deg': a_h, 'a_beam_e_deg': a_e, 'b_beam_h_deg': b_h, 'b_beam_e_deg': b_e, 'a_center_h_deg': center_a[0], 'a_center_e_deg': center_a[1], 'b_center_h_deg': center_b[0], 'b_center_e_deg': center_b[1], 'true_pr_dbm': true_pr_dbm, 'pr_dbm': true_pr_dbm, 'snr_db': snr_db, 'pointing_loss_db': pointing_loss, 'los_rate_deg_s': los_rate, 'pattern_clamped_any': int(a_clamped or b_clamped)}

    def _measurement(self) -> PowerObservation:
        """Sample the receiver observation while retaining the physical truth.

        The ``energy_map`` estimator consumes finite-sample energy: the total
        energy of ``energy_samples`` complex-Gaussian samples follows a Gamma
        distribution ``Gamma(M, (1+SNR)/M)`` (guide section 7).  Low SNR does
        not remove the measurement; noise-dominated observations are still
        processed and rejected updates keep the causal prior.
        """
        if self.config.estimator != 'energy_map':
            return self.observation_model.observe_dbm(
                float(self.current_state['true_pr_dbm']), self.tracking_cycle_index,
            )
        power = float(self.current_state['true_pr_dbm'])
        snr = 10.0 ** ((power - self.link.noise_power_dbm) / 10.0)
        total = (1.0 + snr) * self.energy_factors[self.step_index]
        observed_signal = max(total - 1.0, 1e-10)
        if total <= 1.0:
            self.energy_censored += 1
        observed = self.link.noise_power_dbm + 10.0 * math.log10(observed_signal)
        fil = self.terminals[self.active_index].tracker.filter
        if hasattr(fil, 'energy_batch'):
            fil.energy_batch = (fil.energy_batch + [(total, self.config.energy_samples)])[-9:]
        return PowerObservation(power, observed, observed - power, 0.0)

    def effective_prediction_delay_s(self) -> float:
        """已废弃: 旧数据窗模型诊断。

        旧口径为"最老样本龄期 + 回中时间 + A 端等待 B 扫描的时间", 属于
        分阶段数据窗模型的预测时域, 不再作为新模型的物理时延证据。
        新模型的主执行时延为配置的 ``actuation_delay_s``; 实际命令时延由
        状态参考/命令发出/预期生效/完成时刻计算 (见 summary 的
        ``command_latency_s_mean`` 等字段), 样本龄期单独记录在
        ``sample_age_s``, 不重复相加。
        """
        raise DeprecationWarning(
            'effective_prediction_delay_s is a legacy data-window diagnostic; '
            'use actuation_delay_s and the causal command timing stamps instead.'
        )

    def _decision_diagnostics_enabled(self):
        return self.config.decision_diagnostics_enabled or isinstance(self.policy, ContextPointingPolicy)

    def _decision_context(self):
        terminal = self.terminals[self.active_index]
        tracker, mechanics = terminal.tracker, terminal.gimbal
        refresh = self.config.same_terminal_refresh_interval_s
        if refresh is None:
            # Fixed A/B schedule: each terminal has three probes plus recenter,
            # followed by the declared stationary dwell. Future decisions are
            # unknown, so this is a schedule forecast, not an oracle timestamp.
            refresh = 8 * self.config.command_time_s + self.config.stationary_dwell_s
        refresh = max(refresh, self.config.effective_actuation_delay_s)
        return DecisionContext(
            tracker.filter.x, tracker.filter.p, (tracker.base_az_deg, tracker.base_el_deg),
            MechanicalState(mechanics.az_deg, mechanics.el_deg, mechanics.quantization_deg, mechanics.command_time_s),
            self.link.noise_power_dbm, self.config.energy_samples, self.config.dt_s,
            self.config.effective_actuation_delay_s, refresh,
            self.config.angular_psd, self.config.q_psd,
            float(getattr(self.policy, 'command_limit_deg', None) or self._mean_hpbw_deg()),
            self.effective_config.active_scan)

    def _prepare_scan_plan(self):
        context = self._decision_context()
        if isinstance(self.policy, ContextPointingPolicy):
            self.current_scan_plan = self.policy.prepare_scan_plan(context, self.pattern, self.config.scan_loss_db)
        else:
            offsets = (self.policy.scan_points(self.pattern, self.config.scan_loss_db)
                       if isinstance(self.policy, ExhaustiveScanningBaseline)
                       else self.pattern.scan_points(self.config.scan_loss_db))
            self.current_scan_plan = prepare_scan_design(
                context, self.pattern, self.config.scan_loss_db, offsets,
                score=self._decision_diagnostics_enabled() and self.config.estimator == 'energy_map')

    def _record_decision_diagnostics(self, decision):
        context = self._decision_context()
        if decision.decision_diagnostics is None:
            # Diagnostics for B0/B1 do not alter their estimator/controller path.
            tau = service_horizon(context.actuation_delay_s, context.same_terminal_refresh_interval_s)
            preview = preview_recenter(context, decision.correction_deg)
            realized = local_error_deg(*context.base_az_el, direction_from_az_el(preview.az_deg, preview.el_deg))
            covariance, normalized = future_pointing_covariance(context.p, tau,
                (self.pattern.qa.h_hpbw_deg, self.pattern.qa.e_hpbw_deg))
            values = dict(service_horizon_s=tau, actuation_delay_s=context.actuation_delay_s,
                          same_terminal_refresh_interval_s=context.same_terminal_refresh_interval_s,
                          latency_target=(context.x[2:4] + context.actuation_delay_s * context.x[4:6]).tolist(),
                          service_target=(pointing_projection(tau) @ context.x).tolist(),
                          selected_quantized_command=list(realized),
                          selected_quantized_az_el=[preview.az_deg, preview.el_deg],
                          predicted_command_duration_s=preview.duration_s,
                          predicted_integrated_se=predicted_integrated_se(context, self.pattern, np.asarray(realized)),
                          future_covariance=covariance.tolist(), normalized_future_covariance=normalized.tolist())
        else:
            values = copy.deepcopy(decision.decision_diagnostics)
        plan = self.current_scan_plan
        values.update(time_s=self.time_s, terminal=self.terminals[self.active_index].name,
                      selected_scan_loss_db=plan.scan_loss_db,
                      expected_information=None if plan.expected_information is None else plan.expected_information.tolist(),
                      expected_future_covariance=None if plan.predicted_future_covariance is None else plan.predicted_future_covariance.tolist(),
                      expected_posterior=None if plan.expected_posterior is None else plan.expected_posterior.tolist(),
                      realized_offsets_deg=[x.tolist() for x in plan.realized_offsets_deg],
                      actual_realized_offsets_deg=[sample[0].tolist() for sample in self.terminals[self.active_index].tracker.pending_samples],
                      predicted_sample_ages_s=list(plan.predicted_sample_ages_s),
                      actual_sample_ages_s=list(self.scan_measurement_ages_s),
                      predicted_scan_se=plan.predicted_scan_se, scan_objective=plan.predicted_objective)
        self.cycle_diagnostics.append(values)
        self._last_decision_display = copy.deepcopy(values)
        self._record_estimator_truth_diagnostics(context, values)

    def _record_estimator_truth_diagnostics(self, context, decision_values):
        """Evaluation sink only. No result is returned to any policy or planner."""
        index = self.active_index
        a, b = self.terminals[index], self.terminals[1 - index]
        relative = b.motion.position - a.motion.position

        def velocity(motion):
            if isinstance(motion, SinusoidalVelocityMotion):
                return motion.direction * (motion.base_speed_mps + motion.speed_delta_mps *
                    math.sin(2 * math.pi * motion.frequency_hz * motion.time_s))
            return motion.velocity

        relative_rate = velocity(b.motion) - velocity(a.motion)
        boresight, h_axis, e_axis = tangent_basis(*context.base_az_el)
        denominator = float(relative @ boresight)
        denominator_rate = float(relative_rate @ boresight)
        error = np.asarray(local_error_deg(*context.base_az_el, relative / np.linalg.norm(relative)))
        rate = np.array([math.degrees((denominator * float(relative_rate @ axis) -
                         float(relative @ axis) * denominator_rate) /
                         max(denominator ** 2 + float(relative @ axis) ** 2, 1e-24))
                         for axis in (h_axis, e_axis)])
        delta = context.x[2:6] - np.r_[error, rate]
        tau = decision_values['service_horizon_s']
        future_delta = delta[:2] + tau * delta[2:]
        covariance = future_pointing_covariance(context.p, tau)
        mahalanobis = float(future_delta @ np.linalg.solve(covariance, future_delta))
        row = dict(time_s=self.time_s, terminal=a.name, service_horizon_s=tau,
                   e_H_error_deg=float(delta[0]), e_E_error_deg=float(delta[1]),
                   omega_H_error_deg_s=float(delta[2]), omega_E_error_deg_s=float(delta[3]),
                   future_linear_H_error_deg=float(future_delta[0]),
                   future_linear_E_error_deg=float(future_delta[1]),
                   future_linear_mahalanobis_sq=mahalanobis,
                   future_linear_coverage_95=bool(mahalanobis <= 5.991464547107979),
                   future_evaluated=False)
        self.estimator_diagnostics.append(row)
        self._future_checks.append(dict(row=row, index=index, due_s=self.time_s + tau,
                                         base=context.base_az_el,
                                         prediction=pointing_projection(tau) @ context.x,
                                         covariance=covariance.copy()))

    def _evaluate_future_checks(self):
        remaining = []
        for check in self._future_checks:
            if check['due_s'] > self.time_s + 1e-12:
                remaining.append(check)
                continue
            index = check['index']
            a, b = self.terminals[index], self.terminals[1 - index]
            fraction = float(np.clip((check['due_s'] - (self.time_s - self.config.dt_s)) / self.config.dt_s, 0, 1))
            def position(motion):
                return motion.previous_position + fraction * (motion.position - motion.previous_position)
            relative = position(b.motion) - position(a.motion)
            actual = np.asarray(local_error_deg(*check['base'], relative / np.linalg.norm(relative)))
            delta = check['prediction'] - actual
            mahalanobis = float(delta @ np.linalg.solve(check['covariance'], delta))
            check['row'].update(future_evaluated=True, future_time_s=check['due_s'],
                                future_H_error_deg=float(delta[0]), future_E_error_deg=float(delta[1]),
                                future_mahalanobis_sq=mahalanobis,
                                future_coverage_95=bool(mahalanobis <= 5.991464547107979))
        self._future_checks = remaining

    def _begin_next_measurement(self) -> None:
        """命令活动终端云台指向下一个扫描点。"""
        if self.scan_index == 0 and not self.terminals[self.active_index].tracker.pending_samples:
            self._prepare_scan_plan()
            self.scan_completed_mask = [False] * len(self.scan_points)
            self.scan_measured_powers_dbm = [None] * len(self.scan_points)
            self.scan_true_powers_dbm = [None] * len(self.scan_points)
            self.scan_measurement_noise_db = [None] * len(self.scan_points)
            self.scan_measurement_ages_s = [None] * len(self.scan_points)
            self.last_correction_local = None
            self.old_center_local = None
            self.new_center_local = None
            self.recenter_terminal_name = None
        tracker = self.terminals[self.active_index].tracker
        offset = self.scan_points[self.scan_index]
        self.phase = 'three_point_scan'
        self._command(self.active_index, (tracker.base_az_deg, tracker.base_el_deg), offset)

    def _begin_data_phase_or_next_frame(self) -> None:
        """Enter the mechanical stationary dwell, or restart the tracking frame.

        The stationary dwell is a period without micro-scanning; communication
        stays active throughout (guide section 2).  It does not reserve an
        exclusive data window any more.
        """
        self.current_tracking_phase_duration_s = self.time_s - self.frame_start_s
        self.diagnostics['tracking_phase_duration_s'].append(self.current_tracking_phase_duration_s)
        if isinstance(self.policy, ExhaustiveScanningBaseline):
            # ``scan_interval_s`` means the requested period between complete
            # exhaustive scan frames.  A mechanical 3x3 A/B scan can itself
            # exceed a short requested period; in that case no fictitious
            # dwell is inserted and the measured frame duration documents the
            # physically achievable refresh interval.
            remaining_dwell_s = self.policy.scan_interval_s - self.current_tracking_phase_duration_s
            dwell_s = remaining_dwell_s if remaining_dwell_s > self.config.dt_s else 0.0
        else:
            dwell_s = self.config.stationary_dwell_s
        if dwell_s > 0.0:
            self.phase = 'stationary'
            self.phase_elapsed_s = 0.0
            self.phase_duration_s = float(dwell_s)
            return
        self._finish_tracking_frame()

    def _finish_tracking_frame(self) -> None:
        """Close one A-track/B-track/(optional stationary dwell) frame and restart A tracking."""
        frame_duration_s = self.time_s - self.frame_start_s
        self.diagnostics['tracking_frame_duration_s'].append(frame_duration_s)
        self.diagnostics['terminal_refresh_interval_s'].append(frame_duration_s)
        tracking_duration_s = getattr(self, 'current_tracking_phase_duration_s', frame_duration_s)
        self.diagnostics['data_phase_duration_s'].append(
            max(0.0, frame_duration_s - tracking_duration_s)
        )
        self.frame_start_s = self.time_s
        self.active_index = 0
        self.scan_index = 0
        self._begin_next_measurement()

    def _complete_measurement(self) -> None:
        """当前相位结束时触发: 采集样本 / 批量更新 / recenter / 切换终端。"""
        terminal = self.terminals[self.active_index]
        tracker = terminal.tracker
        if self.phase == 'stationary':
            self._finish_tracking_frame()
            return
        if self.phase == 'recenter':
            self.last_command_complete_time_s = self.time_s
            if self.last_command_complete_time_s is not None:
                self.diagnostics['command_complete_time_s'].append(float(self.last_command_complete_time_s))
            if self.command_issue_time_s is not None:
                self.diagnostics['command_latency_s'].append(
                    float(self.last_command_complete_time_s - self.command_issue_time_s)
                )
            if self.command_issue_time_s is not None and self.expected_effective_time_s is not None:
                self.diagnostics['expected_command_latency_s'].append(
                    float(self.expected_effective_time_s - self.command_issue_time_s)
                )
            if self.estimate_reference_time_s is not None and self.command_issue_time_s is not None:
                self.diagnostics['estimate_to_issue_s'].append(
                    float(self.command_issue_time_s - self.estimate_reference_time_s)
                )
            if self.active_index == 0:
                self.active_index = 1
                self.scan_index = 0
                self._begin_next_measurement()
            else:
                self._begin_data_phase_or_next_frame()
            return
        measurement_started = time.perf_counter()
        observation = self._measurement()
        measurement_time_ms = (time.perf_counter() - measurement_started) * 1e3
        self.diagnostics['power_sample_time_ms'].append(measurement_time_ms)
        self._measurement_compute_accumulator_ms[terminal.name] += measurement_time_ms
        measurement_dbm = observation.observed_pr_dbm
        self.last_observation = observation
        self.last_measured_power_dbm = measurement_dbm
        self.power_measurement_event = True
        self.last_measurement_scan_index = self.scan_index
        self.last_measurement_time_s = self.time_s
        self.measurement_event_id += 1
        self.scan_completed_mask[self.scan_index] = True
        self.scan_measured_powers_dbm[self.scan_index] = measurement_dbm
        self.scan_true_powers_dbm[self.scan_index] = observation.true_pr_dbm
        self.scan_measurement_noise_db[self.scan_index] = observation.measurement_noise_db
        scan_offset = self._current_scan_offset(self.active_index)
        tracker.pending_samples.append((scan_offset, measurement_dbm, 0.0))
        if len(tracker.pending_samples) < len(self.scan_points):
            self.scan_index += 1
            self._begin_next_measurement()
            return
        cycle_update_started = time.perf_counter()
        self.scan_measurement_ages_s = [sample[2] for sample in tracker.pending_samples]
        self.diagnostics['sample_age_s'].extend(self.scan_measurement_ages_s)
        interval_s = max(self.time_s - tracker.last_correction_s, self.config.dt_s)
        # The causal actuation delay goes to the unified control law.  Sample
        # ages are aligned inside the estimator and logged separately; the
        # physical command latency is computed from the recorded timing stamps
        # (guide section 6) and never added twice.
        self.estimate_reference_time_s = self.time_s
        if isinstance(self.policy, ContextPointingPolicy):
            started = time.perf_counter()
            update = tracker.filter.update(tracker.pending_samples)
            update_ms = (time.perf_counter() - started) * 1e3
            decision = self.policy.decide(self._decision_context(), self.pattern)
            decision = replace(decision, iekf_result=update, iekf_update_time_ms=update_ms)
        else:
            decision = self.policy.on_three_samples(
                tracker, tracker.pending_samples, interval_s,
                self.config.effective_actuation_delay_s,
            )
        if self._decision_diagnostics_enabled():
            self._record_decision_diagnostics(decision)
        self.last_iekf_update_time_s = self.time_s
        self.tracking_cycle_index += 1
        tracker.pending_samples.clear()
        tracker.last_correction_s = self.time_s
        self.scan_index = 0
        if decision.iekf_result is not None:
            self.diagnostics['accepted_update'].append(float(decision.iekf_result.accepted))
            self.diagnostics['innovation_score'].append(decision.iekf_result.normalized_innovation_score)
            self.diagnostics['measurement_scale'].append(decision.iekf_result.measurement_scale)
            self.diagnostics['minimum_covariance_eigenvalue'].append(tracker.filter.minimum_eigenvalue())
        self.diagnostics['iekf_update_time_ms'].append(decision.iekf_update_time_ms)
        self.diagnostics['controller_time_ms'].append(decision.controller_time_ms)
        self._recenter_and_command(self.active_index, decision.correction_deg, decision.quantized_target_az_el)
        self.command_issue_time_s = self.time_s
        self.expected_effective_time_s = self.time_s + self.config.effective_actuation_delay_s
        self.diagnostics['estimate_reference_time_s'].append(float(self.estimate_reference_time_s))
        self.diagnostics['command_issue_time_s'].append(float(self.command_issue_time_s))
        self.diagnostics['expected_effective_time_s'].append(float(self.expected_effective_time_s))
        batch_measurement_time_ms = self._measurement_compute_accumulator_ms[terminal.name]
        self.diagnostics['measurement_batch_compute_time_ms'].append(batch_measurement_time_ms)
        self.diagnostics['tracking_cycle_compute_time_ms'].append(
            batch_measurement_time_ms + (time.perf_counter() - cycle_update_started) * 1e3
        )
        self._measurement_compute_accumulator_ms[terminal.name] = 0.0

    def _record(self) -> None:
        """把当前步的链路状态写入内部缓存 (full) 与可选时间序列 (timeseries)。

        机械阶段与通信状态分离 (指南 5.1): 通信贯穿所有机械阶段, ``payload_active``
        恒为 1, 扫描与调整期间同样记录功率与速率; 支持域之外记零业务贡献。
        """
        self.current_state = self._link_state()
        state = self.current_state
        a, b = self.terminals
        a_beam = math.hypot(state['a_beam_h_deg'], state['a_beam_e_deg'])
        b_beam = math.hypot(state['b_beam_h_deg'], state['b_beam_e_deg'])
        mean_beam = 0.5 * (a_beam + b_beam)
        a_center = math.hypot(state['a_center_h_deg'], state['a_center_e_deg'])
        b_center = math.hypot(state['b_center_h_deg'], state['b_center_e_deg'])
        mean_center = 0.5 * (a_center + b_center)
        self.full['time_s'].append(self.time_s)
        self.full['distance_m'].append(state['distance_m'])
        self.full['los_rate_deg_s'].append(state['los_rate_deg_s'])
        self.full['mean_beam_error'].append(mean_beam)
        self.full['a_center_error'].append(a_center)
        self.full['b_center_error'].append(b_center)
        self.full['mean_center_error'].append(mean_center)
        self.full['pointing_loss'].append(state['pointing_loss_db'])
        true_pr_dbm = float(state['true_pr_dbm'])
        snr_db = float(state['snr_db'])
        support_valid = not bool(state['pattern_clamped_any'])
        spectral_efficiency_bps_hz = (
            math.log2(1.0 + 10.0 ** (snr_db / 10.0)) if support_valid else 0.0
        )
        available = support_valid and snr_db >= self.config.snr_threshold_db
        outage = int(not available)
        mechanical_phase = self._phase_for_record
        tracking_active = int(mechanical_phase != 'stationary')
        self.full['true_pr_dbm'].append(true_pr_dbm)
        self.full['snr_db'].append(snr_db)
        self.full['outage'].append(outage)
        self.full['spectral_efficiency_bps_hz'].append(spectral_efficiency_bps_hz)
        self.full['lock'].append(int(state['pointing_loss_db'] <= 3.0))
        self.full['pattern_clamped'].append(state['pattern_clamped_any'])
        self.full['tracking_active'].append(tracking_active)
        self.full['data_active'].append(1)
        self.full['data_snr_db'].append(snr_db)
        self.full['data_outage'].append(float(outage))
        self.full['data_spectral_efficiency_bps_hz'].append(spectral_efficiency_bps_hz)
        self.full['data_beam_error'].append(mean_beam)
        self.full['mechanical_phase'].append(mechanical_phase)
        self.full['payload_active'].append(1)
        self.full['pattern_support_valid'].append(int(support_valid))
        self.full['link_available'].append(int(available))
        self.link_metrics.add(self.time_s, self.config.dt_s, snr_db, support_valid,
                              bool(tracking_active), mean_beam)
        self.link_metrics_steady.add(self.time_s, self.config.dt_s, snr_db, support_valid,
                                     bool(tracking_active), mean_beam)
        if self.record_timeseries and self.step_index % self.config.record_stride == 0:
            self.timeseries.append({'time_s': self.time_s, 'distance_m': state['distance_m'], 'active_terminal': self.terminals[self.active_index].name, 'active_policy': self.policy.name, 'tracking_phase': self.phase, 'resource_phase': 'continuous_payload', 'mechanical_phase': mechanical_phase, 'is_data_slot': 1, 'a_true_h_deg': state['a_center_h_deg'], 'a_true_e_deg': state['a_center_e_deg'], 'b_true_h_deg': state['b_center_h_deg'], 'b_true_e_deg': state['b_center_e_deg'], 'a_center_error_deg': a_center, 'b_center_error_deg': b_center, 'mean_center_error_deg': mean_center, 'a_est_h_deg': a.tracker.estimate_error[0], 'a_est_e_deg': a.tracker.estimate_error[1], 'b_est_h_deg': b.tracker.estimate_error[0], 'b_est_e_deg': b.tracker.estimate_error[1], 'a_beam_error_deg': a_beam, 'b_beam_error_deg': b_beam, 'mean_beam_error_deg': mean_beam, 'pointing_loss_db': state['pointing_loss_db'], 'true_pr_dbm': true_pr_dbm, 'observed_pr_dbm': self.last_observation.observed_pr_dbm, 'measurement_noise_db': self.last_observation.measurement_noise_db, 'model_error_db': self.last_observation.model_error_db, 'measurement_sigma_db': self.config.observation.measurement_sigma_db, 'random_seed': self.trial_randomness.seed, 'pr_dbm': true_pr_dbm, 'snr_db': snr_db, 'outage_flag': outage, 'data_outage_flag': outage, 'spectral_efficiency_bps_hz': spectral_efficiency_bps_hz, 'link_available': int(available), 'payload_active': 1, 'tracking_active': tracking_active, 'a_q_est_dbm': a.tracker.estimate_q, 'b_q_est_dbm': b.tracker.estimate_q, 'a_q_dot_est_db_per_s': a.tracker.estimate_q_dot, 'b_q_dot_est_db_per_s': b.tracker.estimate_q_dot, 'a_omega_h_est_deg_s': a.tracker.estimate_velocity[0], 'a_omega_e_est_deg_s': a.tracker.estimate_velocity[1], 'b_omega_h_est_deg_s': b.tracker.estimate_velocity[0], 'b_omega_e_est_deg_s': b.tracker.estimate_velocity[1], 'power_measurement_event': int(self.power_measurement_event), 'pattern_support_valid': int(support_valid)})

    def _summarize(self) -> dict[str, Any]:
        """统计稳态跟瞄/通信指标及完整帧资源开销 (论文 summary.csv 字段)。"""
        times = np.asarray(self.full['time_s'], dtype=float)
        weights = interval_weights(times, self.config.metric_start_s, self.config.duration_s)
        steady = weights > 1e-12
        steady_weights = weights[steady]
        if not np.any(steady):
            raise ValueError('steady interval contains no samples')

        def values(name: str) -> np.ndarray:
            return np.asarray(self.full[name], dtype=float)[steady]

        def time_mean(name: str) -> float:
            return float(np.average(values(name), weights=steady_weights))

        def time_rmse(name: str) -> float:
            return float(np.sqrt(np.average(values(name) ** 2, weights=steady_weights)))
        pr = values('true_pr_dbm')
        mean_linear_pr = 10.0 * math.log10(float(np.average(10.0 ** (pr / 10.0), weights=steady_weights)))
        steady_lock_fraction = time_mean('lock')
        pattern_clamped_samples = int(np.sum(values('pattern_clamped')))
        measured_pattern_support_valid = int(pattern_clamped_samples == 0)
        tracking_pass = int(
            steady_lock_fraction >= self.config.convergence_lock_fraction
            and measured_pattern_support_valid == 1
        )
        refresh_samples_s = np.asarray(
            self.diagnostics['terminal_refresh_interval_s'], dtype=float
        )
        per_terminal_update_interval_s = (
            float(np.mean(refresh_samples_s)) if len(refresh_samples_s)
            else float('nan')
        )
        mean_hpbw_deg = 0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg)
        mean_los_rate_deg_s = time_mean('los_rate_deg_s')
        nominal_los_rate_v_over_r_deg_s = math.degrees(
            self.config.speed_mps / self.config.initial_distance_m
        )
        normalized_mobility = mean_los_rate_deg_s * per_terminal_update_interval_s / mean_hpbw_deg

        # Full-time continuous communication metrics over the unified
        # evaluation windows (guide 5.2).  Each record represents the interval
        # ``(t_prev, t]``; the initialisation snapshot has weight 0 and
        # intervals crossing the warm-up boundary are cropped by intersection.
        # One entry point shared by summary, exported CSV and GUI; no
        # data-fraction multiplication, no tracking+data fraction assertion.
        support_valid = np.asarray(self.full['pattern_support_valid'], dtype=bool)
        snr_db = np.asarray(self.full['snr_db'], dtype=float)
        mechanical_active = np.asarray(self.full['tracking_active'], dtype=bool)
        full_metrics = link_metrics_from_arrays(
            times, snr_db, support_valid, self.config.snr_threshold_db,
            self.config.service_rate_bps_hz, self.config.dt_s,
            mechanical_active=mechanical_active,
            window_start_s=0.0, window_end_s=self.config.duration_s,
        )
        steady_metrics = link_metrics_from_arrays(
            times, snr_db, support_valid, self.config.snr_threshold_db,
            self.config.service_rate_bps_hz, self.config.dt_s,
            mechanical_active=mechanical_active,
            window_start_s=self.config.metric_start_s, window_end_s=self.config.duration_s,
        )
        weights = interval_weights(times, self.config.metric_start_s, self.config.duration_s)
        beam_error = np.asarray(self.full['mean_beam_error'], dtype=float)
        beam_rmse_deg = float(np.sqrt(np.sum(weights * beam_error ** 2) / np.sum(weights)))
        data_beam_error = beam_error[steady]
        data_beam_error_rmse = beam_rmse_deg
        data_beam_error_p95 = float(np.percentile(data_beam_error, 95.0))
        center_error = values('mean_center_error')
        center_error_rmse = time_rmse('mean_center_error')

        total_full_outage_s = float(full_metrics['total_outage_s'])
        longest_full_outage_s = float(full_metrics['max_outage_s'])
        capture_candidates = np.flatnonzero(np.asarray(self.full['lock'], dtype=bool))
        capture_time_s = (
            float(self.full['time_s'][int(capture_candidates[0])])
            if len(capture_candidates) else float('nan')
        )
        support_criterion_met = int(
            (float(steady_metrics['availability']) > 0.999)
            or (math.isfinite(data_beam_error_rmse) and data_beam_error_rmse / mean_hpbw_deg < 0.5)
        )
        locks = np.asarray(self.full['lock'], dtype=bool)
        required_lock_steps = (
            max(1, int(math.ceil(
                self.config.convergence_consecutive_cycles
                * per_terminal_update_interval_s / self.config.dt_s
            ))) if math.isfinite(per_terminal_update_interval_s)
            else len(locks) + 1
        )
        convergence_index: int | None = None
        for index in range(max(0, len(locks) - required_lock_steps + 1)):
            if bool(np.all(locks[index:index + required_lock_steps])):
                convergence_index = index
                break

        def diagnostic_summary(name: str, statistic: str) -> float:
            samples = np.asarray(self.diagnostics[name], dtype=float)
            if not len(samples):
                return float('nan')
            if statistic == 'mean':
                return float(np.mean(samples))
            if statistic == 'p95':
                return float(np.percentile(samples, 95.0))
            if statistic == 'max':
                return float(np.max(samples))
            if statistic == 'min':
                return float(np.min(samples))
            raise ValueError(f'Unknown diagnostic statistic: {statistic}')

        # Mechanical activity fractions are diagnostics only; they are no
        # longer multiplied into the spectral efficiency.
        completed_frames_s = np.asarray(self.diagnostics['tracking_frame_duration_s'], dtype=float)
        completed_tracking_s = np.asarray(self.diagnostics['tracking_phase_duration_s'], dtype=float)
        completed_data_s = np.asarray(self.diagnostics['data_phase_duration_s'], dtype=float)
        if self.uses_tracking_scheduler and len(completed_frames_s):
            completed_tracking_s = completed_tracking_s[:len(completed_frames_s)]
            completed_data_s = completed_data_s[:len(completed_frames_s)]
            total_complete_frame_s = float(np.sum(completed_frames_s))
            tracking_overhead_fraction = float(np.sum(completed_tracking_s) / total_complete_frame_s)
            data_time_fraction = float(np.sum(completed_data_s) / total_complete_frame_s)
        elif not self.uses_tracking_scheduler:
            total_complete_frame_s = float(self.config.duration_s)
            tracking_overhead_fraction = 0.0
            data_time_fraction = 1.0
        else:
            total_complete_frame_s = 0.0
            tracking_overhead_fraction = float('nan')
            data_time_fraction = float('nan')
        continuous_se = float(steady_metrics['continuous_se'])
        availability = float(steady_metrics['availability'])

        return {
            'active_policy': self.policy.name,
            'method': self.policy.name,
            'random_seed': self.trial_randomness.seed,
            'measurement_sigma_db': self.config.observation.measurement_sigma_db,
            'model_error_sigma_db': self.config.observation.model_error_sigma_db,
            'servo_jitter_sigma_deg': self.config.observation.servo_jitter_sigma_deg,
            'speed_mps': self.config.speed_mps,
            'initial_distance_m': self.config.initial_distance_m,
            'command_time_s': self.config.command_time_s,
            'stationary_dwell_s': self.config.stationary_dwell_s,
            'data_duration_s': self.config.stationary_dwell_s,
            'effective_data_duration_s': diagnostic_summary('data_phase_duration_s', 'mean'),
            'quantization_deg': self.config.quantization_deg,
            'scan_loss_db': self.config.scan_loss_db,
            'estimator': self.config.estimator,
            'feedback': self.effective_config.feedback,
            'prediction_enabled': int(self.effective_config.prediction_enabled),
            'command_limit_deg': float(self.effective_config.command_limit_deg) if self.effective_config.command_limit_deg is not None else float('nan'),
            'actuation_delay_s': self.config.effective_actuation_delay_s,
            'snr_threshold_db': self.config.snr_threshold_db,
            'service_rate_bps_hz': self.config.service_rate_bps_hz,
            'energy_samples': self.config.energy_samples,
            'energy_censored': self.energy_censored,
            'mean_hpbw_deg': mean_hpbw_deg,
            'tracking_frame_duration_s': diagnostic_summary('tracking_frame_duration_s', 'mean'),
            'per_terminal_tracking_refresh_interval_s': per_terminal_update_interval_s,
            'per_terminal_update_interval_s': per_terminal_update_interval_s,
            'mean_los_rate_deg_s': mean_los_rate_deg_s,
            'nominal_los_rate_v_over_r_deg_s': nominal_los_rate_v_over_r_deg_s,
            'normalized_los_variation_per_tracking_cycle': normalized_mobility,
            'normalized_mean_los_rotation_per_update': normalized_mobility,
            'mean_beam_error_deg': time_mean('mean_beam_error'),
            'beam_error_rmse_deg': time_rmse('mean_beam_error'),
            'beam_error_p95_deg': float(np.percentile(values('mean_beam_error'), 95.0)),
            'normalized_pointing_error_rmse_over_hpbw': time_rmse('mean_beam_error') / mean_hpbw_deg,
            'center_error_mean_deg': time_mean('mean_center_error'),
            'center_error_rmse_deg': center_error_rmse,
            'center_error_p95_deg': float(np.percentile(center_error, 95.0)),
            'normalized_center_error_rmse_over_hpbw': center_error_rmse / mean_hpbw_deg,
            'data_beam_error_rmse_deg': data_beam_error_rmse,
            'data_beam_error_p95_deg': data_beam_error_p95,
            'normalized_data_beam_error_rmse_over_hpbw': data_beam_error_rmse / mean_hpbw_deg,
            'mean_pointing_loss_db': time_mean('pointing_loss'),
            'steady_lock_fraction_3db': steady_lock_fraction,
            'mean_pr_dbm': mean_linear_pr,
            'mean_snr_db': time_mean('snr_db'),
            'full_time_outage_probability': 1.0 - float(full_metrics['availability']),
            'total_full_time_outage_duration_s': total_full_outage_s,
            'longest_full_time_outage_duration_s': longest_full_outage_s,
            'capture_time_to_3db_lock_s': capture_time_s,
            'steady_outage_probability': float(1.0 - availability),
            'mean_spectral_efficiency_bps_hz': time_mean('spectral_efficiency_bps_hz'),
            'mean_data_snr_db': time_mean('data_snr_db'),
            'steady_data_outage_probability': float(1.0 - availability),
            'mean_data_spectral_efficiency_bps_hz': continuous_se,
            'tracking_overhead_fraction': tracking_overhead_fraction,
            'tracking_time_fraction': tracking_overhead_fraction,
            'data_time_fraction': data_time_fraction,
            'complete_frame_duration_s': total_complete_frame_s,
            'mean_effective_spectral_efficiency_bps_hz': continuous_se,
            'continuous_se': continuous_se,
            'availability': availability,
            'service_goodput': float(steady_metrics['service_goodput']),
            'max_outage_s': float(steady_metrics['max_outage_s']),
            'scan_fraction': float(steady_metrics['scan_fraction']),
            'scan_se': float(steady_metrics['scan_se']),
            'stationary_se': float(steady_metrics['stationary_se']),
            'scan_availability': float(steady_metrics['scan_availability']),
            'full_support_exit': int(full_metrics['full_support_exit']),
            'support_invalid_fraction': float(steady_metrics['support_invalid_fraction']),
            'payload_active_fraction': time_mean('payload_active'),
            'beam_rmse_deg': beam_rmse_deg,
            'lock_fraction': time_mean('lock'),
            'snr_p05_db': float(np.percentile(values('snr_db'), 5.0)),
            'snr_median_db': float(np.median(values('snr_db'))),
            'mean_sample_age_s': diagnostic_summary('sample_age_s', 'mean'),
            'max_sample_age_s': diagnostic_summary('sample_age_s', 'max'),
            'recenter_duration_s': self.last_recenter_duration_s,
            'configured_prediction_horizon_s': self.config.effective_actuation_delay_s,
            'prediction_horizon_s': self.config.effective_actuation_delay_s if self.effective_config.prediction_enabled else 0.0,
            'command_latency_s_mean': diagnostic_summary('command_latency_s', 'mean'),
            'command_latency_s_p95': diagnostic_summary('command_latency_s', 'p95'),
            'expected_command_latency_s_mean': diagnostic_summary('expected_command_latency_s', 'mean'),
            'estimate_to_issue_s_mean': diagnostic_summary('estimate_to_issue_s', 'mean'),
            'accepted_update_fraction': diagnostic_summary('accepted_update', 'mean'),
            'mean_innovation_score': diagnostic_summary('innovation_score', 'mean'),
            'p95_innovation_score': diagnostic_summary('innovation_score', 'p95'),
            'mean_measurement_scale': diagnostic_summary('measurement_scale', 'mean'),
            'min_covariance_eigenvalue': diagnostic_summary('minimum_covariance_eigenvalue', 'min') if self.diagnostics['minimum_covariance_eigenvalue'] else float('nan'),
            'iekf_update_time_ms_mean': diagnostic_summary('iekf_update_time_ms', 'mean'),
            'iekf_update_time_ms_p95': diagnostic_summary('iekf_update_time_ms', 'p95'),
            'controller_time_ms_mean': diagnostic_summary('controller_time_ms', 'mean'),
            'power_sample_time_ms_mean': diagnostic_summary('power_sample_time_ms', 'mean'),
            'measurement_batch_compute_time_ms_mean': diagnostic_summary('measurement_batch_compute_time_ms', 'mean'),
            'tracking_cycle_compute_time_ms_mean': diagnostic_summary('tracking_cycle_compute_time_ms', 'mean'),
            'tracking_cycle_compute_time_ms_p95': diagnostic_summary('tracking_cycle_compute_time_ms', 'p95'),
            'simulation_step_time_ms_mean': diagnostic_summary('simulation_step_time_ms', 'mean'),
            'simulation_step_time_ms_p95': diagnostic_summary('simulation_step_time_ms', 'p95'),
            'estimate_reference_time_s_mean': diagnostic_summary('estimate_reference_time_s', 'mean'),
            'command_issue_time_s_mean': diagnostic_summary('command_issue_time_s', 'mean'),
            'expected_effective_time_s_mean': diagnostic_summary('expected_effective_time_s', 'mean'),
            'command_complete_time_s_mean': diagnostic_summary('command_complete_time_s', 'mean'),
            'pattern_support_valid': measured_pattern_support_valid,
            'tracking_pass': tracking_pass,
            'convergence_success': int(convergence_index is not None and tracking_pass),
            'convergence_time_s': float(times[convergence_index]) if convergence_index is not None else float('nan'),
            'convergence_lock_fraction_threshold': self.config.convergence_lock_fraction,
            'convergence_definition_cycles': self.config.convergence_consecutive_cycles,
            'initial_phase_a_rad': self.trial_randomness.initial_phase_a_rad,
            'initial_phase_b_rad': self.trial_randomness.initial_phase_b_rad,
            'is_oracle_upper_bound': int(self.policy.is_oracle_upper_bound),
            'mobility_support_criterion_met': support_criterion_met,
            'policy_scan_interval_s': float(getattr(self.policy, 'scan_interval_s', float('nan'))),
        }

    def step(self) -> None:
        """推进一个仿真步 (dt_s)。"""
        step_started = time.perf_counter()
        self.power_measurement_event = False
        self._phase_for_record = self.phase
        for terminal in self.terminals:
            terminal.motion.step(self.config.dt_s)
            if self.uses_tracking_scheduler:
                terminal.tracker.predict(self.config.dt_s)
        if self.uses_tracking_scheduler and self.phase in {'three_point_scan', 'recenter'}:
            self.terminals[self.active_index].gimbal.advance(self.config.dt_s)
        self.time_s += self.config.dt_s
        self.step_index += 1
        self._refresh_servo_jitter()
        self.current_state = self._link_state()
        self.phase_elapsed_s += self.config.dt_s
        if self.uses_tracking_scheduler and self.phase_elapsed_s + 1e-12 >= self.phase_duration_s:
            self._complete_measurement()
        self._record()
        if self._decision_diagnostics_enabled():
            self._evaluate_future_checks()
        if self.config.benchmark_runtime:
            self.diagnostics['simulation_step_time_ms'].append(
                (time.perf_counter() - step_started) * 1e3
            )

    def get_snapshot(self) -> dict[str, Any]:
        """Return a copied, read-only display snapshot.

        This is the only interface intended for an observation GUI.  True LoS
        and true pointing errors are present solely for visualization and
        offline metrics; this method is never called by the IEKF, controller,
        gimbal, measurement model, or TDD scheduler.
        """

        def readonly(value: Any) -> np.ndarray:
            copied = np.asarray(value, dtype=float).copy()
            copied.flags.writeable = False
            return copied
        a, b = self.terminals
        state = self.current_state
        los_a = self._target_direction(0)
        los_b = self._target_direction(1)
        beam_a = self._beam_direction(a, 0)
        beam_b = self._beam_direction(b, 1)
        actual_a = self._mechanical_beam_direction(a, 0)
        actual_b = self._mechanical_beam_direction(b, 1)
        center_a = self._beam_center_direction(a)
        center_b = self._beam_center_direction(b)
        active = self.terminals[self.active_index]
        active_center = self._beam_center_direction(active)
        active_los = los_a if self.active_index == 0 else los_b
        active_actual_beam = actual_a if self.active_index == 0 else actual_b
        if self.phase == 'three_point_scan':
            display_phase = f'Scan S{self.scan_index + 1} / {len(self.scan_points)}'
            current_scan_index: int | None = self.scan_index
        elif self.phase == 'recenter':
            just_updated = self.last_iekf_update_time_s is not None and (
                self.time_s - self.last_iekf_update_time_s < 0.5 * self.config.command_time_s
            )
            display_phase = 'MAP Update → Recenter' if just_updated else 'Recenter'
            current_scan_index = None
        else:
            display_phase = 'Stationary dwell' if self.phase == 'stationary' else self.phase
            current_scan_index = None
        mechanical_phase = self.phase
        # 稳态窗口与 summary 主指标同口径; 全程窗口单独命名并显示时间范围。
        live_metrics = self.link_metrics_steady.snapshot()
        full_live_metrics = self.link_metrics.snapshot()
        complete_frames_s = np.asarray(self.diagnostics['tracking_frame_duration_s'], dtype=float)
        if self.uses_tracking_scheduler and len(complete_frames_s):
            complete_tracking_s = np.asarray(
                self.diagnostics['tracking_phase_duration_s'][:len(complete_frames_s)], dtype=float
            )
            complete_data_s = np.asarray(
                self.diagnostics['data_phase_duration_s'][:len(complete_frames_s)], dtype=float
            )
            total_complete_s = float(np.sum(complete_frames_s))
            tracking_fraction = float(np.sum(complete_tracking_s) / total_complete_s)
            data_fraction = float(np.sum(complete_data_s) / total_complete_s)
        elif not self.uses_tracking_scheduler:
            tracking_fraction, data_fraction = 0.0, 1.0
        else:
            tracking_fraction, data_fraction = float('nan'), float('nan')
        live_ages: list[float | None] = list(self.scan_measurement_ages_s)
        if self.phase == 'three_point_scan':
            for index, sample in enumerate(active.tracker.pending_samples):
                live_ages[index] = float(sample[2])
        support_valid = not bool(state['pattern_clamped_any'])
        link_available = support_valid and state['snr_db'] >= self.config.snr_threshold_db
        return {
            'time_s': float(self.time_s),
            'is_complete': bool(self.step_index >= self.config.steps),
            'initial_distance_m': float(self.config.initial_distance_m),
            'mean_hpbw_deg': float(0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg)),
            'active_terminal': active.name,
            'phase': display_phase,
            'scheduler_phase': self.phase,
            'mechanical_phase': mechanical_phase,
            'pos_A': readonly(a.motion.position), 'pos_B': readonly(b.motion.position),
            'los_A': readonly(los_a), 'los_B': readonly(los_b),
            'beam_A': readonly(beam_a), 'beam_B': readonly(beam_b),
            'center_A': readonly(center_a), 'center_B': readonly(center_b),
            'true_error_A_HE': readonly([state['a_center_h_deg'], state['a_center_e_deg']]),
            'true_error_B_HE': readonly([state['b_center_h_deg'], state['b_center_e_deg']]),
            'est_error_A_HE': readonly(a.tracker.estimate_error),
            'est_error_B_HE': readonly(b.tracker.estimate_error),
            'beam_error_A_deg': float(math.hypot(state['a_beam_h_deg'], state['a_beam_e_deg'])),
            'beam_error_B_deg': float(math.hypot(state['b_beam_h_deg'], state['b_beam_e_deg'])),
            'true_pr_dbm': float(state['true_pr_dbm']),
            'observed_pr_dbm': float(self.last_observation.observed_pr_dbm),
            'measurement_noise_db': float(self.last_observation.measurement_noise_db),
            'model_error_db': float(self.last_observation.model_error_db),
            'measurement_sigma_db': float(self.config.observation.measurement_sigma_db),
            'pr_dbm': float(state['true_pr_dbm']), 'snr_db': float(state['snr_db']),
            'pattern_support_valid': int(support_valid),
            'link_available': int(link_available),
            'outage_flag': int(not link_available),
            'data_outage_flag': int(not link_available),
            'payload_active': 1,
            'tracking_active': int(mechanical_phase != 'stationary'),
            'active_policy': self.policy.name,
            'prediction_enabled': int(self.effective_config.prediction_enabled),
            'estimator': self.config.estimator,
            'feedback': self.effective_config.feedback,
            'command_limit_deg': float(self.effective_config.command_limit_deg) if self.effective_config.command_limit_deg is not None else float('nan'),
            'normalized_los_variation_per_tracking_cycle': float(
                state['los_rate_deg_s'] * (float(np.mean(self.diagnostics['terminal_refresh_interval_s']))
                                            if self.diagnostics['terminal_refresh_interval_s'] else 0.0) /
                (0.5 * (self.pattern.qa.h_hpbw_deg + self.pattern.qa.e_hpbw_deg))
            ),
            'tracking_overhead_fraction': tracking_fraction,
            'data_time_fraction': data_fraction,
            # 稳态窗口 (与 summary 主指标一致) 与全程窗口分别命名。
            'effective_spectral_efficiency_bps_hz': float(live_metrics['continuous_se']),
            'availability': float(live_metrics['availability']),
            'service_goodput': float(live_metrics['service_goodput']),
            'max_outage_s': float(live_metrics['max_outage_s']),
            'support_exits': int(live_metrics['support_exits']),
            'metric_window_start_s': float(self.config.metric_start_s),
            'metric_window_end_s': float(self.config.duration_s),
            'full_continuous_se': float(full_live_metrics['continuous_se']),
            'full_availability': float(full_live_metrics['availability']),
            'full_max_outage_s': float(full_live_metrics['max_outage_s']),
            'full_window_start_s': 0.0,
            'full_window_end_s': float(self.config.duration_s),
            'pointing_loss_db': float(state['pointing_loss_db']), 'distance_m': float(state['distance_m']),
            'selected_scan_loss_db': self.current_scan_plan.scan_loss_db if self.current_scan_plan else self.config.scan_loss_db,
            'service_horizon_s': self._last_decision_display.get('service_horizon_s'),
            'pointing_decision_terminal': self._last_decision_display.get('terminal'),
            'pointing_decision_time_s': self._last_decision_display.get('time_s'),
            'latency_target': readonly(self._last_decision_display.get('latency_target', [np.nan, np.nan])),
            'service_horizon_target': readonly(self._last_decision_display.get('service_target', [np.nan, np.nan])),
            'selected_quantized_target': readonly(self._last_decision_display.get('selected_quantized_command', [np.nan, np.nan])),
            'scan_points_local': tuple(readonly(point) for point in self.scan_points),
            'current_scan_index': current_scan_index,
            'scan_completed_mask': tuple(self.scan_completed_mask),
            'scan_measured_powers_dbm': tuple(self.scan_measured_powers_dbm),
            'scan_true_powers_dbm': tuple(self.scan_true_powers_dbm),
            'scan_measurement_noise_db': tuple(self.scan_measurement_noise_db),
            'scan_measurement_ages_s': tuple(live_ages),
            'measurement_event_id': int(self.measurement_event_id),
            'last_measurement_scan_index': self.last_measurement_scan_index,
            'last_measurement_time_s': self.last_measurement_time_s,
            'tracking_center_local': readonly([0.0, 0.0]),
            'true_los_local': readonly(local_error_deg(*az_el_from_direction(active_center), active_los)),
            'current_beam_local': readonly(local_error_deg(*az_el_from_direction(active_center), active_actual_beam)),
            'actual_mechanical_beam_local': readonly(local_error_deg(*az_el_from_direction(active_center), active_actual_beam)),
            'iekf_est_error_local': readonly(active.tracker.estimate_error),
            'last_correction_local': None if self.last_correction_local is None else readonly(self.last_correction_local),
            'old_center_local': None if self.old_center_local is None else readonly(self.old_center_local),
            'new_center_local': None if self.new_center_local is None else readonly(self.new_center_local),
            'recenter_terminal': self.recenter_terminal_name,
        }

    def get_history(self) -> dict[str, np.ndarray]:
        """Return copied complete history, distinct from any short GUI buffer."""
        if not self.timeseries:
            return {}
        return {name: np.asarray([row[name] for row in self.timeseries]).copy() for name in self.timeseries[0]}

    def run_to_completion(self) -> SimulationResult:
        """Run remaining numerical steps without involving any display code."""
        while self.step_index < self.config.steps:
            self.step()
        return SimulationResult(self._summarize(), self.timeseries)

    def run(self) -> SimulationResult:
        """运行整个仿真,返回 (summary, timeseries)。"""
        while self.step_index < self.config.steps:
            self.step()
        return SimulationResult(summary=self._summarize(), timeseries=self.timeseries)

# =============================================================================
# Paper experiment helpers and main entry point
# =============================================================================

EmpiricalEKF = IEKF  # Compatibility name for existing notebooks.


def stacked_information_matrix(filter_: IEKF,
                               samples: Iterable[tuple[np.ndarray, float, float]],
                               measurement_sigma_db: float | None = None) -> dict[str, Any]:
    """Offline local information diagnostic for repeated asynchronous samples.

    This function never enters the controller or changes ``filter_``.  It
    evaluates the existing measurement Jacobian at the supplied state and
    returns ``J = H.T @ R^-1 @ H``.  A rank below six is expected for one
    three-sample update; the dynamic prior and repeated observations accumulate
    information over tracking cycles.
    """
    rows = [filter_.measurement_model(np.asarray(offset, dtype=float), float(age_s))[1]
            for offset, _, age_s in samples]
    if not rows:
        raise ValueError('stacked_information_matrix needs at least one sample')
    matrix = np.vstack(rows)
    sigma_db = measurement_sigma_db or filter_.measurement_design_sigma_db
    if sigma_db <= 0.0:
        raise ValueError('measurement_sigma_db must be positive')
    information = matrix.T @ matrix / sigma_db ** 2
    eigenvalues = np.linalg.eigvalsh(0.5 * (information + information.T))
    return {
        'matrix': information,
        'jacobian_stack': matrix,
        'rank': int(np.linalg.matrix_rank(matrix)),
        'condition_number': float(np.linalg.cond(information)),
        'eigenvalues': eigenvalues,
    }


@dataclass(frozen=True)
class MonteCarloConfig:
    """Reproducible Monte Carlo settings shared by all fair-comparison runs."""

    runs: int = 500
    base_seed: int = 20260903
    bootstrap_resamples: int = 1000

    def validate(self) -> None:
        if self.runs <= 0:
            raise ValueError('runs must be positive')
        if self.bootstrap_resamples <= 0:
            raise ValueError('bootstrap_resamples must be positive')


@dataclass
class MonteCarloResult:
    raw_trials: pd.DataFrame
    aggregate: pd.DataFrame


def _wilson_interval(successes: int, trials: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson interval for a binary per-trial outcome."""
    if trials <= 0:
        return (float('nan'), float('nan'))
    proportion = successes / trials
    denominator = 1.0 + z_score ** 2 / trials
    centre = (proportion + z_score ** 2 / (2.0 * trials)) / denominator
    radius = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials + z_score ** 2 / (4.0 * trials ** 2)
    ) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator,
                        resamples: int) -> tuple[float, float]:
    """Deterministic percentile bootstrap confidence interval for a mean."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (float('nan'), float('nan'))
    if len(finite) == 1:
        return (float(finite[0]), float(finite[0]))
    indexes = rng.integers(0, len(finite), size=(resamples, len(finite)))
    means = np.mean(finite[indexes], axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_monte_carlo(
    case_factory: Callable[[int], tuple[SimulationConfig, MeasuredAntennaPattern]],
    policy_factory: Callable[[], TrackingPolicy],
    mc_config: MonteCarloConfig,
) -> MonteCarloResult:
    """Run one policy with explicit seeds and common-random-number support.

    ``case_factory(seed)`` returns an otherwise fixed ``(config, pattern)``.
    This runner overwrites only the observation seed.  Calling it once for each
    policy with the same factory/config creates the same trajectory, initial
    state, and event-indexed measurement-noise realization for every trial.
    """
    mc_config.validate()
    rows: list[dict[str, Any]] = []
    for trial_index in range(mc_config.runs):
        seed = mc_config.base_seed + trial_index
        base_config, pattern = case_factory(seed)
        observation = replace(base_config.observation, seed=seed)
        config = replace(base_config, observation=observation)
        trial_randomness = TrialRandomness.from_seed(seed)
        simulation = BidirectionalMechanicalSimulation(
            config, pattern, record_timeseries=False, policy=policy_factory(),
            trial_randomness=trial_randomness,
        )
        result = simulation.run_to_completion()
        rows.append({
            'trial_index': trial_index,
            'random_seed': seed,
            **result.summary,
        })
    raw_trials = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    numeric_columns = raw_trials.select_dtypes(include=[np.number]).columns
    bootstrap_rng = np.random.default_rng(mc_config.base_seed)
    for column in numeric_columns:
        if column in {'trial_index', 'random_seed'}:
            continue
        values = raw_trials[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        if column == 'convergence_success':
            ci_low, ci_high = _wilson_interval(int(np.sum(finite)), len(finite))
        else:
            ci_low, ci_high = _bootstrap_interval(finite, bootstrap_rng, mc_config.bootstrap_resamples)
        aggregate_rows.append({
            'metric': column,
            'runs': len(finite),
            'mean': float(np.mean(finite)),
            'std': float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            'median': float(np.median(finite)),
            'p05': float(np.percentile(finite, 5.0)),
            'p95': float(np.percentile(finite, 95.0)),
            'ci95_low': ci_low,
            'ci95_high': ci_high,
        })
    return MonteCarloResult(raw_trials, pd.DataFrame(aggregate_rows))


@dataclass(frozen=True)
class ExperimentCase:
    """One variable-isolated paper experiment condition."""

    experiment: str
    swept_parameter: str
    swept_value: float
    config: SimulationConfig
    secondary_parameter: str | None = None
    secondary_value: float | None = None


def joint_experiment_registry(pattern):
    from joint_experiments import build_registry
    return build_registry(pattern)


def experiment_registry(pattern: MeasuredAntennaPattern) -> dict[str, list[ExperimentCase]]:
    """Explicit paper matrix; it is a registry, not an automatic mega-run."""
    reference = representative_config(pattern)
    mean_hpbw_deg = 0.5 * (pattern.qa.h_hpbw_deg + pattern.qa.e_hpbw_deg)

    def cases(name: str, parameter: str, values: Iterable[float],
              configure: Callable[[float], SimulationConfig]) -> list[ExperimentCase]:
        return [ExperimentCase(name, parameter, float(value), configure(float(value))) for value in values]

    return {
        **joint_experiment_registry(pattern),
        'representative': [ExperimentCase('representative', 'reference', 1.0, reference)],
        'baseline': [ExperimentCase('baseline', 'reference', 1.0, reference)],
        'mobility': cases('mobility', 'speed_mps', [0.5, 1.0, 2.0, 4.0, 8.0],
                          lambda value: replace(reference, speed_mps=value)),
        'distance': cases('distance', 'initial_distance_m', [100, 250, 500, 1000, 1500, 2000],
                          lambda value: replace(reference, initial_distance_m=value)),
        'mechanical': cases('mechanical', 'command_time_s', [0.05, 0.1, 0.2, 0.4],
                            lambda value: replace(reference, command_time_s=value)),
        'initial_error': cases('initial_error', 'initial_offset_over_hpbw',
                               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                               lambda value: replace(
                                   reference, initial_offset_deg=value * mean_hpbw_deg,
                                   randomize_initial_error_phase=True,
                               )),
        'noise': cases('noise', 'measurement_sigma_db', [0.05, 0.1, 0.25, 0.5, 1.0],
                       lambda value: replace(
                           reference,
                           observation=replace(reference.observation, measurement_sigma_db=value),
                           # The main noise sweep keeps R physically matched to
                           # the configured observation statistics.  A user can
                           # set a different value explicitly for R sensitivity.
                           measurement_design_sigma_db=value,
                       )),
        'scan_radius': cases('scan_radius', 'scan_loss_db', [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                             lambda value: replace(reference, scan_loss_db=value)),
        'prediction_horizon_sweep': cases('prediction_horizon_sweep', 'prediction_horizon_s',
                                          [0.0, 0.15, 0.3, 0.45, 0.6, 0.9],
                                          lambda value: replace(
                                              reference,
                                              prediction_horizon_commands=value / reference.command_time_s,
                                          )),
        'robustness': cases('robustness', 'servo_jitter_sigma_deg', [0, 0.01, 0.02, 0.05, 0.1],
                            lambda value: replace(reference, observation=replace(
                                reference.observation, servo_jitter_sigma_deg=value))),
        'q_sensitivity': cases('q_sensitivity', 'q_process_scale', [0.1, 0.3, 1, 3, 10],
                                lambda value: replace(reference, q_process_scale=value)),
        'r_sensitivity': cases('r_sensitivity', 'r_measurement_scale', [0.5, 1, 2, 4],
                                lambda value: replace(reference, r_measurement_scale=value)),
        'data_overhead': cases('data_overhead', 'stationary_dwell_s', [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                               lambda value: replace(reference, stationary_dwell_s=value)),
        'maneuver': [ExperimentCase('maneuver', 'sinusoidal_frequency_hz', 0.5, replace(
            reference, maneuver='sinusoidal', sinusoidal_speed_delta_mps=1.0,
            sinusoidal_frequency_hz=0.5,
        ))],
        'mobility_stress_test': [
            ExperimentCase(
                'mobility_stress_test', 'speed_mps', float(speed_mps),
                replace(reference, speed_mps=float(speed_mps), initial_distance_m=float(distance_m)),
                'initial_distance_m', float(distance_m),
            )
            for speed_mps in (2, 5, 10, 20, 40)
            for distance_m in (100, 250, 500, 1000)
        ],
        # Main continuous-communication paired ablation (guide section 9):
        # 7 conditions x {prediction, no_prediction} x seeds 81000-81029.
        # Same MAP estimator, unit feedback, scan plan and observation law for
        # both groups; only the angular-rate feed-forward term is switched.
        'continuous_prediction': [
            ExperimentCase('continuous_prediction', 'speed_mps', float(speed_mps),
                           continuous_ablation_config(pattern, speed_mps=speed_mps,
                                                      command_time_s=command_s),
                           'command_time_s', float(command_s))
            for command_s, speed_mps in [(0.1, 0.0), (0.1, 1.0), (0.1, 2.0),
                                         (0.1, 3.0), (0.1, 5.0), (0.1, 8.0),
                                         (0.2, 2.0)]
        ],
    }


def continuous_ablation_config(pattern: MeasuredAntennaPattern, speed_mps: float,
                               command_time_s: float) -> SimulationConfig:
    """Configuration of the continuous-communication paired ablation.

    Reproduces the archived 420-record setup: 1 km range, -15 dBm TX, 256
    energy samples, 1.6 s stationary dwell, 0.75 dB nominal scan loss, 6 dB
    availability threshold, 2 bit/s/Hz fixed-service proxy, first 25% warm-up
    excluded, 48 s duration, 0.02 s step, command time 0.1 s (0.2 s variant at
    2 m/s).  The nominal recenter command time is the causal actuation delay.
    """
    mean_hpbw = 0.5 * (pattern.qa.h_hpbw_deg + pattern.qa.e_hpbw_deg)
    return SimulationConfig(
        duration_s=48.0,
        dt_s=0.02,
        initial_distance_m=1000.0,
        speed_mps=float(speed_mps),
        initial_offset_deg=0.2 * mean_hpbw,
        tx_power_dbm=-15.0,
        command_time_s=float(command_time_s),
        stationary_dwell_s=1.6,
        scan_loss_db=0.75,
        estimator='energy_map',
        feedback='unit',
        prediction_enabled=True,
        actuation_delay_s=float(command_time_s),
        observation=PowerObservationConfig(measurement_sigma_db=0.0, seed=41000),
        snr_threshold_db=6.0,
        service_rate_bps_hz=2.0,
        warm_up_fraction=0.25,
        energy_samples=256,
        angular_psd=0.0025,
        q_psd=0.25,
        randomize_initial_error_phase=True,
    )


def policy_registry() -> dict[str, Callable[[], TrackingPolicy]]:
    """Named policy factories.  Oracle remains labelled as an upper bound."""
    return {
        'no_tracking_static_beam': NoTrackingStaticBeam,
        'exhaustive_scanning': ExhaustiveScanningBaseline,
        'local_three_point_no_estimator': LocalThreePointNoEstimator,
        'synchronous_three_point': SynchronousThreePointBaseline,
        'synchronous_fixed_horizon_prediction': FixedHorizonPredictionBaseline,
        'age_aware_no_prediction': AgeAwareNoPrediction,
        'proposed_age_aware_predictive': ProposedAgeAwarePredictive,
        'motion_prediction': MotionPredictivePolicy,
        'motion_no_prediction': MotionNoPredictionPolicy,
        'fixed_service_horizon': FixedServiceHorizonPolicy,
        'active_latency': ActiveLatencyPolicy,
        'active_service_horizon': ActiveServiceHorizonPolicy,
        'external_literature_adapter': ExternalLiteratureBaselineAdapter,
        'oracle_perfect_pointing_no_overhead': OraclePerfectPointingNoOverhead,
    }


def baseline_policy_registry() -> dict[str, Callable[[], TrackingPolicy]]:
    """Paper baselines, including the requested periodic exhaustive scans."""
    return {
        'no_tracking_static_beam': NoTrackingStaticBeam,
        'exhaustive_scan_1s': lambda: ExhaustiveScanningBaseline(1.0),
        'exhaustive_scan_2s': lambda: ExhaustiveScanningBaseline(2.0),
        'exhaustive_scan_5s': lambda: ExhaustiveScanningBaseline(5.0),
        'local_three_point_no_estimator': LocalThreePointNoEstimator,
        'synchronous_three_point': SynchronousThreePointBaseline,
        'synchronous_fixed_horizon_prediction': FixedHorizonPredictionBaseline,
        'age_aware_no_prediction': AgeAwareNoPrediction,
        'proposed_age_aware_predictive': ProposedAgeAwarePredictive,
        'external_literature_adapter': ExternalLiteratureBaselineAdapter,
        'oracle_perfect_pointing_no_overhead': OraclePerfectPointingNoOverhead,
    }


def policy_metadata(policy: TrackingPolicy) -> dict[str, int | float | str]:
    """Fair-comparison fields exported with every baseline result table."""
    return {
        'method': policy.name,
        'tracking_measurements_per_update': policy.tracking_measurements_per_update,
        'uses_sample_age': int(policy.uses_sample_age),
        'estimates_angular_rate': int(policy.estimates_angular_rate),
        'uses_prediction': int(policy.uses_prediction),
        'uses_tracking_scheduler': int(policy.uses_tracking_scheduler),
        'is_oracle_upper_bound': int(policy.is_oracle_upper_bound),
        'scan_interval_s': float(getattr(policy, 'scan_interval_s', float('nan'))),
    }


def load_measured_pattern(path: Path | None = None) -> MeasuredAntennaPattern:
    """Load the fixed measured H/E workbook used by this project."""
    candidates = [WORKSPACE / '实验记录' / '20260731_TWC_三点单端直接功率跟瞄' /
                  'paper' / 'TWC_CA_J3MT_simplified_20260801' / 'data' /
                  DEFAULT_ANTENNA_FILENAME,
                  WORKSPACE / DEFAULT_ANTENNA_FILENAME]
    source = path or next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None or not source.is_file():
        raise FileNotFoundError(f'Measured antenna workbook not found: {source}')
    return MeasuredAntennaPattern.from_excel(source)


def representative_config(pattern: MeasuredAntennaPattern) -> SimulationConfig:
    """1 km, 2 m/s representative case with a 0.32-HPBW initial offset."""
    mean_hpbw = 0.5 * (pattern.qa.h_hpbw_deg + pattern.qa.e_hpbw_deg)
    return SimulationConfig(
        initial_distance_m=1000.0,
        speed_mps=2.0,
        initial_offset_deg=0.32 * mean_hpbw,
        stationary_dwell_s=0.8,
    )


def run_simulation(config: SimulationConfig, pattern: MeasuredAntennaPattern) -> SimulationResult:
    """Run one paper case through the single simulation core."""
    return BidirectionalMechanicalSimulation(config, pattern).run()


def _write_runtime_summary(simulation: BidirectionalMechanicalSimulation, output_path: Path) -> None:
    """Write algorithm-only runtime statistics; GUI rendering is never included."""
    rows: list[dict[str, float | str]] = []
    for name in (
        'power_sample_time_ms', 'measurement_batch_compute_time_ms',
        'iekf_update_time_ms', 'controller_time_ms',
        'tracking_cycle_compute_time_ms', 'simulation_step_time_ms',
    ):
        values = np.asarray(simulation.diagnostics[name], dtype=float)
        if len(values):
            rows.append({
                'metric': name,
                'mean': float(np.mean(values)),
                'p95': float(np.percentile(values, 95.0)),
                'max': float(np.max(values)),
            })
    physical_acquisition_s = (
        simulation.policy.tracking_measurements_per_update * simulation.config.command_time_s
    )
    rows.append({
        'metric': 'mechanical_measurement_acquisition_time_s',
        'mean': float(physical_acquisition_s),
        'p95': float(physical_acquisition_s),
        'max': float(physical_acquisition_s),
    })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_method_complexity_table(output_path: Path) -> None:
    """Write the fixed algorithmic-complexity table used in the paper text."""
    rows = [
        {
            'method': 'Exhaustive beam sweeping', 'complexity': 'O(N)',
            'measurement_count_per_update': 9, 'state_dimension': 0,
            'uses_iekf': 0, 'uses_prediction': 0,
        },
        {
            'method': 'Local three-point without IEKF', 'complexity': 'O(3)',
            'measurement_count_per_update': 3, 'state_dimension': 0,
            'uses_iekf': 0, 'uses_prediction': 0,
        },
        {
            'method': 'Synchronous three-point IEKF', 'complexity': 'O(n^2)',
            'measurement_count_per_update': 3, 'state_dimension': 6,
            'uses_iekf': 1, 'uses_prediction': 0,
        },
        {
            'method': 'Age-aware IEKF prediction (proposed)', 'complexity': 'O(n^2)',
            'measurement_count_per_update': 3, 'state_dimension': 6,
            'uses_iekf': 1, 'uses_prediction': 1,
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_paper_experiment(pattern: MeasuredAntennaPattern, experiment: str,
                         output_root: Path, mc_config: MonteCarloConfig,
                         duration_s: float | None = None) -> None:
    """Run one registry entry and persist only CSV data required for paper figures."""
    registry = experiment_registry(pattern)
    if experiment not in registry:
        raise ValueError(f"Unknown experiment '{experiment}'; choose one of {sorted(registry)}")
    cases = registry[experiment]
    output_dir = output_root / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_method_complexity_table(output_root / 'method_complexity.csv')
    if experiment in joint_experiment_registry(pattern):
        from joint_experiments import run_joint_experiment
        run_joint_experiment(pattern, cases, output_dir, duration_s=duration_s,
                             runs=mc_config.runs)
        return
    if experiment == 'representative':
        config = cases[0].config
        if duration_s is not None:
            config = replace(config, duration_s=duration_s)
        config = replace(config, benchmark_runtime=True)
        simulation = BidirectionalMechanicalSimulation(config, pattern)
        result = simulation.run_to_completion()
        pd.DataFrame(result.timeseries).to_csv(output_dir / 'timeseries.csv', index=False)
        pd.DataFrame([result.summary]).to_csv(output_dir / 'summary.csv', index=False)
        _write_runtime_summary(simulation, output_dir / 'runtime_summary.csv')
        return

    if experiment == 'continuous_prediction':
        _run_continuous_prediction_experiment(pattern, cases, output_dir, duration_s)
        return

    if experiment == 'baseline':
        policy_factories = baseline_policy_registry()
    elif experiment == 'mobility_stress_test':
        # Mobility stress reports every implementable comparison method; the
        # external adapter remains intentionally unavailable until cited code
        # is supplied by the researcher.
        policy_factories = {
            name: factory for name, factory in baseline_policy_registry().items()
            if name != 'external_literature_adapter'
        }
    elif experiment == 'prediction_horizon_sweep':
        policy_factories = {
            'age_aware_no_prediction': AgeAwareNoPrediction,
            'synchronous_fixed_horizon_prediction': FixedHorizonPredictionBaseline,
            'proposed_age_aware_predictive': ProposedAgeAwarePredictive,
        }
    else:
        policy_factories = {'proposed_age_aware_predictive': ProposedAgeAwarePredictive}
    raw_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    for case in cases:
        case_config = replace(case.config, duration_s=duration_s) if duration_s is not None else case.config
        for policy_name, factory in policy_factories.items():
            candidate = factory()
            if isinstance(candidate, ExternalLiteratureBaselineAdapter) and not candidate.is_available:
                warnings.warn(
                    'External literature baseline is not run: attach the selected cited method '
                    'to ExternalLiteratureBaselineAdapter first.', RuntimeWarning,
                )
                continue
            metadata = policy_metadata(candidate)
            def case_factory(seed: int, config: SimulationConfig = case_config) -> tuple[SimulationConfig, MeasuredAntennaPattern]:
                del seed
                return config, pattern

            result = run_monte_carlo(case_factory, factory, mc_config)
            raw = result.raw_trials.copy()
            raw.insert(0, 'experiment', case.experiment)
            raw.insert(1, 'swept_parameter', case.swept_parameter)
            raw.insert(2, 'swept_value', case.swept_value)
            raw.insert(3, 'secondary_parameter', case.secondary_parameter or '')
            raw.insert(4, 'secondary_value', (
                float(case.secondary_value) if case.secondary_value is not None else float('nan')
            ))
            raw.insert(5, 'policy', policy_name)
            for column, value in reversed(list(metadata.items())):
                if column not in raw:
                    raw.insert(6, column, value)
            aggregate = result.aggregate.copy()
            aggregate.insert(0, 'experiment', case.experiment)
            aggregate.insert(1, 'swept_parameter', case.swept_parameter)
            aggregate.insert(2, 'swept_value', case.swept_value)
            aggregate.insert(3, 'secondary_parameter', case.secondary_parameter or '')
            aggregate.insert(4, 'secondary_value', (
                float(case.secondary_value) if case.secondary_value is not None else float('nan')
            ))
            aggregate.insert(5, 'policy', policy_name)
            for column, value in reversed(list(metadata.items())):
                if column not in aggregate:
                    aggregate.insert(6, column, value)
            raw_frames.append(raw)
            aggregate_frames.append(aggregate)
    raw_trials = pd.concat(raw_frames, ignore_index=True)
    aggregate_data = pd.concat(aggregate_frames, ignore_index=True)
    if experiment == 'baseline':
        oracle = raw_trials.loc[
            raw_trials['policy'] == 'oracle_perfect_pointing_no_overhead',
            ['random_seed', 'mean_snr_db', 'mean_effective_spectral_efficiency_bps_hz'],
        ].rename(columns={
            'mean_snr_db': '_oracle_mean_snr_db',
            'mean_effective_spectral_efficiency_bps_hz': '_oracle_mean_rate',
        })
        raw_trials = raw_trials.merge(oracle, on='random_seed', how='left')
        raw_trials['oracle_snr_gap_db'] = raw_trials['_oracle_mean_snr_db'] - raw_trials['mean_snr_db']
        raw_trials['oracle_rate_ratio'] = raw_trials['mean_effective_spectral_efficiency_bps_hz'] / raw_trials['_oracle_mean_rate']
        raw_trials.drop(columns=['_oracle_mean_snr_db', '_oracle_mean_rate'], inplace=True)
        oracle_rows: list[dict[str, Any]] = []
        statistic_rng = np.random.default_rng(mc_config.base_seed)
        for policy_name, group in raw_trials.groupby('policy'):
            if policy_name == 'oracle_perfect_pointing_no_overhead':
                continue
            for metric in ('oracle_snr_gap_db', 'oracle_rate_ratio'):
                values = group[metric].to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap_interval(values, statistic_rng, mc_config.bootstrap_resamples)
                oracle_rows.append({
                    'experiment': 'baseline', 'swept_parameter': 'reference',
                    'swept_value': 1.0, 'policy': policy_name, 'metric': metric,
                    'runs': len(values), 'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'median': float(np.median(values)), 'p05': float(np.percentile(values, 5.0)),
                    'p95': float(np.percentile(values, 95.0)),
                    'ci95_low': ci_low, 'ci95_high': ci_high,
                })
        aggregate_data = pd.concat([aggregate_data, pd.DataFrame(oracle_rows)], ignore_index=True)
    if experiment == 'mobility_stress_test':
        support_rows: list[dict[str, float | int | str]] = []
        grouped = raw_trials.groupby(
            ['policy', 'swept_value', 'secondary_value'], dropna=False, sort=True
        )
        for (policy_name, speed_mps, distance_m), group in grouped:
            mean_outage = float(group['steady_data_outage_probability'].mean())
            mean_rmse = float(group['normalized_data_beam_error_rmse_over_hpbw'].mean())
            mean_rho = float(group['normalized_los_variation_per_tracking_cycle'].mean())
            supported = int((mean_outage < 1e-3) or (mean_rmse < 0.5))
            support_rows.append({
                'policy': str(policy_name), 'speed_mps': float(speed_mps),
                'initial_distance_m': float(distance_m), 'rho_m': mean_rho,
                'mean_data_outage_probability': mean_outage,
                'mean_normalized_data_beam_rmse': mean_rmse,
                'mobility_supported': supported,
            })
        support = pd.DataFrame(support_rows)
        if not support.empty:
            maxima = (
                support.loc[support['mobility_supported'] == 1]
                .groupby('policy', as_index=False)['rho_m'].max()
                .rename(columns={'rho_m': 'maximum_supported_mobility_rho_m'})
            )
            support = support.merge(maxima, on='policy', how='left')
        support.to_csv(output_dir / 'maximum_supported_mobility.csv', index=False)
    raw_trials.to_csv(output_dir / 'raw_trials.csv', index=False)
    aggregate_data.to_csv(output_dir / 'aggregate.csv', index=False)


def _continuous_manifest_fingerprint(manifest_payload: dict[str, Any]) -> str:
    from experiment_contract import manifest_fingerprint
    return manifest_fingerprint(manifest_payload)


def _build_continuous_manifest(pattern, cases, duration_s=None):
    from dataclasses import asdict
    from experiment_contract import SCHEMA_VERSION, pattern_identity
    import hashlib
    import json
    methods = ['prediction', 'no_prediction']
    seeds = [81000,81001,81002] if duration_s is not None else list(range(81000,81030))
    configs = {}
    for case in cases:
        label = _continuous_case_label(case)
        if label in configs:
            raise ValueError('Duplicate condition label: '+label)
        base = replace(case.config, duration_s=duration_s) if duration_s is not None else case.config
        configs[label] = {}
        for method in methods:
            cfg = replace(base,prediction_enabled=method=='prediction')
            probe = BidirectionalMechanicalSimulation(cfg,pattern,record_timeseries=False)
            # Canonical JSON containers keep nested tuple-valued design grids
            # identical before and after manifest serialization/resume.
            configs[label][method] = json.loads(json.dumps(asdict(probe.effective_config)))
    source_files = [SCRIPT_PATH, SCRIPT_PATH.parent/'experiment_contract.py',
                    SCRIPT_PATH.parent/'main_THZ_220GHz_tracking_gui.py']
    manifest = dict(schema_version=SCHEMA_VERSION, experiment='continuous_prediction',
        purpose='Motion prediction during continuous communication',
        primary_metric='Time-weighted availability after declared warm-up',
        methods=methods,seeds=seeds,
        conditions=[dict(case=_continuous_case_label(x),speed_mps=x.config.speed_mps,
                         command_time_s=x.config.command_time_s) for x in cases],
        method_configs=configs,pattern=pattern_identity(pattern),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    manifest['fingerprint'] = _continuous_manifest_fingerprint(manifest)
    return manifest

def _continuous_trial(job):
    from dataclasses import asdict
    from experiment_contract import canonical_json, content_hash
    label, method, cfg_dict, pattern = job
    cfg_dict = dict(cfg_dict)
    cfg_dict['observation'] = PowerObservationConfig(**cfg_dict['observation'])
    cfg_dict['active_scan'] = ActiveScanConfig(**cfg_dict.get('active_scan', {}))
    cfg = SimulationConfig(**cfg_dict)
    sim = BidirectionalMechanicalSimulation(cfg,pattern,record_timeseries=(cfg.observation.seed==81000))
    result = sim.run_to_completion()
    effective = asdict(sim.effective_config)
    row = dict(result.summary,case=label,method=method,seed=cfg.observation.seed,
               speed=cfg.speed_mps,command=cfg.command_time_s,recorded_duration_s=cfg.duration_s,
               effective_config_json=canonical_json(effective),trial_config_sha256=content_hash(effective))
    return row, result.timeseries, {k:np.asarray(v) for k,v in sim.full.items()}

def _run_continuous_prediction_experiment(pattern, cases, output_dir, duration_s=None, workers=1):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiment_contract import expected_config, validate_trials
    import json
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest = _build_continuous_manifest(pattern,cases,duration_s)
    mp=output_dir/'experiment_manifest.json';tp=output_dir/'trials.csv'
    if mp.exists():
        previous=json.loads(mp.read_text(encoding='utf-8'))
        if previous.get('fingerprint') != _continuous_manifest_fingerprint(previous):
            raise RuntimeError('Existing manifest is internally inconsistent')
        if previous['fingerprint'] != manifest['fingerprint']:
            raise RuntimeError('Effective configuration, actual pattern or source changed; use a new run directory')
    elif tp.exists():
        raise RuntimeError('Existing trial data have no verifiable manifest')
    done=set()
    if tp.exists():
        previous_rows=pd.read_csv(tp)
        validate_trials(manifest,previous_rows,allow_missing=True)
        done=set(zip(previous_rows['case'],previous_rows['method'],previous_rows['seed']))
    if not mp.exists():mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    jobs=[]
    for case in manifest['conditions']:
        for method in manifest['methods']:
            for seed in manifest['seeds']:
                key=(case['case'],method,seed)
                if key not in done:jobs.append((*key[:2],expected_config(manifest,*key),pattern))
    trace_dir=output_dir/'full_traces';trace_dir.mkdir(exist_ok=True)
    def save(result):
        row,trace,full=result
        validate_trials(manifest,pd.DataFrame([row]),allow_missing=True)
        stem=f"{row['case']}_{row['method']}_{row['seed']}"
        np.savez_compressed(trace_dir/(stem+'.npz'),**full)
        if trace:pd.DataFrame(trace).to_csv(output_dir/(stem+'.csv'),index=False)
        pd.DataFrame([row]).to_csv(tp,index=False,mode='a',header=not tp.exists())
    if workers==1:
        for job in jobs:save(_continuous_trial(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_continuous_trial,job) for job in jobs]
            for i,f in enumerate(as_completed(futures),1):
                save(f.result())
                if i%30==0 or i==len(jobs):print(f'{i}/{len(jobs)} trials completed',flush=True)
    if tp.exists():
        rows=pd.read_csv(tp)
        validate_trials(manifest,rows,allow_missing=False)
        rows.sort_values(['case','method','seed']).to_csv(tp,index=False)


def _continuous_case_label(case: ExperimentCase) -> str:
    """Archived-condition label, e.g. ``v2_tc0.1``."""
    speed = float(case.config.speed_mps)
    command = float(case.config.command_time_s)
    prefix = '' if case.experiment == 'continuous_prediction' else case.experiment + '_'
    return f'{prefix}v{speed:g}_tc{command:g}'


def main(argv: list[str] | None = None) -> int:
    """Run selected, reproducible paper data experiments without a GUI dependency."""
    parser = argparse.ArgumentParser(description='220 GHz TWC mechanical-tracking experiments')
    parser.add_argument('--experiment', default='representative',
                        help='representative, mobility, noise, baseline, a registry key, or all')
    parser.add_argument('--runs', type=int, default=500, help='Monte Carlo trials for non-representative cases')
    parser.add_argument('--seed', type=int, default=20260903, help='base random seed')
    parser.add_argument('--duration-s', type=float, default=None, help='optional common duration override for smoke runs')
    parser.add_argument('--output', type=Path, default=WORKSPACE / DEFAULT_TWC_RESULTS_DIRECTORY)
    args = parser.parse_args(argv)
    pattern = load_measured_pattern()
    mc_config = MonteCarloConfig(runs=args.runs, base_seed=args.seed)
    selected = list(experiment_registry(pattern).keys()) if args.experiment == 'all' else [args.experiment]
    for experiment in selected:
        run_paper_experiment(pattern, experiment, args.output, mc_config, args.duration_s)
    print(f"Saved {', '.join(selected)} CSV results to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
