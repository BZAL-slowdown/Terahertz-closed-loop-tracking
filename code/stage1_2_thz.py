"""阶段一、二：220 GHz 实测方向图与六状态 IEKF 静态闭环验证。

运行：python stage1_2_thz.py
输入：同目录 LWB_CRS-220-100MM-2406-2.xlsx
输出：outputs_stage1_2/ 下的方向图、逐步 CSV 和运行摘要。
"""
from __future__ import annotations

import csv
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import main_THZ_220GHz_deterministic_mechanical_tracking as thz


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
WORKBOOK = WORKSPACE / "data" / "LWB_CRS-220-100MM-2406-2.xlsx"
OUT = WORKSPACE / "outputs" / "stage1_2_0.92m"
DT_S = 0.5
STEPS = 200
DISTANCE_M = 0.92
SEED = 20261028
MEASUREMENT_NOISE_SIGMA_DB = 0.10


class WorkbookPattern:
    """将四列 H/V 增益切面转成峰值居中的保形 PCHIP 损耗模型。"""

    def __init__(self, path: Path):
        # 该工作簿表头在部分 Windows/Pandas 组合中会被错误解码，按已核对的
        # 列顺序读取比对乱码列名做猜测可靠；数据列依次为 H增益/H角/V增益/V角。
        frame = pd.read_excel(path, sheet_name=0, header=None)
        if frame.shape[1] < 4:
            raise ValueError(f"Expected four antenna measurement columns, got {frame.shape[1]}")
        numeric = frame.iloc[:, :4].apply(pd.to_numeric, errors="coerce").dropna()
        if len(numeric) < 5:
            raise ValueError("Antenna workbook contains fewer than five complete numeric rows")
        h_gain, h_angle, e_gain, e_angle = (numeric.iloc[:, i].to_numpy(float) for i in range(4))
        self.h_angle, self.h_gain, self.h_interp, self.h_derivative = self._axis(h_angle, h_gain)
        self.e_angle, self.e_gain, self.e_interp, self.e_derivative = self._axis(e_angle, e_gain)
        self.h_peak = float(np.max(h_gain))
        self.e_peak = float(np.max(e_gain))
        if abs(self.h_peak - self.e_peak) > 1.0:
            raise ValueError("H/V peak mismatch exceeds 1 dB; check workbook cut definitions")

    @staticmethod
    def _axis(angle: np.ndarray, gain: np.ndarray):
        peak_index = int(np.argmax(gain))
        centered = angle - angle[peak_index]
        loss = gain[peak_index] - gain
        order = np.argsort(centered)
        centered, loss = centered[order], loss[order]
        unique, indices = np.unique(centered, return_index=True)
        loss = loss[indices]
        return unique, loss, PchipInterpolator(unique, loss, extrapolate=False), PchipInterpolator(unique, loss, extrapolate=False).derivative()

    def axis_loss(self, axis: int, angle_deg: float) -> tuple[float, float, bool]:
        angles = self.h_angle if axis == 0 else self.e_angle
        interp = self.h_interp if axis == 0 else self.e_interp
        deriv = self.h_derivative if axis == 0 else self.e_derivative
        bounded = float(np.clip(angle_deg, angles[0], angles[-1]))
        clamped = not math.isclose(bounded, angle_deg, abs_tol=1e-14)
        raw = float(interp(bounded))
        loss = max(raw, 0.0)
        slope = 0.0 if clamped or raw <= 0.0 else float(deriv(bounded))
        return loss, slope, clamped

    def loss_and_gradient(self, h_deg: float, e_deg: float):
        h_loss, h_slope, h_clamped = self.axis_loss(0, h_deg)
        e_loss, e_slope, e_clamped = self.axis_loss(1, e_deg)
        # 返回方向图相对增益 g=-L 的梯度，单位 dB/deg。
        return h_loss + e_loss, np.array([-h_slope, -e_slope]), h_clamped or e_clamped

    def local_relative_gain(self, h_deg: float, e_deg: float) -> float:
        return -self.loss_and_gradient(h_deg, e_deg)[0]

    def local_gradient(self, h_deg: float, e_deg: float) -> np.ndarray:
        return self.loss_and_gradient(h_deg, e_deg)[1]

    def hpbw(self, axis: int) -> float:
        angles = self.h_angle if axis == 0 else self.e_angle
        interp = self.h_interp if axis == 0 else self.e_interp
        dense = np.linspace(angles[0], angles[-1], 20001)
        # HPBW 取峰值两侧满足损耗 <= 3 dB 的主瓣区间。
        below = dense[np.asarray(interp(dense)) <= 3.0]
        return float(below[-1] - below[0])


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage1(pattern: WorkbookPattern, link: thz.LinkBudget) -> dict[str, float]:
    """输出实测主瓣曲线和当前距离下的链路损耗拆分。"""
    h_grid = np.linspace(pattern.h_angle[0], pattern.h_angle[-1], 3000)
    e_grid = np.linspace(pattern.e_angle[0], pattern.e_angle[-1], 3000)
    h_loss = np.array([pattern.axis_loss(0, value)[0] for value in h_grid])
    e_loss = np.array([pattern.axis_loss(1, value)[0] for value in e_grid])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(h_grid, h_loss, color="#137c8b", lw=1.8)
    axes[0].set(title="Measured H-cut loss (PCHIP)", xlabel="H offset (deg)", ylabel="Relative loss (dB)")
    axes[1].plot(e_grid, e_loss, color="#d17a22", lw=1.8)
    axes[1].set(title="Measured V-cut loss (PCHIP)", xlabel="V offset (deg)", ylabel="Relative loss (dB)")
    for axis in axes:
        axis.axhline(3.0, color="#666666", lw=0.8, ls="--")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "stage1_measured_pattern.png", dpi=180)
    plt.close(fig)
    components = link.propagation_components_db(DISTANCE_M)
    peak_received_power_dbm = link.received_power_dbm(DISTANCE_M, pattern.h_peak, pattern.e_peak)
    return {
        "samples_per_cut": float(len(pattern.h_angle)),
        "h_hpbw_deg": pattern.hpbw(0),
        "v_hpbw_deg": pattern.hpbw(1),
        "h_peak_dbi": pattern.h_peak,
        "v_peak_dbi": pattern.e_peak,
        "distance_m": DISTANCE_M,
        "peak_received_power_dbm": peak_received_power_dbm,
        "noise_power_dbm": link.noise_power_dbm,
        "peak_snr_db": link.snr_db(peak_received_power_dbm),
        **components,
        "oxygen_specific_db_per_km": link.gaseous_specific_attenuation_db_per_km[0],
        "water_vapour_specific_db_per_km": link.gaseous_specific_attenuation_db_per_km[1],
    }


def validate_measurement_jacobian(pattern: WorkbookPattern, q_peak_dbm: float) -> float:
    """用中心差分独立核对六状态解析量测雅可比。"""
    state = np.array([q_peak_dbm + 0.4, -0.03, 0.07, -0.05, 0.012, -0.009], dtype=float)
    scan_offset = np.array([0.03, -0.02], dtype=float)
    age_s = 0.17
    _, analytic = thz.pointing_measurement(state, pattern, scan_offset, age_s)
    numeric = np.empty(6, dtype=float)
    for index in range(6):
        # 按状态量纲选取小扰动，中心差分误差为 O(h^2)。
        step = 1e-5 if index in (0, 1) else 1e-6
        plus, minus = state.copy(), state.copy()
        plus[index] += step
        minus[index] -= step
        y_plus = thz.pointing_measurement(plus, pattern, scan_offset, age_s)[0]
        y_minus = thz.pointing_measurement(minus, pattern, scan_offset, age_s)[0]
        numeric[index] = (y_plus - y_minus) / (2.0 * step)
    max_error = float(np.max(np.abs(numeric - analytic)))
    if not np.allclose(numeric, analytic, rtol=2e-5, atol=2e-6):
        raise AssertionError(f"IEKF measurement Jacobian mismatch: max abs error={max_error:.3e}")
    return max_error


def stage2(pattern: WorkbookPattern, link: thz.LinkBudget,
           feedback: str = "unit") -> list[dict[str, float]]:
    """单端、静态 LOS 的三点功率扫描与六状态 IEKF 闭环。

    feedback 取 ``unit``（默认，直接以角度估计回中）或 ``pi``（原 PI 控制器）。
    原 PI 在 0.05° 死区内会把积分冻住，形成恒定位置增量，误差呈锯齿轮廓，
    终值 0.054° 是极限环上的采样点，不是稳态；单位反馈给出 0.0070°。
    """
    rng = np.random.default_rng(SEED)
    q_peak = link.received_power_dbm(DISTANCE_M, pattern.h_peak, pattern.e_peak)
    half_width = np.array([pattern.hpbw(0), pattern.hpbw(1)])
    # 阶段二先验证局部闭环：初值取在三点扫描的可捕获域内；大偏差重捕获
    # 需由后续粗扫/搜索状态机单独评估，不能用 IEKF 局部线性化强行外推。
    initial_error = np.array([0.12 * half_width[0], -0.10 * half_width[1]])
    filt = thz.IEKF(pattern, q_peak, angular_accel_deg_s2=0.003,
                    measurement_design_sigma_db=0.18,
                    q_accel_sigma_db_per_s2=0.015)
    filt.x[:] = [q_peak, 0.0, 0.0, 0.0, 0.0, 0.0]
    filt.p = np.diag([4.0, 0.08, (0.8 * half_width[0]) ** 2,
                      (0.8 * half_width[1]) ** 2, 0.02, 0.02])
    # 0.12 HPBW 半径使三点扫描位于实测主瓣斜率区，不越过强损耗边缘。
    radius = 0.12 * float(np.mean(half_width))
    offsets = [radius * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
               for a in (0.0, 120.0, 240.0)]
    true_error = initial_error.copy()
    controller = thz.PIController()
    rows: list[dict[str, float]] = []
    covariance_minima: list[float] = []

    for index in range(STEPS):
        time_s = index * DT_S
        filt.predict(DT_S)
        # 三个功率样本按扫描先后到达，age 显式交给异步观测模型。
        sample_ages = [0.20, 0.10, 0.0]
        samples = []
        for offset, age_s in zip(offsets, sample_ages):
            apparent = true_error - offset
            gain = pattern.local_relative_gain(float(apparent[0]), float(apparent[1]))
            z = q_peak + gain + float(rng.normal(0.0, MEASUREMENT_NOISE_SIGMA_DB))
            samples.append((offset.copy(), z, age_s))
        result = filt.batch_update(samples)
        if not np.all(np.isfinite(filt.p)):
            raise FloatingPointError(f"non-finite IEKF covariance at step {index}")
        minimum_covariance_eigenvalue = filt.minimum_eigenvalue()
        if minimum_covariance_eigenvalue < -1e-9:
            raise FloatingPointError(
                f"IEKF covariance is not PSD at step {index}: {minimum_covariance_eigenvalue:.3e}"
            )
        covariance_minima.append(minimum_covariance_eigenvalue)
        if feedback == "unit":
            # 直接以角度估计回中，限幅到 1.5 倍半功率宽度。
            command = np.clip(filt.error, -1.5 * half_width, 1.5 * half_width)
        else:
            command = controller.update(filt.error, DT_S)
        # 回中修正为正向角度增量；真实 LOS-中心误差与滤波器局部
        # 角误差都必须转到新扫描中心坐标系，仿照主程序的 recenter_transform。
        pre_error = true_error.copy()
        true_error -= command
        filt.x[2:4] -= command
        rows.append({
            "time_s": time_s,
            "pre_h_error_deg": float(pre_error[0]),
            "pre_v_error_deg": float(pre_error[1]),
            "true_h_error_deg": float(true_error[0]),
            "true_v_error_deg": float(true_error[1]),
            "estimated_h_error_deg": float(filt.x[2]),
            "estimated_v_error_deg": float(filt.x[3]),
            "estimated_q_dbm": float(filt.x[0]),
            "estimated_q_dot_db_s": float(filt.x[1]),
            "estimated_omega_h_deg_s": float(filt.x[4]),
            "estimated_omega_v_deg_s": float(filt.x[5]),
            "error_norm_deg": float(np.linalg.norm(true_error)),
            "scan_radius_deg": radius,
            "command_h_deg": float(command[0]),
            "command_v_deg": float(command[1]),
            "update_accepted": float(result.accepted),
            "innovation_score": float(result.normalized_innovation_score),
            "measurement_scale": float(result.measurement_scale),
            "minimum_covariance_eigenvalue": minimum_covariance_eigenvalue,
        })
    return rows


def main() -> None:
    global DISTANCE_M, OUT
    parser = argparse.ArgumentParser(description="Run the stage 1-2 220 GHz static-link validation.")
    parser.add_argument("--distance-m", type=float, default=DISTANCE_M,
                        help="terminal separation in metres (default: 0.92)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="output directory; defaults to outputs/stage1_2_<distance>m")
    parser.add_argument("--feedback", choices=("unit", "pi"), default="unit",
                        help="closed-loop feedback law (default: unit)")
    args = parser.parse_args()
    if not math.isfinite(args.distance_m) or args.distance_m <= 0.0:
        parser.error("--distance-m must be a finite positive number")
    DISTANCE_M = args.distance_m
    OUT = args.output_dir or WORKSPACE / "outputs" / f"stage1_2_{DISTANCE_M:g}m"
    OUT.mkdir(parents=True, exist_ok=True)
    pattern = WorkbookPattern(WORKBOOK)
    link = thz.LinkBudget()
    budget = stage1(pattern, link)
    q_peak = link.received_power_dbm(DISTANCE_M, pattern.h_peak, pattern.e_peak)
    jacobian_max_abs_error = validate_measurement_jacobian(pattern, q_peak)
    rows = stage2(pattern, link, feedback=args.feedback)
    covariance_minima = [row["minimum_covariance_eigenvalue"] for row in rows]
    initial_error = np.array([0.12 * pattern.hpbw(0), -0.10 * pattern.hpbw(1)])
    _write_csv(OUT / "stage2_static_iekf_data.csv", rows)

    time_s = np.array([row["time_s"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, [row["true_h_error_deg"] for row in rows], label="H true", color="#137c8b")
    axes[0].plot(time_s, [row["estimated_h_error_deg"] for row in rows], "--", label="H IEKF")
    axes[0].plot(time_s, [row["true_v_error_deg"] for row in rows], label="V true", color="#d17a22")
    axes[0].plot(time_s, [row["estimated_v_error_deg"] for row in rows], "--", label="V IEKF")
    axes[0].set_ylabel("Pointing error (deg)")
    axes[0].legend(ncol=2)
    axes[1].plot(time_s, [row["estimated_q_dbm"] for row in rows], color="#5b7f3b")
    axes[1].set_ylabel("q estimate (dBm)")
    axes[2].plot(time_s, [row["error_norm_deg"] for row in rows], color="#8b4e75")
    axes[2].set(xlabel="Time (s)", ylabel="True error norm (deg)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "stage2_static_iekf_closed_loop.png", dpi=180)
    plt.close(fig)

    summary = {
        "stage": "1-2 THz port",
        "status": "single deterministic static-link check; not bidirectional or Monte Carlo",
        "feedback_law": args.feedback,
        "sample_period_s": DT_S,
        "steps": STEPS,
        "duration_s": DT_S * STEPS,
        "seed": SEED,
        "measurement_noise_sigma_db": MEASUREMENT_NOISE_SIGMA_DB,
        "initial_error_h_v_deg": initial_error.tolist(),
        "final_error_h_v_deg": [rows[-1]["true_h_error_deg"], rows[-1]["true_v_error_deg"]],
        "final_error_norm_deg": rows[-1]["error_norm_deg"],
        "final_20_step_error_rms_deg": float(np.sqrt(np.mean([row["error_norm_deg"] ** 2 for row in rows[-20:]]))),
        "accepted_iekf_batches": int(sum(row["update_accepted"] for row in rows)),
        "minimum_covariance_eigenvalue_over_run": float(min(covariance_minima)),
        "p676_model": "ITU-R P.676-12 exact line-by-line via itur 0.4.0",
        "measurement_jacobian_check": "central finite difference vs analytic PCHIP derivative",
        "measurement_jacobian_max_abs_error": jacobian_max_abs_error,
        "p676_atmosphere": {
            "pressure_hpa": thz.ATMOSPHERIC_PRESSURE_HPA,
            "temperature_k": thz.ATMOSPHERIC_TEMPERATURE_K,
            "water_vapour_density_g_m3": thz.WATER_VAPOUR_DENSITY_G_M3,
        },
        "stage1_link_budget": budget,
        "filter_state_order": ["q_dbm", "q_dot_db_s", "e_H_deg", "e_E_deg", "omega_H_deg_s", "omega_E_deg_s"],
    }
    (OUT / "stage1_2_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert np.isfinite(np.asarray(list(budget.values()), dtype=float)).all()
    assert all(np.isfinite(list(row.values())).all() for row in rows)
    initial_error_norm = float(np.linalg.norm(initial_error))
    final_window_rms = float(np.sqrt(np.mean([row["error_norm_deg"] ** 2 for row in rows[-20:]])))
    assert final_window_rms < initial_error_norm
    print(f"Stage 1-2 checks passed: {OUT}")
    print(f"H/V HPBW: {budget['h_hpbw_deg']:.4f}/{budget['v_hpbw_deg']:.4f} deg")
    print(f"P.676 O2/H2O: {budget['oxygen_specific_db_per_km']:.6f}/"
          f"{budget['water_vapour_specific_db_per_km']:.6f} dB/km")
    print(f"Measurement Jacobian max abs error: {jacobian_max_abs_error:.3e}")
    print(f"Minimum covariance eigenvalue: {min(covariance_minima):.3e}")
    print(f"Static example final angular error: {rows[-1]['error_norm_deg']:.6f} deg "
          f"({args.feedback} feedback)")


if __name__ == "__main__":
    main()
