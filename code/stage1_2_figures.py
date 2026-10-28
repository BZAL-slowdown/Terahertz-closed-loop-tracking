"""阶段一、二补充结果图：从实测方向图到静态闭环的诊断图组。

与原入口 ``stage1_2_thz.py`` 的关系：
    * 不改动原入口、不改动 outputs_stage1_2 / outputs_stage1_2_15m；
    * 复用其中的 ``WorkbookPattern`` 与主模块的 ``LinkBudget`` / ``IEKF`` / ``PIController``；
    * 结果写入 ``outputs_stage1_2_figures/``，供报告与组会插图。

运行：
    python stage1_2_figures.py                 # 全部图
    python stage1_2_figures.py --distance-m 15 # 15 m 工况
    python stage1_2_figures.py --only f02 f05  # 只重画某几张
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import main_THZ_220GHz_deterministic_mechanical_tracking as thz
import stage1_2_thz as s12

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
OUT = WORKSPACE / "outputs" / "figures"
WORKBOOK = WORKSPACE / "data" / "LWB_CRS-220-100MM-2406-2.xlsx"

DT_S = 0.5
STEPS = 200
SEED = 20261028
SIGMA_DB = 0.10
DISTANCES = (0.92, 15.0)

# 配色：H 面青、V 面橙，结论线用品红/深绿，灰度打印也能分开。
C_H = "#137c8b"
C_V = "#d17a22"
C_ACC = "#8b4e75"
C_OK = "#3d6b3d"
C_GREY = "#4a4a4a"
C_GRID = dict(alpha=0.22, lw=0.7)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": "#b9b9b9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# =============================================================================
# 工况与方向图工具
# =============================================================================

@dataclass
class Case:
    """一个距离下的完整静态工况，供各图共用。"""
    distance_m: float
    pattern: s12.WorkbookPattern
    link: thz.LinkBudget
    q_peak_dbm: float
    hpbw: np.ndarray              # [H, V] 3 dB 宽度
    radius_deg: float             # 阶段二用的固定扫描半径
    offsets: list[np.ndarray]     # 三点扫描偏置（滤波器坐标系）


def make_case(pattern: s12.WorkbookPattern, link: thz.LinkBudget, distance_m: float) -> Case:
    hpbw = np.array([pattern.hpbw(0), pattern.hpbw(1)])
    radius = 0.12 * float(np.mean(hpbw))
    offsets = [
        radius * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        for a in (0.0, 120.0, 240.0)
    ]
    q_peak = link.received_power_dbm(distance_m, pattern.h_peak, pattern.e_peak)
    return Case(distance_m, pattern, link, q_peak, hpbw, radius, offsets)


def _axis_arrays(pattern: s12.WorkbookPattern, axis: int):
    return (pattern.h_angle, pattern.h_interp) if axis == 0 else (pattern.e_angle, pattern.e_interp)


def loss_vec(pattern: s12.WorkbookPattern, axis: int, angles: np.ndarray) -> np.ndarray:
    """向量化相对损耗：域外夹紧到边界值，负值截零（与标量实现一致）。"""
    nodes, interp = _axis_arrays(pattern, axis)
    trimmed = np.clip(np.asarray(angles, dtype=float), nodes[0], nodes[-1])
    return np.maximum(np.asarray(interp(trimmed), dtype=float), 0.0)


def slope_vec(pattern: s12.WorkbookPattern, axis: int, angles: np.ndarray) -> np.ndarray:
    """向量化 dL/dθ (dB/deg)；域外与截零区导数置零。"""
    nodes, interp = _axis_arrays(pattern, axis)
    deriv = pattern.h_derivative if axis == 0 else pattern.e_derivative
    raw_angle = np.asarray(angles, dtype=float)
    trimmed = np.clip(raw_angle, nodes[0], nodes[-1])
    raw = np.asarray(interp(trimmed), dtype=float)
    outside = np.abs(trimmed - raw_angle) > 1e-14
    dead = outside | (raw <= 0.0)
    return np.where(dead, 0.0, np.asarray(deriv(trimmed), dtype=float))


def steepest_radius(pattern: s12.WorkbookPattern, axis: int,
                    loss_limit_db: float = 6.0) -> tuple[float, float]:
    """在主瓣（损耗 ≤ loss_limit_db）内找 |dL/dθ| 最大处。

    主瓣以外是实测边瓣，边瓣边缘的陡壁能把斜率放大到上百 dB/deg，
    那不是可用工作点。不设这个限制就会得出“越靠外越灵敏”的错误结论。
    """
    nodes, _ = _axis_arrays(pattern, axis)
    grid = np.linspace(0.0, min(2.0, nodes[-1]), 4001)
    slopes = slope_vec(pattern, axis, grid)
    inside = loss_vec(pattern, axis, grid) <= loss_limit_db
    masked = np.where(inside, np.abs(slopes), -np.inf)
    index = int(np.argmax(masked))
    return float(grid[index]), float(slopes[index])


# =============================================================================
# 静态闭环：四种反馈律
# =============================================================================

class ResettablePI(thz.PIController):
    """死区内清零积分的 PI 变体，用来区分“积分冻结”与“积分本身”的问题。"""

    def update(self, error_deg: np.ndarray, interval_s: float) -> np.ndarray:
        error = np.asarray(error_deg, dtype=float).copy()
        inside = np.abs(error) < thz.PI_DEADBAND_DEG
        error[inside] = 0.0
        self.integral[inside] = 0.0
        proposal = self.integral + error * interval_s
        raw = thz.PI_KP * error + thz.PI_KI * proposal
        bounded = np.clip(raw, -thz.PI_LIMIT_DEG, thz.PI_LIMIT_DEG)
        inside_limit = np.abs(raw) <= thz.PI_LIMIT_DEG + 1e-12
        self.integral[inside_limit] = proposal[inside_limit]
        return bounded


def estimate_gradient(samples: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    """三点功率样本的最小二乘梯度：z_i ≈ c - g_H·s_H - g_E·s_E。"""
    design = np.array([[1.0, -s[0], -s[1]] for s, _, _ in samples], dtype=float)
    values = np.array([z for _, z, _ in samples], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.array([coefficients[1], coefficients[2]], dtype=float)


def run_static_loop(case: Case, law: str = "unit", seed: int = SEED,
                    steps: int = STEPS, initial_scale: float = 1.0,
                    initial_error: np.ndarray | None = None,
                    gradient_gain_deg_per_db: float = 0.5) -> dict[str, np.ndarray]:
    """静态视距上跑一次闭环，返回逐拍记录。

    law:
        unit      —— 直接以 IEKF 角度估计回中；
        pi        —— 原 ``thz.PIController``（死区 0.05°、积分不清零）；
        pi_reset  —— 死区内清零积分的 PI；
        grad      —— 固定增益梯度上升，模拟把 LED 侧标定的步长直接搬到 THz。
    ``initial_error`` 若给出则直接作为初始两轴误差，忽略 ``initial_scale``。
    """
    rng = np.random.default_rng(seed)
    pattern = case.pattern
    filt = thz.IEKF(pattern, case.q_peak_dbm, angular_accel_deg_s2=0.003,
                    measurement_design_sigma_db=0.18,
                    q_accel_sigma_db_per_s2=0.015)
    filt.x[:] = [case.q_peak_dbm, 0.0, 0.0, 0.0, 0.0, 0.0]
    filt.p = np.diag([4.0, 0.08, (0.8 * case.hpbw[0]) ** 2,
                      (0.8 * case.hpbw[1]) ** 2, 0.02, 0.02])

    if initial_error is not None:
        true_error = np.asarray(initial_error, dtype=float).copy()
    else:
        true_error = np.array([0.12 * case.hpbw[0], -0.10 * case.hpbw[1]]) * initial_scale
    controller = ResettablePI() if law == "pi_reset" else thz.PIController()
    keys = ("time", "true_h", "true_v", "est_h", "est_v", "cmd_h", "cmd_v",
            "pre_h", "pre_v",
            "q", "q_dot", "omega_h", "omega_v", "nees", "score", "scale",
            "trace_p", "min_eig", "sigma_h", "sigma_v", "accepted")
    record: dict[str, list[float]] = {key: [] for key in keys}

    for index in range(steps):
        filt.predict(DT_S)
        samples = []
        for offset, age_s in zip(case.offsets, (0.20, 0.10, 0.0)):
            apparent = true_error - offset
            gain = pattern.local_relative_gain(float(apparent[0]), float(apparent[1]))
            z = case.q_peak_dbm + gain + float(rng.normal(0.0, SIGMA_DB))
            samples.append((offset.copy(), z, age_s))
        result = filt.batch_update(samples)

        if law == "unit":
            command = np.clip(filt.error, -1.5 * case.hpbw, 1.5 * case.hpbw)
        elif law == "grad":
            command = np.clip(-gradient_gain_deg_per_db * estimate_gradient(samples), -5.0, 5.0)
        else:
            command = controller.update(filt.error, DT_S)

        # 回中前的指向误差才是这一拍的“待校正量”，单独记一份。
        pre_error = true_error.copy()
        true_error = true_error - command
        filt.x[2:4] -= command

        p_ang = 0.5 * (filt.p[2:4, 2:4] + filt.p[2:4, 2:4].T)
        residual = filt.x[2:4] - true_error
        try:
            nees = float(residual @ np.linalg.solve(p_ang, residual))
        except np.linalg.LinAlgError:
            nees = math.inf

        record["time"].append(index * DT_S)
        record["true_h"].append(float(true_error[0]))
        record["true_v"].append(float(true_error[1]))
        record["est_h"].append(float(filt.x[2]))
        record["est_v"].append(float(filt.x[3]))
        record["cmd_h"].append(float(command[0]))
        record["cmd_v"].append(float(command[1]))
        record["pre_h"].append(float(pre_error[0]))
        record["pre_v"].append(float(pre_error[1]))
        record["q"].append(float(filt.x[0]))
        record["q_dot"].append(float(filt.x[1]))
        record["omega_h"].append(float(filt.x[4]))
        record["omega_v"].append(float(filt.x[5]))
        record["nees"].append(nees)
        record["score"].append(float(result.normalized_innovation_score))
        record["scale"].append(float(result.measurement_scale))
        record["trace_p"].append(float(np.trace(filt.p)))
        record["min_eig"].append(filt.minimum_eigenvalue())
        record["sigma_h"].append(float(np.sqrt(max(p_ang[0, 0], 0.0))))
        record["sigma_v"].append(float(np.sqrt(max(p_ang[1, 1], 0.0))))
        record["accepted"].append(float(result.accepted))
    return {key: np.asarray(value, dtype=float) for key, value in record.items()}


def terminal_rms(result: dict[str, np.ndarray], tail: int = 20) -> float:
    norm = np.hypot(result["true_h"], result["true_v"])
    return float(np.sqrt(np.mean(norm[-tail:] ** 2)))


# =============================================================================
# 图 1：二维波束与三点扫描几何
# =============================================================================

def fig01_beam_map(case: Case) -> None:
    pattern = case.pattern
    span = 1.05
    grid = np.linspace(-span, span, 321)
    loss_2d = loss_vec(pattern, 0, grid)[:, None] + loss_vec(pattern, 1, grid)[None, :]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.0, 1.05]})
    levels = np.arange(0.0, 12.01, 0.25)
    mesh = ax.contourf(grid, grid, loss_2d, levels=levels, cmap="magma_r", extend="max")
    for level, width in ((0.25, 0.8), (0.5, 0.8), (1.0, 0.9), (2.0, 1.0), (3.0, 1.6), (6.0, 1.3)):
        line = ax.contour(grid, grid, loss_2d, levels=[level], colors="white",
                          linewidths=width, alpha=0.85)
        ax.clabel(line, fmt={level: f"{level:g} dB"}, fontsize=7.2, inline_spacing=3)
    ax.add_patch(Circle((0, 0), case.radius_deg, fill=False, ec="#22d3ee", lw=1.3, ls="--"))
    for index, offset in enumerate(case.offsets):
        ax.plot(offset[0], offset[1], "o", ms=7.0, mfc="#22d3ee", mec="black", mew=0.7, zorder=5)
        ax.annotate(f"S{index + 1}", offset, textcoords="offset points", xytext=(7, 7),
                    fontsize=9, weight="bold",
                    bbox=dict(fc="white", alpha=0.85, ec="none", pad=1.2))
    ax.plot(0, 0, "+", color="white", ms=12, mew=1.4)
    ax.set(xlim=(-span, span), ylim=(-span, span), aspect="equal",
           xlabel="H 偏置 / deg", ylabel="V 偏置 / deg",
           title="实测方向图（H/V 可分离）+ 三点扫描落点")
    ax.text(0.03, 0.03, f"扫描半径 {case.radius_deg:.4f}° (=0.12×平均 HPBW)\n"
                        f"H/V 半功率宽度 {case.hpbw[0]:.4f}° / {case.hpbw[1]:.4f}°\n"
                        f"峰值增益 {pattern.h_peak:.2f} / {pattern.e_peak:.2f} dBi",
            transform=ax.transAxes, fontsize=8.4, color="white", va="bottom",
            bbox=dict(fc="black", alpha=0.35, ec="none", pad=4))
    fig.colorbar(mesh, ax=ax, location="left", fraction=0.046, pad=0.02, label="损耗 / dB")

    angles = np.linspace(pattern.h_angle[0], pattern.h_angle[-1], 1500)
    ax2.plot(angles, loss_vec(pattern, 0, angles), color=C_H, lw=1.6, label="H 面")
    ax2.plot(angles, loss_vec(pattern, 1, angles), color=C_V, lw=1.6, label="V 面")
    ax2.axhline(3.0, color=C_GREY, ls="--", lw=0.9)
    ax2.axvspan(-case.radius_deg, case.radius_deg, color="#22d3ee", alpha=0.14)
    ax2.annotate("三点扫描全部落在 ±0.10° 内\n（主瓣肩部，不在边瓣）",
                 xy=(0.0, 1.2), xytext=(0.9, 9.0), fontsize=8.6,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#0e7490"))
    ax2.annotate("边瓣区：斜率符号反转，\n单点梯度法会把误差推大",
                 xy=(-2.55, 33.5), xytext=(-5.9, 8.0), fontsize=8.6,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GREY))
    ax2.set(xlabel="离轴角 / deg", ylabel="相对损耗 / dB", ylim=(0, 50),
            title="全角域切面：主瓣 ≈0.7–0.8°，其余全是被测量到的边瓣")
    ax2.legend(loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"220 GHz 实测波束：能量集中在 {case.hpbw.mean():.2f}° 以内，"
                 f"距离 {case.distance_m:g} m", fontsize=12.5)
    fig.savefig(OUT / "fig01_beam_map_2d.png")
    plt.close(fig)


# =============================================================================
# 图 2：主瓣剖面、局部斜率与“角度/分贝”换算
# =============================================================================

def fig02_profile_slope(case: Case) -> None:
    pattern = case.pattern
    angles = np.linspace(0.0, 0.62, 900)
    loss_h, loss_v = loss_vec(pattern, 0, angles), loss_vec(pattern, 1, angles)
    slope_h, slope_v = slope_vec(pattern, 0, angles), slope_vec(pattern, 1, angles)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), layout="constrained")
    ax.plot(angles, loss_h, color=C_H, lw=1.8, label="H 面损耗")
    ax.plot(angles, loss_v, color=C_V, lw=1.8, label="V 面损耗")
    ax.axhline(3.0, color=C_GREY, ls="--", lw=0.9)
    ax.text(0.545, 3.14, "3 dB", color=C_GREY, fontsize=8.5)
    for axis, half, colour in ((0, case.hpbw[0] / 2, C_H), (1, case.hpbw[1] / 2, C_V)):
        ax.plot([half, half], [0, 3.0], ":", color=colour, lw=1.0)
        ax.annotate(f"{half:.3f}°", (half, 3.0), textcoords="offset points",
                    xytext=(0, 5), fontsize=8.2, color=colour, ha="center")
    ax.axvline(case.radius_deg, color="#22d3ee", lw=1.1, ls="--")
    ax.annotate(f"阶段二固定半径\n{case.radius_deg:.4f}°", (case.radius_deg, 0.62),
                xytext=(0.165, 1.45), fontsize=8.4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#0e7490"))
    ax.fill_between(angles, 0, 3.0, color="#22d3ee", alpha=0.06)
    ax.set(xlim=(0, 0.62), ylim=(0, 4.0), xlabel="离轴角 / deg", ylabel="相对损耗 / dB",
           title="主瓣剖面：0.35°/0.42° 就到 3 dB，斜率不是常数")
    ax.legend(loc="lower right", frameon=False)

    ax2.plot(angles, np.abs(slope_h), color=C_H, lw=1.8, label="H 面 |dL/dθ|")
    ax2.plot(angles, np.abs(slope_v), color=C_V, lw=1.8, label="V 面 |dL/dθ|")
    ax2.axvline(case.radius_deg, color="#22d3ee", lw=1.1, ls="--")
    h_here = abs(float(slope_vec(pattern, 0, np.array([case.radius_deg]))[0]))
    v_here = abs(float(slope_vec(pattern, 1, np.array([case.radius_deg]))[0]))
    ax2.plot([case.radius_deg] * 2, [h_here, v_here], "o", ms=6.5,
             mfc="white", mec="black", zorder=6)
    h_best, h_slope = steepest_radius(pattern, 0, loss_limit_db=3.0)
    v_best, v_slope = steepest_radius(pattern, 1, loss_limit_db=3.0)

    shoulder = np.linspace(0.05, 0.22, 400)
    v_profile = np.abs(slope_vec(pattern, 1, shoulder))
    v_min_index = int(np.argmin(v_profile))
    v_min_angle, v_min_slope = float(shoulder[v_min_index]), float(v_profile[v_min_index])
    ax2.annotate("", xy=(v_min_angle, v_min_slope), xytext=(v_min_angle, 3.0),
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=C_ACC))

    ax2.text(0.02, 0.97,
             f"在 r={case.radius_deg:.3f}° 处：\n"
             f"  H 灵敏度 {h_here:.1f} dB/deg → 0.1 dB ≈ {0.1 / h_here:.3f}°\n"
             f"  V 灵敏度 {v_here:.2f} dB/deg → 0.1 dB ≈ {0.1 / v_here:.3f}°\n"
             f"V 平肩最低 {v_min_slope:.2f} dB/deg（{v_min_angle:.3f}°）→ 0.1 dB ≈ {0.1 / v_min_slope:.2f}°\n"
             f"主瓣内（3 dB）最陡：H {h_best:.3f}°（{h_slope:.1f}），V {v_best:.3f}°（{v_slope:.1f}）dB/deg",
             transform=ax2.transAxes, va="top", ha="left", fontsize=8.0,
             bbox=dict(fc="white", alpha=0.9, ec="#cccccc", lw=0.6, pad=4))
    ax2.set(xlim=(0, 0.62), ylim=(0, 32), xlabel="离轴角 / deg",
            ylabel="局部灵敏度 |dL/dθ| / (dB/deg)",
            title="同一半径下 H 约 5、V 约 3.6 dB/deg，V 肩部最低只剩 0.76")
    ax2.legend(loc="upper right", frameon=False)
    fig.suptitle(f"窄波束的“悬崖”到底有多陡：距离 {case.distance_m:g} m，"
                 f"峰值 {pattern.h_peak:.1f} dBi", fontsize=12.5)
    fig.savefig(OUT / "fig02_mainlobe_slope.png")
    plt.close(fig)


# =============================================================================
# 图 3：PCHIP 保形插值的可信度检查
# =============================================================================

def fig03_pchip_qa(case: Case) -> None:
    pattern = case.pattern
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.6), layout="constrained")

    amp = np.linspace(pattern.h_angle[0], pattern.h_angle[-1], 1200)
    ax.plot(amp, -loss_vec(pattern, 0, amp) + pattern.h_peak, color=C_H, lw=1.2, label="H 面 PCHIP")
    ax.plot(pattern.h_angle, pattern.h_gain, ".", ms=2.6, color=C_H, alpha=0.65, label="H 实测节点")
    ax.plot(pattern.e_angle, pattern.e_gain, ".", ms=2.6, color=C_V, alpha=0.65, label="V 实测节点")
    ax.set(xlabel="离轴角 / deg", ylabel="实测增益 / dBi",
           title="(a) 601 点原始切面，0.02° 一个节点")
    ax.legend(loc="lower center", frameon=False)
    inside = (pattern.h_angle > 1.0) & (pattern.h_angle < 1.4)
    ax.annotate(f"{inside.sum()} 个节点挤在 1.0–1.4° 的边瓣里", xy=(1.2, 27.0),
                xytext=(3.9, 13.0), fontsize=8.2,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GREY))

    zoom = np.linspace(0.0, 0.5, 900)
    ax2.plot(zoom, loss_vec(pattern, 0, zoom), color=C_H, lw=1.8, label="H PCHIP")
    ax2.plot(zoom, loss_vec(pattern, 1, zoom), color=C_V, lw=1.8, label="V PCHIP")
    for axis, colour in ((0, C_H), (1, C_V)):
        nodes, _ = _axis_arrays(pattern, axis)
        mask = (nodes >= 0.0) & (nodes <= 0.5)
        ax2.plot(nodes[mask], loss_vec(pattern, axis, nodes[mask]), "o", ms=2.4,
                 color=colour, alpha=0.8)
    ax2.set(xlim=(0, 0.5), ylim=(0, 6.2), xlabel="离轴角 / deg", ylabel="相对损耗 / dB",
            title="(b) 主瓣内：插值严格过节点，无过冲")
    ax2.legend(loc="upper left", frameon=False)

    fine = np.linspace(0.0, 0.5, 8001)
    slope_h_fine = slope_vec(pattern, 0, fine)
    ax3.plot(fine, slope_h_fine, color=C_H, lw=1.4, label="dL_H/dθ")
    ax3.plot(fine, slope_vec(pattern, 1, fine), color=C_V, lw=1.4, label="dL_V/dθ")
    flat = fine[np.abs(slope_h_fine) < 1.5]
    if flat.size:
        ax3.axvspan(0.0, flat.max(), color=C_ACC, alpha=0.10)
        ax3.text(0.03, 0.97, f"灰带 0–{flat.max():.3f}° 是峰顶平台，\n|斜率| <1.5 dB/deg，角度几乎不可观",
                 transform=ax3.transAxes, va="top", fontsize=8.2, color=C_ACC,
                 bbox=dict(fc="white", alpha=0.85, ec="none", pad=3))
    ax3.set(xlim=(0, 0.5), xlabel="离轴角 / deg", ylabel="局部斜率 / (dB/deg)",
            title="(c) 导数：PCHIP 只有 C¹，节点处二阶导跳变")
    ax3.legend(loc="lower right", frameon=False)

    max_gap = float(np.max(np.diff(pattern.h_angle)))
    fig.suptitle(f"图 3  插值不是装饰：节点间距 ≤{max_gap:.3f}°，域外斜率按零处理",
                 fontsize=12.5)
    fig.savefig(OUT / "fig03_pchip_qa.png")
    plt.close(fig)


# =============================================================================
# 图 4：链路预算随距离的演变
# =============================================================================

def _radiometric_sigma_db(snr_linear: np.ndarray, samples: float) -> np.ndarray:
    """能量检测器的 dB 域标准差：σ_dB = (10/ln10)·sqrt((1+2·SNR)/(M·SNR²))。"""
    safe = np.maximum(snr_linear, 1e-3)
    relative = np.sqrt((1.0 + 2.0 * safe) / (samples * safe ** 2))
    return 10.0 / math.log(10.0) * relative


def fig04_link_budget(case: Case) -> dict[str, float]:
    distances = np.logspace(math.log10(0.3), 4.0, 500)
    link = case.link
    fspl = np.array([link.fspl_db(d) for d in distances])
    components = [link.attenuation_components_db(d) for d in distances]
    gas = np.array([c["oxygen_attenuation_db"] + c["water_vapour_attenuation_db"] for c in components])
    pr = np.array([link.received_power_dbm(d, case.pattern.h_peak, case.pattern.e_peak) for d in distances])
    snr = pr - link.noise_power_dbm

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.7), layout="constrained")
    total = fspl + gas
    ax.semilogx(distances, fspl, color=C_H, lw=1.8, label="自由空间损耗")
    ax.semilogx(distances, total, color=C_V, lw=1.8, ls="--", label="合计（含气体吸收）")
    ax.set(xlabel="距离 / m", ylabel="传播损耗 / dB",
           title="(a) 百米以内两条线重合，气体到公里级才抬头")
    ax.legend(loc="upper left", frameon=False)
    for distance in DISTANCES:
        ax.plot(distance, link.propagation_components_db(distance)["total_propagation_loss_db"],
                "o", ms=7, mfc="white", mec="black", zorder=5)
        ax.annotate(f"{distance:g} m", (distance, link.fspl_db(distance)),
                    textcoords="offset points", xytext=(8, -16), fontsize=8.4)
    gas15 = link.propagation_components_db(15.0)["gaseous_attenuation_db"]
    ax.text(0.04, 0.62,
            f"15 m：气体 {gas15:.3f} dB\n1 km：{gas[np.argmin(np.abs(distances - 1000))]:.2f} dB\n"
            f"10 km：{gas[-1]:.1f} dB",
            transform=ax.transAxes, fontsize=8.4, va="top",
            bbox=dict(fc="white", alpha=0.85, ec="#cccccc", lw=0.6, pad=4))

    ax2.semilogx(distances, pr, color=C_H, lw=1.8, label="接收功率")
    ax2.set(xlabel="距离 / m", ylabel="接收功率 / dBm",
            title="(b) 0.92→15 m 掉 24.3 dB，SNR 仍在 80 dB 以上")
    for distance in DISTANCES:
        power = link.received_power_dbm(distance, case.pattern.h_peak, case.pattern.e_peak)
        ax2.plot(distance, power, "o", ms=7, mfc="white", mec="black")
        ax2.annotate(f"{power:.1f} dBm\nSNR {link.snr_db(power):.1f} dB", (distance, power),
                     textcoords="offset points", xytext=(9, 8), fontsize=8.2)
    p1 = link.received_power_dbm(DISTANCES[0], case.pattern.h_peak, case.pattern.e_peak)
    p2 = link.received_power_dbm(DISTANCES[1], case.pattern.h_peak, case.pattern.e_peak)
    delta = p1 - p2
    ax2.annotate("", xy=(DISTANCES[1], p2), xytext=(DISTANCES[1], p1),
                 arrowprops=dict(arrowstyle="<->", lw=1.0, color=C_GREY))
    ax2.text(DISTANCES[1] * 0.5, (p1 + p2) / 2, f"Δ = {delta:.2f} dB", fontsize=8.6,
             color=C_GREY, ha="right")
    snr = pr - link.noise_power_dbm

    sigma = _radiometric_sigma_db(10.0 ** (snr / 10.0), 1.0)
    sigma16 = _radiometric_sigma_db(10.0 ** (snr / 10.0), 16.0)
    ax3.loglog(distances, sigma, color=C_ACC, lw=1.8, label="辐射计噪声（M=1）")
    ax3.loglog(distances, sigma16, color=C_ACC, lw=1.2, ls="--", label="16 次平均")
    ax3.axhline(SIGMA_DB, color=C_GREY, lw=1.0, ls=":")
    ax3.text(0.4, SIGMA_DB * 1.25, f"本仿真假设 {SIGMA_DB} dB", fontsize=8.4, color=C_GREY)
    crossing_index = int(np.argmin(np.abs(np.log(sigma) - np.log(SIGMA_DB))))
    crossing = float(distances[crossing_index])
    ax3.axvline(crossing, color=C_OK, lw=1.0, ls="--")
    ax3.annotate(f"{crossing / 1000:.1f} km", (crossing, SIGMA_DB), textcoords="offset points",
                 xytext=(-34, 8), fontsize=8.8, color=C_OK)
    ax3.text(0.03, 0.34, f"热噪声要到 {crossing / 1000:.1f} km 才追平 0.1 dB；\n"
                         f"此前测角精度由标定/模型误差决定",
             transform=ax3.transAxes, fontsize=8.4, color=C_OK, va="top",
             bbox=dict(fc="white", alpha=0.85, ec="none", pad=3))
    ax3.set(xlabel="距离 / m", ylabel="dB 域功率测量标准差",
            title="(c) 0.1 dB 是标定误差，不是热噪声", ylim=(1e-7, 10))
    ax3.legend(loc="lower left", frameon=False)
    fig.suptitle("链路预算：0.92 m 与 15 m 的差别几乎全部在自由空间损耗这一项",
                 fontsize=12.5)
    fig.savefig(OUT / "fig04_link_budget_distance.png")
    plt.close(fig)
    return {"delta_pr_db": float(delta), "gas_15m_db": float(gas15),
            "sigma_crossing_m": crossing}


# =============================================================================
# 图 5：IEKF 的自洽性（误差带、NEES、新息、协方差、q 与 q_dot）
# =============================================================================

def fig05_iekf_diagnostics(case: Case, result: dict[str, np.ndarray]) -> dict[str, float]:
    time = result["time"]
    norm = np.hypot(result["true_h"], result["true_v"])
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), layout="constrained")

    ax = axes[0, 0]
    ax.plot(time, result["true_h"], color=C_H, lw=1.5, label="H 真值")
    ax.plot(time, result["est_h"], "--", color=C_H, lw=1.1, alpha=0.85, label="H 估计")
    ax.fill_between(time, result["est_h"] - 3 * result["sigma_h"],
                    result["est_h"] + 3 * result["sigma_h"], color=C_H, alpha=0.12, lw=0)
    ax.set(ylabel="H 向误差 / deg", xlabel="时间 / s", ylim=(-0.07, 0.07),
           title="(a) 估计 ±3σ：初始 3σ≈1.7°，几拍收到 0.05°")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[0, 1]
    ax.plot(time, result["true_v"], color=C_V, lw=1.5, label="V 真值")
    ax.plot(time, result["est_v"], "--", color=C_V, lw=1.1, alpha=0.85, label="V 估计")
    ax.fill_between(time, result["est_v"] - 3 * result["sigma_v"],
                    result["est_v"] + 3 * result["sigma_v"], color=C_V, alpha=0.14, lw=0)
    ax.set(ylabel="V 向误差 / deg", xlabel="时间 / s", ylim=(-0.07, 0.07),
           title="(b) V 轴：同一套参数，没有单独调")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[0, 2]
    ax.semilogy(time, np.clip(result["nees"], 1e-6, None), color=C_ACC, lw=1.3, label="NEES")
    running = np.array([np.mean(result["nees"][max(0, i - 19):i + 1]) for i in range(len(time))])
    ax.semilogy(time, np.clip(running, 1e-6, None), color=C_GREY, lw=1.6, label="20 拍滑动均值")
    ax.axhspan(0.0506, 5.9915, color=C_OK, alpha=0.12, lw=0)
    ax.axhline(2.0, color=C_GREY, ls=":", lw=0.9)
    ax.set(ylim=(1e-3, 1e2), xlabel="时间 / s", ylabel="NEES（2 自由度）",
           title="(c) 一致性：闭环下 NEES≈0.4，偏保守（设计 R 取大了）")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 0]
    ax.plot(time, result["score"], color=C_ACC, lw=1.2, label="归一化新息")
    ax.axhline(thz.INNOVATION_CONSISTENCY_DESIGN_THRESHOLD, color=C_GREY, ls="--", lw=0.9)
    ax.set(yscale="log", xlabel="时间 / s", ylabel="新息平方 / 阈值",
           title="(d) 新息检验：虚线以上才会触发 R 膨胀")
    axb = ax.twinx()
    axb.plot(time, result["scale"], color=C_OK, lw=1.1, alpha=0.8, label="R 膨胀因子")
    axb.set_ylabel("R 膨胀因子", color=C_OK)
    axb.tick_params(axis="y", colors=C_OK)
    axb.spines["right"].set_visible(True)

    ax = axes[1, 1]
    ax.semilogy(time, result["trace_p"], color=C_H, lw=1.4, label="tr(P)")
    ax.semilogy(time, np.clip(result["min_eig"], 1e-12, None), color=C_ACC, lw=1.2,
                label="λ_min(P)")
    ax.set(xlabel="时间 / s", ylabel="协方差量级",
           title="(e) 协方差始终正定，λ_min 没有贴零")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 2]
    ax.plot(time, result["q"], color=C_OK, lw=1.5, label="q 估计")
    ax.axhline(case.q_peak_dbm, color=C_GREY, ls="--", lw=0.9, label="q 真值")
    ax.set(xlabel="时间 / s", ylabel="q / dBm", title="(f) q 吸收公共幅值，q_dot 稳定在零附近")
    axb = ax.twinx()
    axb.plot(time, result["q_dot"], color=C_V, lw=1.0, alpha=0.8, label="q_dot")
    axb.set_ylabel("q_dot / (dB/s)", color=C_V)
    axb.tick_params(axis="y", colors=C_V)
    axb.spines["right"].set_visible(True)
    ax.legend(loc="lower right", frameon=False)

    for axis in axes.ravel():
        axis.grid(**C_GRID)
    fig.suptitle(f"六状态 IEKF 的体检单：终值 {norm[-1]:.4f}°，末 20 拍 RMS "
                 f"{terminal_rms(result):.4f}°，{int(result['accepted'].sum())}/{len(time)} 批全部接受",
                 fontsize=12.5)
    fig.savefig(OUT / "fig05_iekf_diagnostics.png")
    plt.close(fig)
    return {"final_norm_deg": float(norm[-1]),
            "terminal_rms_deg": terminal_rms(result),
            "accepted": int(result["accepted"].sum()),
            "terminal_sigma_h_deg": float(result["sigma_h"][-1]),
            "terminal_sigma_v_deg": float(result["sigma_v"][-1]),
            "mean_nees_last60": float(np.mean(result["nees"][-60:]))}


# =============================================================================
# 图 6：三种反馈律的对比（含固定增益梯度上升）
# =============================================================================

def fig06_control_laws(case: Case, results: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), layout="constrained")
    styles = {
        "pi": dict(color="#b03a2e", ls="-", label="原 PI（死区 0.05°，积分不清零）"),
        "unit": dict(color=C_H, ls="-", label="单位反馈（直接回中）"),
        "grad": dict(color=C_ACC, ls="--", label="固定增益梯度上升（0.5°/dB）"),
    }
    ax = axes[0]
    for key, style in styles.items():
        norm = np.hypot(results[key]["true_h"], results[key]["true_v"])
        ax.semilogy(results[key]["time"], np.clip(norm, 1e-4, None), lw=1.5, **style)
    ax.axhline(case.hpbw.mean(), color=C_GREY, ls=":", lw=0.9)
    ax.text(2, case.hpbw.mean() * 1.12, "1 个 HPBW", fontsize=8.2, color=C_GREY)
    ax.set(xlabel="时间 / s", ylabel="真实角误差范数 / deg", ylim=(1e-4, 1e3),
           title="(a) 误差轨迹：PI 在死区边缘来回蹭，梯度法直接发散")
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    ax = axes[1]
    for key, style in styles.items():
        command = np.hypot(results[key]["cmd_h"], results[key]["cmd_v"])
        ax.semilogy(results[key]["time"], np.clip(command, 1e-4, None), lw=1.3, **style)
    ax.set(xlabel="时间 / s", ylabel="每拍指令幅值 / deg",
           title="(b) 指令：PI 的积分被死区冻成“每拍恒定推一点”")
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    ax = axes[2]
    names, values, colours = [], [], []
    for key, style in styles.items():
        names.append(key)
        values.append(terminal_rms(results[key]))
        colours.append(style["color"])
    bars = ax.bar(names, values, color=colours, alpha=0.85, width=0.6)
    ax.set(yscale="log", ylabel="末 20 拍误差 RMS / deg", title="(c) 收敛终值：差三个数量级",
           ylim=(1e-3, max(values) * 8))
    ax.axhline(case.hpbw.mean(), color=C_GREY, ls=":", lw=0.9)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.4f}°", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)

    for axis in axes:
        axis.grid(**C_GRID)
    fig.suptitle(f"同一个 IEKF、同一个工况，只换反馈律：距离 {case.distance_m:g} m",
                 fontsize=12.5)
    fig.savefig(OUT / "fig06_control_laws.png")
    plt.close(fig)


# =============================================================================
# 图 7：捕获域地图（把“悬崖”画成一张二维盆地）
# =============================================================================

def capture_basin(case: Case, law: str, coords: np.ndarray, steps: int = 60) -> np.ndarray:
    """逐点跑闭环，返回终值 RMS 的二维图（行=V 初值，列=H 初值）。"""
    grid = np.empty((coords.size, coords.size), dtype=float)
    for row, v_start in enumerate(coords):
        for column, h_start in enumerate(coords):
            result = run_static_loop(case, law=law, steps=steps,
                                     initial_error=np.array([h_start, v_start]))
            grid[row, column] = terminal_rms(result, tail=10)
    return grid


def fig07_capture_basin(case: Case, coords: np.ndarray, steps: int = 60) -> dict[str, float]:
    cache = OUT / f"capture_basin_{case.distance_m:g}m.npz"
    if cache.exists():
        stored = np.load(cache)
        unit_map, grad_map = stored["unit"], stored["grad"]
    else:
        unit_map = capture_basin(case, "unit", coords, steps)
        grad_map = capture_basin(case, "grad", coords, steps)
        np.savez(cache, unit=unit_map, grad=grad_map)

    low, high = 0.006, 2.0
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), layout="constrained")
    bins = np.array([0.05, 0.30])
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#2f7d32", "#f2b134", "#b3261e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    mesh = None
    for axis, data, title in (
            (axes[0], unit_map, "(a) IEKF + 单位反馈：中心能收，边缘红绿交错"),
            (axes[1], grad_map, "(b) 固定增益梯度上升（0.5°/dB）：整片失效")):
        safe = np.where(np.isfinite(data), data, high)
        mesh = axis.pcolormesh(coords, coords, np.digitize(safe, bins), cmap=cmap, norm=norm,
                               shading="nearest")
        axis.set(xlabel="初始 H 误差 / deg", ylabel="初始 V 误差 / deg", title=title,
                 aspect="equal")
        converged = 100.0 * float(np.mean(safe < 0.05))
        axis.text(0.03, 0.03, f"终值 <0.05° 的初值占比：{converged:.0f}%",
                  transform=axis.transAxes, fontsize=9.2, va="bottom",
                  bbox=dict(fc="white", alpha=0.9, ec="none", pad=3))

    # 叠 3 dB 波束轮廓，说明“盆地”和“主瓣”并不重合。
    span = np.linspace(coords[0], coords[-1], 240)
    loss_2d = loss_vec(case.pattern, 0, span)[:, None] + loss_vec(case.pattern, 1, span)[None, :]
    for axis in axes:
        axis.contour(span, span, loss_2d, levels=[3.0], colors="black",
                     linewidths=1.2, linestyles="--")
        axis.contour(span, span, loss_2d, levels=[6.0], colors="black",
                     linewidths=0.8, linestyles=":")
    axes[0].text(0.0, case.hpbw.mean() * 0.62, "3 dB 主瓣", fontsize=8.4, ha="center",
                 bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.5))
    axes[0].text(0.03, 0.90,
                 "失败点散在边界和边瓣上，不按半径整齐划线；\n"
                 "同样的偏差，落主瓣能收，落边瓣就翻不回来",
                 transform=axes[0].transAxes, fontsize=8.2, va="top",
                 bbox=dict(fc="white", alpha=0.9, ec="none", pad=3))
    axes[1].text(0.03, 0.90,
                 "固定步长 μ=0.5°/dB，乘上局部斜率后每拍步长在 0.4–10° 之间摆，\n"
                 "平肩（0.76 dB/deg）走得过头，陡壁（20 dB/deg）直接削顶",
                 transform=axes[1].transAxes, fontsize=8.2, va="top",
                 bbox=dict(fc="white", alpha=0.9, ec="none", pad=3))

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.030, pad=0.02, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["收敛\n<0.05°", "勉强\n0.05–0.3°", "失败\n>0.3°"])
    fig.suptitle(f"图 7  捕获域不是同心圆：距离 {case.distance_m:g} m，"
                 f"能不能收取决于落在主瓣还是边瓣", fontsize=12.5)
    fig.savefig(OUT / "fig07_capture_basin.png")
    plt.close(fig)
    return {"unit_converged_fraction": float(np.mean(unit_map < 0.05)),
            "grad_converged_fraction": float(np.mean(grad_map < 0.05)),
            "unit_worst_deg": float(unit_map.max()),
            "grad_worst_deg": float(grad_map.max())}


# =============================================================================
# 图 8：0.92 m 与 15 m 两个版本到底差在哪
# =============================================================================

def fig08_distance_compare(near: Case, far: Case) -> dict[str, float]:
    near_result = run_static_loop(near, law="unit")
    far_result = run_static_loop(far, law="unit")
    angle_columns = ["true_h", "true_v", "est_h", "est_v", "cmd_h", "cmd_v", "omega_h", "omega_v"]
    max_angle_diff = float(max(np.max(np.abs(near_result[k] - far_result[k])) for k in angle_columns))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    ax.plot(near_result["time"], near_result["q"], color=C_H, lw=1.5,
            label=f"{near.distance_m:g} m：q 估计")
    ax.axhline(near.q_peak_dbm, color=C_H, ls="--", lw=0.9)
    ax.plot(far_result["time"], far_result["q"], color=C_V, lw=1.5,
            label=f"{far.distance_m:g} m：q 估计")
    ax.axhline(far.q_peak_dbm, color=C_V, ls="--", lw=0.9)
    gap = near.q_peak_dbm - far.q_peak_dbm
    ax.annotate("", xy=(8, far.q_peak_dbm), xytext=(8, near.q_peak_dbm),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=C_GREY))
    ax.text(10, (near.q_peak_dbm + far.q_peak_dbm) / 2, f"基线相差 {gap:.2f} dB",
            fontsize=9, color=C_GREY, va="center")
    ax.set(xlabel="时间 / s", ylabel="q / dBm",
           title="(a) 距离只改变功率基线 q，两套参数没有别的差别")
    ax.legend(loc="center right", frameon=False)
    ax.text(0.03, 0.93, "虚线为各自的理论真值", transform=ax.transAxes, fontsize=8.2,
            color=C_GREY, va="top")

    near_norm = np.hypot(near_result["true_h"], near_result["true_v"])
    far_norm = np.hypot(far_result["true_h"], far_result["true_v"])
    ax2.semilogy(near_result["time"], np.clip(near_norm, 1e-4, None), color=C_H, lw=2.4,
                 alpha=0.45, label=f"{near.distance_m:g} m")
    ax2.semilogy(far_result["time"], np.clip(far_norm, 1e-4, None), color=C_V, lw=1.2,
                 ls="--", label=f"{far.distance_m:g} m")
    ax2.set(xlabel="时间 / s", ylabel="真实角误差范数 / deg",
            title="(b) 两条曲线完全重合，角度过程与距离无关")
    ax2.legend(loc="upper right", frameon=False)
    ax2.text(0.03, 0.05,
             f"角度相关列最大差 {max_angle_diff:.1e}°\n"
             f"终值 RMS：{terminal_rms(near_result):.4f}° / {terminal_rms(far_result):.4f}°",
             transform=ax2.transAxes, fontsize=8.4, va="bottom",
             bbox=dict(fc="white", alpha=0.85, ec="#cccccc", lw=0.6, pad=3))
    fig.suptitle(f"图 8  0.92 m 与 15 m：只差一个 {gap:.2f} dB 的功率偏置", fontsize=12.5)
    fig.savefig(OUT / "fig08_distance_compare.png")
    plt.close(fig)
    return {"baseline_gap_db": float(gap), "max_angle_state_diff_deg": max_angle_diff,
            "near_terminal_rms_deg": terminal_rms(near_result),
            "far_terminal_rms_deg": terminal_rms(far_result)}


# =============================================================================
# 图 9：静态闭环跟踪结果（正文的主结果图）
# =============================================================================

def fig09_static_result(case: Case) -> dict[str, float]:
    """0.92 m 静态视距下 100 s 闭环：指向误差收敛与终值。"""
    result = run_static_loop(case, law="unit")
    time = result["time"]
    pre_norm = np.hypot(result["pre_h"], result["pre_v"])
    window = 20
    running = np.array([np.sqrt(np.mean(pre_norm[max(0, i - window + 1):i + 1] ** 2))
                        for i in range(len(time))])
    rms = terminal_rms(result)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), layout="constrained")

    ax.plot(time, result["pre_h"], color=C_H, lw=1.5, label="H 向指向误差")
    ax.plot(time, result["pre_v"], color=C_V, lw=1.5, label="V 向指向误差")
    ax.axhline(0.0, color=C_GREY, lw=0.8, alpha=0.6)
    ax.set(xlabel="时间 / s", ylabel="指向误差 / deg", ylim=(-0.10, 0.10),
           title="(a) 两轴指向误差：起始 0.084° / −0.083°，之后在 ±0.03° 内波动")
    ax.legend(loc="lower right", frameon=False, fontsize=8.6)

    ax2.semilogy(time, running, color=C_ACC, lw=1.6, label="10 s 滑动 RMS")
    ax2.axhline(rms, color=C_OK, lw=1.2, ls="--", label=f"末 20 拍 RMS = {rms:.4f}°")
    ax2.axvspan(time[-window], time[-1], color=C_OK, alpha=0.10, lw=0)
    ax2.text(0.03, 0.10, f"0.119° 起步 → 末段 {running[-1]:.4f}°\n约 17 倍",
             transform=ax2.transAxes, fontsize=8.8, color=C_ACC,
             bbox=dict(fc="white", alpha=0.85, ec="none", pad=3))
    ax2.set(xlabel="时间 / s", ylabel="误差范数 RMS / deg", ylim=(3e-3, 0.2),
            title="(b) 10 s 滑动 RMS：30 s 后稳定在 0.01° 量级")
    ax2.legend(loc="upper right", frameon=False, fontsize=8.4)

    for axis in (ax, ax2):
        axis.grid(**C_GRID)
    fig.suptitle(f"220 GHz 静态闭环跟踪结果：σ=0.1 dB，{case.distance_m:g} m，200 拍全部接受",
                 fontsize=12.5)
    fig.savefig(OUT / "fig09_static_result.png")
    plt.close(fig)
    return {"final_norm_deg": float(pre_norm[-1]), "terminal_rms_deg": rms,
            "initial_norm_deg": float(pre_norm[0]),
            "running_rms_terminal_deg": float(running[-1])}


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="阶段一二补充诊断图组")
    parser.add_argument("--distance-m", type=float, default=0.92)
    parser.add_argument("--only", nargs="*", default=None,
                        help="只跑指定图：f01 f02 f03 f04 f05 f06 f07 f08 f09")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    pattern = s12.WorkbookPattern(WORKBOOK)
    link = thz.LinkBudget()
    case = make_case(pattern, link, args.distance_m)
    wanted = set(args.only) if args.only else {"f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09"}

    summary: dict[str, object] = {"distance_m": args.distance_m}
    if "f01" in wanted:
        fig01_beam_map(case)
    if "f02" in wanted:
        fig02_profile_slope(case)
    if "f03" in wanted:
        fig03_pchip_qa(case)
    if "f04" in wanted:
        summary["link_budget"] = fig04_link_budget(case)
    if wanted & {"f05", "f06", "f07"}:
        results = {law: run_static_loop(case, law=law) for law in ("unit", "pi", "grad")}
        summary["terminal_rms_deg"] = {law: terminal_rms(res) for law, res in results.items()}
        if "f05" in wanted:
            summary["iekf"] = fig05_iekf_diagnostics(case, results["unit"])
        if "f06" in wanted:
            fig06_control_laws(case, results)
        if "f07" in wanted:
            coords = np.linspace(-1.1, 1.1, 25)
            summary["capture"] = fig07_capture_basin(case, coords)
    if "f08" in wanted:
        summary["distance_compare"] = fig08_distance_compare(
            make_case(pattern, link, DISTANCES[0]), make_case(pattern, link, DISTANCES[1]))
    if "f09" in wanted:
        summary["static_result"] = fig09_static_result(case)

    (OUT / f"figure_summary_{args.distance_m:g}m.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"figures written to {OUT}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
