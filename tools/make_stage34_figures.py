"""阶段三四结果图：双向半径、距离-误差对标、正交跟踪、速度-误差。

图 1 与图 3 来自本仓库的 THz 运行结果；图 2 与图 4 重绘作者归档的 MAT 数据，
用来说明 LED 基线的位置，不能当作 THz 的定量结论。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "stage3_4" / "figures"
DATA = ROOT / "outputs" / "stage3_4"
AUTHOR = ROOT / "data" / "author_reference"

C_H = "#137c8b"
C_V = "#d17a22"
C_ACC = "#8b4e75"
C_OK = "#3d6b3d"
C_GREY = "#4a4a4a"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
})


def fig1_radius_pulse() -> None:
    d = pd.read_csv(DATA / "stage34_static_v0.csv")
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    # 先铺色带：每 8 拍换一端，全高显示
    block_edges = sorted(d.time_s.unique())[:: 2 * 8]
    for edge in block_edges:
        ax.axvspan(edge, edge + 8 * 0.5, color=C_H if int(edge / 4) % 2 == 0 else C_V,
                   alpha=0.06, lw=0)
    for name, colour in (("R1", C_H), ("R2", C_V)):
        sub = d[d.terminal == name]
        ax.step(sub.time_s, sub.radius_deg, where="post", color=colour, lw=1.8,
                label=f"{name} 扫描半径")
    ax.set(xlabel="时间 / s", ylabel="扫描半径 δ_r / deg",
           title="双向时分交替：主动端才有半径，被动端门控归零")
    ax.set_ylim(0, 0.10)
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.02, 0.90, "浅色带 = 该端处于 Active，每块 8 拍", transform=ax.transAxes,
            fontsize=8.6, color=C_GREY)
    fig.savefig(OUT / "fig1_radius_pulse.png")
    plt.close(fig)


def fig2_distance_reference() -> None:
    d = pd.read_csv(AUTHOR / "author_fig8_distance.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(d.distances, d.ekf_error, "o-", color=C_H, lw=1.6, ms=5,
            label="EKF 指向误差")
    ax.plot(d.distances, d.es_error, "s--", color=C_V, lw=1.6, ms=5,
            label="传统 ES")
    for x, y in ((d.distances.iloc[0], d.ekf_error.iloc[0]),
                 (d.distances.iloc[-1], d.ekf_error.iloc[-1])):
        ax.annotate(f"{y:.3f}°", (x, y), textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.annotate(f"{d.es_error.iloc[0]:.3f}°", (d.distances.iloc[0], d.es_error.iloc[0]),
                textcoords="offset points", xytext=(8, -14), fontsize=9, color=C_V)
    ax.set(xlabel="距离 / m", ylabel="稳态指向误差 / deg",
           title="作者归档 MAT 数据（LED 信道）：EKF 随距离抬头，ES 近距离直接崩")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 30)
    ax.text(0.98, 0.55, "ES 在 0.92 m 处 28.250°，\n之后被限幅在 3° 附近",
            transform=ax.transAxes, ha="right", fontsize=8.6, color=C_V)
    fig.savefig(OUT / "fig2_distance_reference.png")
    plt.close(fig)


def fig3_dynamic_tracking() -> None:
    d = pd.read_csv(DATA / "stage34_dynamic_v0.2.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4), layout="constrained")
    ax.plot(d.time_s[::2], d.distance_m[::2], color=C_GREY, lw=1.5)
    ax.set(xlabel="时间 / s", ylabel="两端距离 / m",
           title="(a) 正交移动：Rover 沿 y、Elevator 沿 z")
    ax.text(0.04, 0.86, f"2.3000 → {d.distance_m.iloc[-1]:.4f} m", transform=ax.transAxes,
            fontsize=9)

    for name, colour in (("R1", C_H), ("R2", C_V)):
        sub = d[d.terminal == name].reset_index(drop=True)
        norm = np.hypot(sub.true_h_deg, sub.true_v_deg)
        ax2.plot(sub.time_s, norm, color=colour, lw=1.3, label=f"{name} 指向误差")
        ax2.annotate(f"{norm.iloc[-1]:.3f}°", (sub.time_s.iloc[-1], norm.iloc[-1]),
                     textcoords="offset points", xytext=(-6, 8), fontsize=9,
                     color=colour, ha="right")
    ax2.set(xlabel="时间 / s", ylabel="指向误差范数 / deg",
            title="(b) 两车末端误差：0.122° / 0.309°")
    ax2.legend(loc="upper right", frameon=False)
    ax2.set_ylim(0, 0.55)
    fig.savefig(OUT / "fig3_dynamic_tracking.png")
    plt.close(fig)


def fig4_speed_reference() -> None:
    d = pd.read_csv(AUTHOR / "author_fig10_speed.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(d.speeds, d.speed_ekf_error, "o-", color=C_H, lw=1.6, ms=5, label="EKF 指向误差")
    ax.plot(d.speeds, d.speed_es_error, "s--", color=C_V, lw=1.6, ms=5, label="传统 ES")
    ax.annotate(f"{d.speed_ekf_error.iloc[-1]:.3f}°",
                (d.speeds.iloc[-1], d.speed_ekf_error.iloc[-1]),
                textcoords="offset points", xytext=(-10, 10), fontsize=9, ha="right")
    ax.annotate(f"{d.speed_ekf_error.iloc[0]:.3f}°",
                (d.speeds.iloc[0], d.speed_ekf_error.iloc[0]),
                textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.set(xlabel="相对速度 / (cm/s)", ylabel="稳态指向误差 / deg",
           title="作者归档 MAT 数据（LED 信道）：过 1 cm/s 后准静态假设崩溃")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.30, "ES 曲线基本贴着 3°：\n它根本没跟上，也就无所谓发散",
            transform=ax.transAxes, ha="right", fontsize=8.6, color=C_V)
    fig.savefig(OUT / "fig4_speed_reference.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_radius_pulse()
    fig2_distance_reference()
    fig3_dynamic_tracking()
    fig4_speed_reference()
    print(f"figures -> {OUT}")
