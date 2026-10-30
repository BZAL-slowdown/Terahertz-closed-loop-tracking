"""阶段三/四：220 GHz 双向时分交替、自适应变半径与正交移动跟踪。

与手写 MATLAB 版本（阶段一二复现/matlab）的对应关系：
    simulate_stage3.m           -> Stage3Run（ns=8 点/块，两端互斥）
    ekf_bidirectional_step.m    -> _active_step / _passive_step（式 27-31 的标量分支）
    stage_config.m 的式(25)(26) -> RadiusLaw（窗口内 P_ang 最大特征值驱动）
    simulate_stage4_dynamic.m   -> Stage4Run（Rover 沿 y、Elevator 沿 z）

与 MATLAB 版的差别只有两处，都在文档里写明：
    1. 信道换成 220 GHz 实测方向图 + P.676，状态从 3 维扩到 6 维；
    2. 角度量纲按波束宽度换算：LED 的 15° 高斯对应 35.3° 半功率宽度，
       THz 实测波束只有 0.768°，所以初始偏差、扫描半径都按 HPBW 比例缩放。

运行：
    python code/stage3_4_thz.py --mode static
    python code/stage3_4_thz.py --mode dynamic --speed-factor 0.2
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import main_THZ_220GHz_deterministic_mechanical_tracking as thz
import stage1_2_thz as s12

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
WORKBOOK = WORKSPACE / "data" / "LWB_CRS-220-100MM-2406-2.xlsx"
OUT = WORKSPACE / "outputs" / "stage3_4"

# ---------------------------------------------------------------------------
# 参数：THz 侧按波束宽度换算，LED 侧原值写在注释里备查
# ---------------------------------------------------------------------------
DT_S = 0.5                  # 论文 Fig.7 的采样周期
NS = 8                      # 每端每块扫描点数，360°/45°
SCAN_STEP_DEG = 45.0
STEPS_STATIC = 160          # R1/R2 各 Active 80 拍
STEPS_DYNAMIC = 200
SEED = 20261030
MEASUREMENT_SIGMA_DB = 0.10

# LED 版：高斯宽度 15°，半功率宽度 2.355*15 = 35.3°，初始 8°/-6°，半径 2°-5°
LED_HPBW_DEG = 35.3
LED_INITIAL_DEG = np.array([8.0, -6.0])
LED_RADIUS_RANGE_DEG = (2.0, 5.0)


@dataclass
class Case:
    """一套 220 GHz 信道与几何参数。"""
    pattern: s12.WorkbookPattern
    link: thz.LinkBudget
    distance0_m: float
    hpbw: np.ndarray
    beam_scale: float               # THz HPBW / LED HPBW
    radius_min_deg: float
    radius_max_deg: float
    initial_error_deg: np.ndarray
    q_peak_dbm: float


def make_case(distance_m: float = 2.3) -> Case:
    pattern = s12.WorkbookPattern(WORKBOOK)
    link = thz.LinkBudget()
    hpbw = np.array([pattern.hpbw(0), pattern.hpbw(1)])
    beam_scale = float(np.mean(hpbw)) / LED_HPBW_DEG
    q_peak = link.received_power_dbm(distance_m, pattern.h_peak, pattern.e_peak)
    return Case(
        pattern=pattern,
        link=link,
        distance0_m=distance_m,
        hpbw=hpbw,
        beam_scale=beam_scale,
        radius_min_deg=LED_RADIUS_RANGE_DEG[0] * beam_scale,
        radius_max_deg=min(LED_RADIUS_RANGE_DEG[1] * beam_scale,
                           RADIUS_MAX_SCALE * float(np.mean(hpbw))),
        initial_error_deg=LED_INITIAL_DEG * beam_scale,
        q_peak_dbm=q_peak,
    )


# ---------------------------------------------------------------------------
# 变半径：论文式(25)(26) 的窗口求和结构
# ---------------------------------------------------------------------------
K_DELTA_AUTHOR = 1.0 / (7000.0 * NS)   # 作者 NodeEKF.m 的窗口增益，仅作对照
K_RADIUS = 2.2                          # 本文：半径 = K_RADIUS × 角度标准差
FEEDBACK_GAIN = 1.0                     # 回中增益；1.0 会把估计噪声积成随机游走
RADIUS_MAX_SCALE = 0.11                 # 半径上限按 HPBW 折算：5°×0.0218≈0.109°


class RadiusLaw:
    """窗口内 P_ang 最大特征值驱动扫描半径。

    论文式(26)写的是 r = K_delta·Σλ_max，再限幅到 [δ_l, δ_h]。作者的 P_ang 被
    Q_scaling=1e5 放大过，特征值在 1e4 deg² 量级，乘 K_delta=1/(7000·ns) 正好落在
    2°–5°；本滤波器直接用真实 deg²，特征值只有 1e-4 量级，直接乘会恒等于 0。
    因此保留“窗口求和 + 限幅”的结构，把尺度换成角度标准差 sqrt(λ_max)。
    """

    def __init__(self, r_min_deg: float, r_max_deg: float, window: int = NS):
        self.r_min = float(r_min_deg)
        self.r_max = float(r_max_deg)
        self.window = int(window)

    def update(self, eigen_history: list[float]) -> float:
        if not eigen_history:
            return self.r_max
        recent = eigen_history[-self.window:]
        sigma_deg = math.sqrt(max(float(np.mean(recent)), 0.0))
        radius = K_RADIUS * sigma_deg
        return float(np.clip(radius, self.r_min, self.r_max))


# ---------------------------------------------------------------------------
# 单端终端：6 状态 IEKF + 云台指向误差
# ---------------------------------------------------------------------------
class Terminal:
    def __init__(self, name: str, case: Case, seed: int):
        self.name = name
        self.case = case
        self.rng = np.random.default_rng(seed)
        self.filter = thz.IEKF(
            case.pattern, case.q_peak_dbm,
            angular_accel_deg_s2=0.003,
            measurement_design_sigma_db=0.18,
            q_accel_sigma_db_per_s2=0.015,
        )
        self.filter.x[:] = [case.q_peak_dbm, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.filter.p = np.diag([4.0, 0.08,
                                 (0.8 * case.hpbw[0]) ** 2,
                                 (0.8 * case.hpbw[1]) ** 2, 0.02, 0.02])
        # 真实指向误差（相对视距），不是滤波器状态
        self.error = case.initial_error_deg.copy()
        self.radius_law = RadiusLaw(case.radius_min_deg, case.radius_max_deg)
        self.radius = case.radius_max_deg
        self.eigen_window: list[float] = []
        self.q_history: list[float] = []
        self.q_true_dbm = case.q_peak_dbm
        self.pending: list[tuple[np.ndarray, float, float]] = []

    # -- 量测 ---------------------------------------------------------------
    def set_distance(self, distance_m: float) -> None:
        """链路基线随距离变化：q 真值按 FSPL + 气体项重算。"""
        self.q_true_dbm = self.case.link.received_power_dbm(
            max(distance_m, 0.1), self.case.pattern.h_peak, self.case.pattern.e_peak)

    def _sample(self, offset: np.ndarray, age_s: float) -> tuple[np.ndarray, float, float]:
        apparent = self.error - offset
        gain = self.case.pattern.local_relative_gain(float(apparent[0]), float(apparent[1]))
        z = (self.q_true_dbm + gain
             + float(self.rng.normal(0.0, MEASUREMENT_SIGMA_DB)))
        return offset.copy(), z, age_s

    # -- Active：块内采样，块末一次 6 状态批量更新 -------------------------
    def scan_sample(self, psi_deg: float, age_s: float) -> None:
        """块内每个扫描点：只做预测并把样本排队，不做更新。"""
        self.filter.predict(DT_S)
        offset = self.radius * np.array([math.cos(math.radians(psi_deg)),
                                         math.sin(math.radians(psi_deg))])
        self.pending.append(self._sample(offset, age_s))

    def finish_block(self) -> dict[str, float]:
        """一圈扫完：8 个方位的样本一起做 IEKF 批量更新。

        逐点顺序更新时，单点量测分不清“整体掉功率”和“角度偏了”，
        V 面平肩上甚至会把符号判反；8 点联立才能把共同项 q 与角度项分开。
        """
        result = self.filter.batch_update(self.pending)
        self.pending = []
        p_ang = 0.5 * (self.filter.p[2:4, 2:4] + self.filter.p[2:4, 2:4].T)
        lambda_max = float(np.linalg.eigvalsh(p_ang).max())
        return {
            "lambda_max_deg2": lambda_max,
            "accepted": float(result.accepted),
            "innovation_score": float(result.normalized_innovation_score),
        }

    # -- Passive：式(27)-(31) 的标量退化 ------------------------------------
    def passive_step(self) -> dict[str, float]:
        """只动 q 与 P(1,1)，四个角度状态、q_dot 及全部交叉项原样携带。

        与全维更新的区别不只是“少更新几个状态”：这里根本没有 6×6 求逆，
        Kalman 增益退化成标量 K1 = P11/(P11 + R)，因此不会产生任何数值漂移。
        """
        x_before = self.filter.x.copy()
        p_before = self.filter.p.copy()

        # 被动期仍收到一个 0 偏置的功率样本（对端在扫，本端只收不发）
        _, z, _ = self._sample(np.zeros(2), 0.0)
        predicted = (self.filter.x[0] + self.case.pattern.local_relative_gain(
            float(self.filter.x[2]), float(self.filter.x[3])))
        innovation = z - predicted

        # 式(27)：被动期 q 不做动力学外推（作者原文 x1_forecast = x1）。
        # q_dot 在被动期是冻结的，若拿它外推 q，一个估计偏大的 q_dot 会在
        # 8 拍里把基线推走 1 dB 以上，转主动后角度就被这个偏置带跑。
        # 式(28)：只有 P11 加过程噪声
        self.filter.p[0, 0] = self.filter.p[0, 0] + 1e-8

        # 式(29)-(31)：1D 标量更新。只对 q 这一行/列做边缘更新，
        # 角状态与 P_ang 一个元素都不碰；交叉项按 α=(1-K1) 同步缩放，
        # 这样 P 依旧是半正定的（等价于 D·P·D 的相似缩放），不会破坏下一拍预测。
        # 被动期量测里混着“角度未知”造成的损耗，不能全算到 q 头上，
        # 否则 P00 会被 8 次更新压死在一个带偏的 q 上，转主动后再也拉不回来。
        # 这里用 ±1σ 的指向不确定度去估损耗不确定度：在峰顶处梯度为零，
        # 但偏离 1σ 的损耗可能是好几 dB，单看梯度会严重低估。
        p_ang = 0.5 * (self.filter.p[2:4, 2:4] + self.filter.p[2:4, 2:4].T)
        eig = np.linalg.eigvalsh(p_ang)
        sigma_h = math.sqrt(max(float(eig.max()), 0.0))
        sigma_v = math.sqrt(max(float(eig.min()), 0.0))
        h0, v0 = float(self.filter.x[2]), float(self.filter.x[3])
        g0 = self.case.pattern.local_relative_gain(h0, v0)
        loss_spread = max(
            abs(self.case.pattern.local_relative_gain(h0 + sigma_h, v0) - g0),
            abs(self.case.pattern.local_relative_gain(h0 - sigma_h, v0) - g0),
            abs(self.case.pattern.local_relative_gain(h0, v0 + sigma_v) - g0),
            abs(self.case.pattern.local_relative_gain(h0, v0 - sigma_v) - g0),
        )
        r_eff = thz.EKF_ASSUMED_MEASUREMENT_SIGMA_DB ** 2 + loss_spread ** 2

        s = self.filter.p[0, 0] + r_eff
        k1 = self.filter.p[0, 0] / max(s, 1e-15)
        self.filter.x[0] = self.filter.x[0] + k1 * innovation
        alpha = 1.0 - k1
        self.filter.p[0, 0] = max(alpha * self.filter.p[0, 0], 1e-15)
        self.filter.p[0, 1:] = alpha * self.filter.p[0, 1:]
        self.filter.p[1:, 0] = alpha * self.filter.p[1:, 0]

        # 硬性断言：除 x0 与 q 行/列外，角状态与 P_ang 必须逐位不变
        frozen_states = np.max(np.abs(self.filter.x[1:] - x_before[1:]))
        idx = [1, 2, 3, 4, 5]
        frozen_cov = np.max(np.abs(self.filter.p[np.ix_(idx, idx)] - p_before[np.ix_(idx, idx)]))
        if max(frozen_states, frozen_cov) > 1e-12:
            raise AssertionError(
                f"{self.name} Passive 冻结失效: states={frozen_states:.3e}, "
                f"cov={frozen_cov:.3e}")
        return {
            "frozen_state_delta": float(frozen_states),
            "frozen_cov_delta": float(frozen_cov),
        }

    # -- 控制与真实指向 -----------------------------------------------------
    def apply_los_drift(self, los_rate_deg_s: np.ndarray, add_noise: bool = True) -> None:
        """视距转动与平台抖动：只改真实指向，不动滤波器状态。"""
        self.error = self.error + los_rate_deg_s * DT_S
        if add_noise:
            self.error = self.error + self.rng.normal(0.0, 0.002, size=2)

    def block_end_command(self) -> np.ndarray:
        """扫描一圈后回中：估计值乘阻尼增益，再把滤波器切平面原点平移过去。

        块内不逐拍回中，是为了让 8 个不同方位的样本把“角度误差”和“功率基线”
        分开；逐拍回中会把估计清零，q 会把角度信息全部吃掉。
        增益取 0.6 而不是 1：0.77° 的波束上，V 面平肩的斜率只有 0.8 dB/deg，
        单位增益会把估计噪声直接积分成真实指向的随机游走。
        """
        command = np.clip(FEEDBACK_GAIN * self.filter.error,
                          -0.5 * self.case.hpbw, 0.5 * self.case.hpbw)
        self.error = self.error - command
        self.filter.x[2:4] -= command
        return command

    def refresh_radius(self) -> None:
        p_ang = 0.5 * (self.filter.p[2:4, 2:4] + self.filter.p[2:4, 2:4].T)
        self.eigen_window.append(float(np.linalg.eigvalsh(p_ang).max()))
        self.radius = self.radius_law.update(self.eigen_window)


# ---------------------------------------------------------------------------
# 几何：正交移动（Fig.9 构型）
# ---------------------------------------------------------------------------
@dataclass
class OrthogonalMotion:
    p_rover: np.ndarray = field(default_factory=lambda: np.array([1.5, -1.37, 0.25]))
    p_elevator: np.ndarray = field(default_factory=lambda: np.array([-0.8, -1.37, 0.25]))
    speed_factor: float = 0.0

    def step(self) -> None:
        # 作者 MultiSpeedSimulation.m：每步位移 0.005·v/√2，两端各走一路
        d = 0.005 * self.speed_factor / math.sqrt(2.0)
        self.p_rover[1] += d      # Rover 沿 +y 水平走
        self.p_elevator[2] += d   # Elevator 沿 +z 垂直爬

    def distance(self) -> float:
        return float(np.linalg.norm(self.p_elevator - self.p_rover))

    def los_az_el_deg(self, from_first: bool = True) -> np.ndarray:
        v = (self.p_elevator - self.p_rover) if from_first else (self.p_rover - self.p_elevator)
        az = math.degrees(math.atan2(v[1], v[0]))
        el = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        return np.array([az, el])


def los_rate(prev: np.ndarray, cur: np.ndarray, dt: float = DT_S) -> np.ndarray:
    delta = cur - prev
    delta[0] = (delta[0] + 180.0) % 360.0 - 180.0
    return delta / dt


# ---------------------------------------------------------------------------
# 主循环：两端互斥时分
# ---------------------------------------------------------------------------
def run(mode: str, steps: int, speed_factor: float, seed: int = SEED) -> dict:
    case = make_case(2.3)
    motion = OrthogonalMotion(speed_factor=speed_factor)
    terminals = [Terminal("R1", case, seed + 1), Terminal("R2", case, seed + 2)]

    rows: list[dict] = []
    prev_los = motion.los_az_el_deg()
    frozen_failures = 0

    for k in range(steps):
        slot = (k % (2 * NS))
        active_index = 0 if slot < NS else 1
        sample = slot % NS
        psi = sample * SCAN_STEP_DEG
        # 样本年龄：块内越早采到的越老
        age_s = (NS - 1 - sample) * DT_S / NS

        if speed_factor > 0.0:
            motion.step()
        cur_los = motion.los_az_el_deg()
        rate = los_rate(prev_los, cur_los) if speed_factor > 0.0 else np.zeros(2)
        prev_los = cur_los
        for terminal in terminals:
            terminal.set_distance(motion.distance())

        for index, terminal in enumerate(terminals):
            active = index == active_index
            if active:
                terminal.scan_sample(psi, age_s)
                diagnostics = {"lambda_max_deg2": float("nan"),
                               "frozen_state_delta": 0.0, "frozen_cov_delta": 0.0}
                command = np.zeros(2)
                if sample == NS - 1:
                    diagnostics = terminal.finish_block()
                    command = terminal.block_end_command()
                    terminal.refresh_radius()
            else:
                diagnostics = terminal.passive_step()
                if max(diagnostics["frozen_state_delta"],
                       diagnostics["frozen_cov_delta"]) > 1e-12:
                    frozen_failures += 1
                command = np.zeros(2)
            terminal.apply_los_drift(rate)
            rows.append({
                "time_s": k * DT_S,
                "terminal": terminal.name,
                "active": int(active),
                "sample_in_block": sample,
                "radius_deg": float(terminal.radius if active else 0.0),
                "command_h_deg": float(command[0]),
                "command_v_deg": float(command[1]),
                "true_h_deg": float(terminal.error[0]),
                "true_v_deg": float(terminal.error[1]),
                "est_h_deg": float(terminal.filter.x[2]),
                "est_v_deg": float(terminal.filter.x[3]),
                "q_dbm": float(terminal.filter.x[0]),
                "q_dot_db_s": float(terminal.filter.x[1]),
                "lambda_max_deg2": diagnostics.get("lambda_max_deg2", float("nan")),
                "frozen_state_delta": diagnostics.get("frozen_state_delta", 0.0),
                "frozen_cov_delta": diagnostics.get("frozen_cov_delta", 0.0),
                "distance_m": motion.distance(),
            })

    return {
        "mode": mode,
        "steps": steps,
        "dt_s": DT_S,
        "speed_factor": speed_factor,
        "distance_start_m": 2.3,
        "distance_end_m": motion.distance(),
        "beam_scale": case.beam_scale,
        "radius_range_deg": [case.radius_min_deg, case.radius_max_deg],
        "initial_error_h_v_deg": case.initial_error_deg.tolist(),
        "frozen_assert_failures": frozen_failures,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="220 GHz 双向时分跟瞄：阶段三/四")
    parser.add_argument("--mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--speed-factor", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    steps = args.steps or (STEPS_STATIC if args.mode == "static" else STEPS_DYNAMIC)
    speed = 0.0 if args.mode == "static" else args.speed_factor
    OUT.mkdir(parents=True, exist_ok=True)
    result = run(args.mode, steps, speed, args.seed)

    rows = result.pop("rows")
    tag = f"{args.mode}_v{speed:g}"
    _write_csv(OUT / f"stage34_{tag}.csv", rows)

    summary = dict(result)
    terminal_rows = [r for r in rows if r["active"] == 1]
    for name in ("R1", "R2"):
        subset = [r for r in rows if r["terminal"] == name]
        tail = subset[-10:]
        final = subset[-1]
        summary[name] = {
            "final_error_deg": float(np.hypot(final["true_h_deg"], final["true_v_deg"])),
            "tail10_error_rms_deg": float(np.sqrt(np.mean(
                [np.hypot(r["true_h_deg"], r["true_v_deg"]) ** 2 for r in tail]))),
            "radius_final_deg": float(subset[-1]["radius_deg"]),
            "q_final_dbm": float(final["q_dbm"]),
        }
    summary["max_frozen_state_delta"] = float(max(r["frozen_state_delta"] for r in rows))
    summary["max_frozen_cov_delta"] = float(max(r["frozen_cov_delta"] for r in rows))
    summary["active_samples"] = len(terminal_rows)

    (OUT / f"stage34_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
