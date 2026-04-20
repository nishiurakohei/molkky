"""
molkky_sim.py
=============
モルック棒（円柱）の転がりシミュレーション
- 空間的にガウス分布する摩擦係数
- 初期条件への摂動安定性解析
- モンテカルロ解析

使い方:
    python molkky_sim.py                     # デフォルト設定で全解析を実行
    python molkky_sim.py --v0 4.0 --omega0 10 # 初期条件を指定
    python molkky_sim.py --mode sensitivity  # 感度解析のみ
    python molkky_sim.py --mode montecarlo   # モンテカルロのみ
    python molkky_sim.py --mode trajectory   # 軌道比較のみ
    python molkky_sim.py --mode all          # 全解析（デフォルト）
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────
# 1. パラメータ定義
# ─────────────────────────────────────────

@dataclass
class CylinderParams:
    """円柱（棒）の物理パラメータ"""
    radius: float = 0.03    # 半径 [m]
    mass: float   = 0.22    # 質量 [kg]

    @property
    def inertia(self) -> float:
        """中実円柱の慣性モーメント I = (1/2)mR²"""
        return 0.5 * self.mass * self.radius ** 2


@dataclass
class GroundParams:
    """地面の摩擦パラメータ"""
    mu_mean:  float = 0.30   # 摩擦係数の平均
    mu_sigma: float = 0.05   # 摩擦係数の標準偏差
    patch_size: float = 0.01 # 空間パッチサイズ [m]（この長さごとに μ が変化）


@dataclass
class InitialCondition:
    """初期条件"""
    v0:     float = 3.0   # 初期重心速度 [m/s]
    omega0: float = 8.0   # 初期回転角速度 [rad/s]
    theta0: float = 0.0   # 着地時の軸傾き [deg]


@dataclass
class SimConfig:
    """シミュレーション設定"""
    dt:     float = 0.005  # 時間ステップ [s]
    t_max:  float = 10.0   # 最大シミュレーション時間 [s]
    x_max:  float = 30.0   # 最大到達距離 [m]（計算打ち切り）
    v_stop: float = 0.005  # 停止判定速度 [m/s]


# ─────────────────────────────────────────
# 2. 摩擦フィールド生成
# ─────────────────────────────────────────

class FrictionField:
    """
    空間的にガウス分布する摩擦係数フィールド。
    x 座標を patch_size で量子化し、各セルに N(μ, σ²) の値を割り当てる。
    """

    def __init__(self, ground: GroundParams, seed: Optional[int] = None):
        self.ground = ground
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, float] = {}

    def get(self, x: float) -> float:
        """位置 x での摩擦係数を返す（未訪問セルは新規サンプリング）"""
        key = int(x / self.ground.patch_size)
        if key not in self._cache:
            mu = self.rng.normal(self.ground.mu_mean, self.ground.mu_sigma)
            self._cache[key] = float(np.clip(mu, 0.01, 1.0))
        return self._cache[key]


# ─────────────────────────────────────────
# 3. 1回のシミュレーション（ODE 前進差分）
# ─────────────────────────────────────────

@dataclass
class SimResult:
    """シミュレーション結果"""
    t:      np.ndarray   # 時刻 [s]
    x:      np.ndarray   # 重心位置 [m]
    v:      np.ndarray   # 並進速度 [m/s]
    omega:  np.ndarray   # 回転角速度 [rad/s]
    phi:    np.ndarray   # 軸傾き [rad]
    mu_arr: np.ndarray   # 各ステップでの μ
    dist:   float = 0.0  # 最終停止距離 [m]


def run_simulation(
    ic: InitialCondition,
    cyl: CylinderParams,
    ground: GroundParams,
    cfg: SimConfig,
    seed: Optional[int] = None,
) -> SimResult:
    """
    円柱の転がりを数値積分する。

    運動方程式:
        m * dv/dt  = F_f + m*g*sin(φ)          (並進)
        I * dω/dt  = F_f * R * cos(φ)          (回転)
        dφ/dt      = -k_φ * φ - k_v * |v| * φ  (傾き復元)

    接触点速度 v_c = v - R*ω*cos(φ) でスリップ判定:
        スリップあり: F_f = -sign(v_c) * μ * N  (動摩擦)
        スリップなし: rolling constraint から F_f を計算
    """
    R, m, I = cyl.radius, cyl.mass, cyl.inertia
    g  = 9.81
    dt = cfg.dt

    phi0 = np.deg2rad(ic.theta0)
    friction = FrictionField(ground, seed=seed)

    # 状態変数
    x, v, omega, phi = 0.0, float(ic.v0), float(ic.omega0), phi0

    # 記録用リスト
    ts, xs, vs, omegas, phis, mus = [0.0], [x], [v], [omega], [phi], [friction.get(0.0)]

    t = 0.0
    while t < cfg.t_max and x < cfg.x_max and v > cfg.v_stop:
        mu  = friction.get(x)
        N   = m * g * np.cos(phi)        # 法線力

        # 接触点速度
        v_contact = v - R * omega * np.cos(phi)

        # 摩擦力の計算
        if abs(v_contact) > 1e-3:
            # 動摩擦（スリップ）
            F_f = -np.sign(v_contact) * mu * N
        else:
            # 転がり拘束: dv/dt = R*cos(φ)*dω/dt
            # => F_f/m = R*cos(φ)*(F_f*R*cos(φ)/I)
            # => F_f * (1/m + R²cos²φ/I) = -g*sin(φ) （重力の寄与）
            cos_phi = np.cos(phi)
            denom = 1.0 / m + (R * cos_phi) ** 2 / I
            F_f = -g * np.sin(phi) / denom

        # 加速度
        a_v     = (F_f - m * g * np.sin(phi)) / m
        a_omega = F_f * R * np.cos(phi) / I

        # 傾き復元（自己整合モデル: 速度が大きいほど速く立ち直る）
        k_phi = 0.5
        k_v   = 0.1
        dphi  = -(k_phi + k_v * abs(v)) * phi

        # 前進オイラー積分
        v     = max(0.0, v     + a_v     * dt)
        omega =         omega + a_omega  * dt
        phi   =         phi   + dphi     * dt
        x     =         x     + v        * dt
        t     +=                          dt

        # 間引いて記録（ファイルサイズ・速度のため 0.02 s ごと）
        if len(ts) == 0 or t - ts[-1] >= 0.02:
            ts.append(t); xs.append(x); vs.append(v)
            omegas.append(omega); phis.append(phi); mus.append(mu)

    # 最終点を必ず追加
    ts.append(t); xs.append(x); vs.append(v)
    omegas.append(omega); phis.append(phi); mus.append(mu)

    return SimResult(
        t=np.array(ts), x=np.array(xs), v=np.array(vs),
        omega=np.array(omegas), phi=np.array(phis), mu_arr=np.array(mus),
        dist=x,
    )


# ─────────────────────────────────────────
# 4. 摂動安定性解析
# ─────────────────────────────────────────

@dataclass
class SensitivityResult:
    """感度解析の結果"""
    param_names:  list[str]
    nominal_dist: float
    perturbed_dists: dict[str, float]
    sensitivity_indices: dict[str, float]   # S_i = |Δx| / (x₀ * ε)
    trajectories: dict[str, SimResult]


def sensitivity_analysis(
    ic: InitialCondition,
    cyl: CylinderParams,
    ground: GroundParams,
    cfg: SimConfig,
    epsilon: float = 0.05,
    seed: int = 42,
) -> SensitivityResult:
    """
    各初期条件パラメータに対する 1 次感度指数を計算する。

    S_i = |x(p_i*(1+ε)) - x(p_i)| / (x(p_i) * ε)

    対象パラメータ: v0, ω0, θ0, μ̄
    """
    nom = run_simulation(ic, cyl, ground, cfg, seed=seed)
    x0  = nom.dist if nom.dist > 1e-6 else 1e-6

    perturbations = {
        "Δv₀": InitialCondition(ic.v0 * (1 + epsilon), ic.omega0, ic.theta0),
        "Δω₀": InitialCondition(ic.v0, ic.omega0 * (1 + epsilon), ic.theta0),
        "Δθ₀": InitialCondition(ic.v0, ic.omega0, ic.theta0 + epsilon * 30),
        "Δμ̄":  None,   # 摩擦パラメータは ground を変更
    }

    trajs    = {"名義": nom}
    p_dists  = {}
    s_index  = {}

    for name, ic_p in perturbations.items():
        if ic_p is not None:
            res = run_simulation(ic_p, cyl, ground, cfg, seed=seed)
        else:
            g_p = GroundParams(ground.mu_mean * (1 + epsilon), ground.mu_sigma, ground.patch_size)
            res = run_simulation(ic, cyl, g_p, cfg, seed=seed)

        p_dists[name] = res.dist
        s_index[name] = abs(res.dist - x0) / (x0 * epsilon + 1e-9)
        trajs[name]   = res

    return SensitivityResult(
        param_names=list(perturbations.keys()),
        nominal_dist=nom.dist,
        perturbed_dists=p_dists,
        sensitivity_indices=s_index,
        trajectories=trajs,
    )


# ─────────────────────────────────────────
# 5. モンテカルロ解析
# ─────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """モンテカルロ解析の結果"""
    dists:    np.ndarray   # 各試行の停止距離
    mean:     float
    std:      float
    cv:       float        # 変動係数 = std / mean
    p5:       float        # 5 パーセンタイル
    p95:      float        # 95 パーセンタイル
    is_stable: bool        # CV < 0.15 を安定と判定


def monte_carlo(
    ic: InitialCondition,
    cyl: CylinderParams,
    ground: GroundParams,
    cfg: SimConfig,
    n_samples: int = 500,
    epsilon: float = 0.05,
    rng_seed: int = 0,
) -> MonteCarloResult:
    """
    全パラメータに独立なガウス摂動を加えて N 回シミュレーション。

    v0, ω0 は ±ε*100% の相対ノイズ
    θ0       は ±ε*30° の絶対ノイズ
    μ̄        は ground.mu_sigma で別途変動（FrictionField で処理済み）
    """
    rng = np.random.default_rng(rng_seed)
    dists = np.empty(n_samples)

    for i in range(n_samples):
        v0_p  = ic.v0    * (1 + epsilon * rng.standard_normal())
        w0_p  = ic.omega0 * (1 + epsilon * rng.standard_normal())
        th0_p = ic.theta0 + epsilon * 30 * rng.standard_normal()
        ic_p  = InitialCondition(max(0.1, v0_p), max(0.0, w0_p), th0_p)
        res   = run_simulation(ic_p, cyl, ground, cfg, seed=int(rng.integers(0, 2**31)))
        dists[i] = res.dist

    mean = float(np.mean(dists))
    std  = float(np.std(dists))
    cv   = std / mean if mean > 1e-6 else 0.0

    return MonteCarloResult(
        dists=dists, mean=mean, std=std, cv=cv,
        p5=float(np.percentile(dists, 5)),
        p95=float(np.percentile(dists, 95)),
        is_stable=cv < 0.15,
    )


# ─────────────────────────────────────────
# 6. プロット
# ─────────────────────────────────────────

COLORS = {
    "名義":  "#2f80ed",
    "Δv₀":  "#e24b4a",
    "Δω₀":  "#ef9f27",
    "Δθ₀":  "#534ab7",
    "Δμ̄":   "#1d9e75",
}
DASHES = {
    "名義":  "-",
    "Δv₀":  "--",
    "Δω₀":  "-.",
    "Δθ₀":  ":",
    "Δμ̄":   "--",
}


def plot_trajectories(sens: SensitivityResult, save_path: Optional[str] = None):
    """軌道比較プロット（距離・速度・角速度・傾き）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("軌道比較：名義 vs 各パラメータへの摂動", fontsize=13, y=1.01)

    ylabels = ["位置 x [m]", "速度 v [m/s]", "角速度 ω [rad/s]", "傾き φ [deg]"]
    keys    = ["x", "v", "omega", "phi"]

    for ax, ylabel, key in zip(axes.flat, ylabels, keys):
        for name, res in sens.trajectories.items():
            arr = getattr(res, key)
            if key == "phi":
                arr = np.rad2deg(arr)
            ax.plot(res.t, arr,
                    color=COLORS.get(name, "gray"),
                    linestyle=DASHES.get(name, "-"),
                    lw=1.8 if name == "名義" else 1.2,
                    label=name, alpha=0.9)
        ax.set_xlabel("時間 [s]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.5)
        ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_phase_space(sens: SensitivityResult, save_path: Optional[str] = None):
    """位相空間プロット（v–ω 平面）"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("位相空間：(v, ω) アトラクター構造", fontsize=13)

    for name, res in sens.trajectories.items():
        axes[0].plot(res.v, res.omega,
                     color=COLORS.get(name, "gray"),
                     linestyle=DASHES.get(name, "-"),
                     lw=1.8 if name == "名義" else 1.2,
                     label=name, alpha=0.85)
        axes[1].plot(res.v, np.rad2deg(res.phi),
                     color=COLORS.get(name, "gray"),
                     linestyle=DASHES.get(name, "-"),
                     lw=1.8 if name == "名義" else 1.2,
                     label=name, alpha=0.85)

    axes[0].set_xlabel("速度 v [m/s]"); axes[0].set_ylabel("角速度 ω [rad/s]")
    axes[1].set_xlabel("速度 v [m/s]"); axes[1].set_ylabel("傾き φ [deg]")
    for ax in axes:
        ax.legend(fontsize=8, framealpha=0.5)
        ax.grid(True, lw=0.4, alpha=0.5)
        # 始点をマーク
        for name, res in sens.trajectories.items():
            ax.plot(res.v[0], (res.omega[0] if ax == axes[0] else np.rad2deg(res.phi[0])),
                    "o", color=COLORS.get(name, "gray"), ms=5, alpha=0.7)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_sensitivity(sens: SensitivityResult, save_path: Optional[str] = None):
    """感度指数の棒グラフ"""
    names  = list(sens.sensitivity_indices.keys())
    values = [sens.sensitivity_indices[n] for n in names]
    colors = [COLORS.get(n, "gray") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"摂動感度解析  （名義停止距離 = {sens.nominal_dist:.2f} m）", fontsize=13)

    # 感度指数
    bars = axes[0].bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].axhline(0.5, color="red", lw=1, linestyle="--", label="不安定閾値 S=0.5")
    axes[0].set_ylabel("正規化感度指数 Sᵢ", fontsize=10)
    axes[0].set_title("Sᵢ = |Δx| / (x₀ · ε)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", lw=0.4, alpha=0.5)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    # 停止距離の比較
    dist_vals = [sens.nominal_dist] + [sens.perturbed_dists[n] for n in names]
    dist_labels = ["名義"] + names
    dist_colors = ["#2f80ed"] + colors
    axes[1].bar(dist_labels, dist_vals, color=dist_colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("停止距離 [m]", fontsize=10)
    axes[1].set_title("各摂動での停止距離", fontsize=10)
    axes[1].grid(axis="y", lw=0.4, alpha=0.5)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_monte_carlo(mc: MonteCarloResult, ic: InitialCondition,
                     save_path: Optional[str] = None):
    """モンテカルロ結果のヒストグラム＋統計サマリー"""
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle(
        f"モンテカルロ解析  N={len(mc.dists)}  "
        f"（{'安定 ✓' if mc.is_stable else '不安定 ✗'}  CV={mc.cv:.3f}）",
        fontsize=13
    )

    # ヒストグラム
    counts, bins, patches = ax1.hist(
        mc.dists, bins=30, color="#2f80ed", edgecolor="white",
        linewidth=0.6, alpha=0.85, label="停止距離"
    )
    ax1.axvline(mc.mean, color="red",    lw=1.8, linestyle="-",  label=f"平均 {mc.mean:.2f} m")
    ax1.axvline(mc.p5,   color="orange", lw=1.2, linestyle="--", label=f"5%ile {mc.p5:.2f} m")
    ax1.axvline(mc.p95,  color="orange", lw=1.2, linestyle="--", label=f"95%ile {mc.p95:.2f} m")

    # 正規分布フィット
    x_fit = np.linspace(mc.dists.min(), mc.dists.max(), 200)
    bin_w = bins[1] - bins[0]
    ax1.plot(x_fit, norm.pdf(x_fit, mc.mean, mc.std) * len(mc.dists) * bin_w,
             "k--", lw=1.2, label="正規分布フィット")

    ax1.set_xlabel("停止距離 [m]", fontsize=10)
    ax1.set_ylabel("頻度", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.5)
    ax1.grid(True, lw=0.4, alpha=0.5)

    # 統計サマリーテキスト
    stats = [
        ("平均 μ",   f"{mc.mean:.3f} m"),
        ("標準偏差 σ", f"{mc.std:.3f} m"),
        ("変動係数 CV", f"{mc.cv:.3f}"),
        ("5%ile",   f"{mc.p5:.3f} m"),
        ("95%ile",  f"{mc.p95:.3f} m"),
        ("90% 区間幅", f"{mc.p95 - mc.p5:.3f} m"),
        ("安定性",   "安定 ✓" if mc.is_stable else "不安定 ✗"),
    ]
    ax2.axis("off")
    y = 0.95
    ax2.text(0.05, y, "統計サマリー", fontsize=11, fontweight="bold",
             transform=ax2.transAxes, va="top")
    for label, value in stats:
        y -= 0.11
        ax2.text(0.05, y, label,   fontsize=9, color="gray",
                 transform=ax2.transAxes, va="top")
        ax2.text(0.60, y, value,   fontsize=9, fontweight="bold",
                 transform=ax2.transAxes, va="top",
                 color="green" if "安定" in value else ("red" if "不安定" in value else "black"))

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_friction_field(ground: GroundParams, x_max: float = 5.0,
                        seed: int = 42, save_path: Optional[str] = None):
    """摩擦係数の空間分布を可視化"""
    field = FrictionField(ground, seed=seed)
    xs    = np.arange(0, x_max, ground.patch_size)
    mus   = np.array([field.get(x) for x in xs])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(xs, mus, where="post", color="#2f80ed", lw=1.2, label="μ(x)")
    ax.axhline(ground.mu_mean,  color="red",    lw=1.2, linestyle="--",
               label=f"平均 μ̄ = {ground.mu_mean:.2f}")
    ax.axhline(ground.mu_mean + ground.mu_sigma, color="orange", lw=0.8, linestyle=":",
               label=f"±σ = {ground.mu_sigma:.2f}")
    ax.axhline(ground.mu_mean - ground.mu_sigma, color="orange", lw=0.8, linestyle=":")
    ax.fill_between(xs, ground.mu_mean - ground.mu_sigma,
                        ground.mu_mean + ground.mu_sigma,
                    alpha=0.1, color="orange")
    ax.set_xlabel("位置 x [m]", fontsize=10)
    ax.set_ylabel("摩擦係数 μ", fontsize=10)
    ax.set_title("地面摩擦係数の空間分布（ガウス分布サンプリング）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_ylim(0, min(1.0, ground.mu_mean + 4 * ground.mu_sigma))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, path: Optional[str]):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  保存: {path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────
# 7. メインエントリーポイント
# ─────────────────────────────────────────

def print_summary(sens: SensitivityResult, mc: Optional[MonteCarloResult] = None):
    print("\n" + "=" * 55)
    print("  モルックシミュレーション 解析サマリー")
    print("=" * 55)
    print(f"  名義停止距離:  {sens.nominal_dist:.3f} m")
    print("\n  [ 感度指数 Sᵢ ]")
    for name, s in sens.sensitivity_indices.items():
        bar   = "█" * int(s * 20)
        tag   = " ⚠ 不安定" if s > 0.5 else ""
        print(f"  {name:6s}  {s:.4f}  {bar}{tag}")
    if mc:
        print(f"\n  [ モンテカルロ  N={len(mc.dists)} ]")
        print(f"  平均     : {mc.mean:.3f} m")
        print(f"  標準偏差 : {mc.std:.3f} m")
        print(f"  変動係数 : {mc.cv:.3f}  →  {'安定 ✓' if mc.is_stable else '不安定 ✗'}")
        print(f"  90%区間  : [{mc.p5:.2f}, {mc.p95:.2f}] m")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="モルック棒 転がりシミュレーション")
    parser.add_argument("--v0",     type=float, default=3.0,   help="初期速度 [m/s]")
    parser.add_argument("--omega0", type=float, default=8.0,   help="初期回転速度 [rad/s]")
    parser.add_argument("--theta0", type=float, default=0.0,   help="初期傾き [deg]")
    parser.add_argument("--mu",     type=float, default=0.30,  help="摩擦係数平均")
    parser.add_argument("--sigma",  type=float, default=0.05,  help="摩擦係数標準偏差")
    parser.add_argument("--eps",    type=float, default=0.05,  help="摂動強度 ε")
    parser.add_argument("--n_mc",   type=int,   default=500,   help="モンテカルロサンプル数")
    parser.add_argument("--seed",   type=int,   default=42,    help="乱数シード")
    parser.add_argument("--mode",   type=str,   default="all",
                        choices=["all", "trajectory", "phase", "sensitivity", "montecarlo", "friction"],
                        help="実行モード")
    parser.add_argument("--save",   action="store_true",       help="図をファイルに保存")
    args = parser.parse_args()

    ic     = InitialCondition(args.v0, args.omega0, args.theta0)
    cyl    = CylinderParams()
    ground = GroundParams(args.mu, args.sigma)
    cfg    = SimConfig()

    def sp(name):  # save path helper
        return f"molkky_{name}.png" if args.save else None

    print(f"\n初期条件: v₀={args.v0} m/s, ω₀={args.omega0} rad/s, θ₀={args.theta0}°")
    print(f"摩擦係数: μ̄={args.mu}, σ={args.sigma}")
    print(f"摂動強度: ε={args.eps}")

    # 感度解析（trajectory/phase/sensitivity は共通で使う）
    run_sens = args.mode in ("all", "trajectory", "phase", "sensitivity")
    run_mc   = args.mode in ("all", "montecarlo")
    run_fric = args.mode in ("all", "friction")

    sens = None
    mc   = None

    if run_sens:
        print("\n感度解析を実行中...")
        sens = sensitivity_analysis(ic, cyl, ground, cfg, epsilon=args.eps, seed=args.seed)

    if run_mc:
        print(f"モンテカルロ解析を実行中 (N={args.n_mc})...")
        mc = monte_carlo(ic, cyl, ground, cfg, n_samples=args.n_mc,
                         epsilon=args.eps, rng_seed=args.seed)

    if run_fric:
        print("摩擦フィールドを描画中...")
        plot_friction_field(ground, x_max=8.0, seed=args.seed, save_path=sp("friction"))

    if sens:
        print_summary(sens, mc)
        if args.mode in ("all", "trajectory"):
            plot_trajectories(sens, save_path=sp("trajectory"))
        if args.mode in ("all", "phase"):
            plot_phase_space(sens, save_path=sp("phase"))
        if args.mode in ("all", "sensitivity"):
            plot_sensitivity(sens, save_path=sp("sensitivity"))

    if mc:
        plot_monte_carlo(mc, ic, save_path=sp("montecarlo"))


if __name__ == "__main__":
    main()
