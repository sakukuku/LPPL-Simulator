import streamlit as st
import numpy as np
import pandas as pd
import pywt
from scipy import optimize
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import os
import datetime
import warnings

# 忽略 scipy optimize 求解时的警告
warnings.filterwarnings('ignore', category=optimize.OptimizeWarning)

# ====================================================================
# 0. 参数区：全局常量定义
# ====================================================================

DELTA = 0.02  # δ = 2%，取 top 2% DD 作为 outliers
WAVELET = 'db4'  # Daubechies 4
WAVELET_LEVEL = 10  # n = 10
PEAK_ORDER = 48  # 在 d10 上找峰时的窗口
MIN_REGIME_LEN = 200  # 每个泡沫区间至少 X 点
BETA_BOUNDS = (0.01, 0.99)
OMEGA_BOUNDS = (2.0, 20.0)
TACC_HOURS = 48  # |t_c - t_DD| <= 24 小时


# ====================================================================
# 1. 数据读取 & 预处理
# ====================================================================

def load_price_csv(path: str, start_dt: datetime.datetime, end_dt: datetime.datetime) -> pd.DataFrame:
    """
    加载数据并根据起始和结束时间进行过滤。
    """
    if not os.path.exists(path):
        st.error(f"数据文件未找到: {path}。请确保文件在正确路径下。")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    df = df[['time', 'close']].dropna()
    df['close'] = df['close'].astype(float)

    # 确保时间对象都具有 UTC 时区信息
    start_dt_utc = start_dt.replace(tzinfo=datetime.timezone.utc)
    end_dt_utc = end_dt.replace(tzinfo=datetime.timezone.utc)

    df = df[(df['time'] >= start_dt_utc) & (df['time'] <= end_dt_utc)]
    df = df.reset_index(drop=True)

    if len(df) < 100:
        st.warning(f"所选时间范围内的数据点太少 ({len(df)}点)，请扩大范围。")
        return pd.DataFrame()

    return df


# ====================================================================
# 2. ε-drawdown 相关函数
# ====================================================================

def epsilon_from_vol(log_returns: np.ndarray) -> float:
    """ε = 对数收益率的标准差"""
    return float(np.std(log_returns))


def find_drawdowns(prices: np.ndarray, epsilon: float):
    """近似实现 Johansen & Sornette 的 ε-drawdown"""
    n = len(prices)
    i = 0
    drawdowns = []

    while i < n - 1:
        peak = i
        while peak + 1 < n and prices[peak + 1] >= prices[peak]:
            peak += 1
        if peak >= n - 1:
            break

        trough = peak + 1
        acc_drawup = 0.0
        while trough + 1 < n:
            step = np.log(prices[trough + 1]) - np.log(prices[trough])
            if step > 0:
                acc_drawup += step
                if acc_drawup >= epsilon:
                    break
            else:
                acc_drawup = 0.0
            trough += 1

        if trough >= n:
            break

        DD = np.log(prices[peak] / prices[trough])
        if DD > 0:
            drawdowns.append((peak, trough, DD))

        i = trough + 1

    return drawdowns


def fit_exponential_law(DD_values, delta=DELTA):
    """对 DD 拟合指数律 N(x)=N0*exp(-x/DDc)"""
    DD = np.array(DD_values, dtype=float)
    DD = DD[DD > 0]
    if len(DD) < 10:
        raise RuntimeError("有效 drawdown 数量太少，无法拟合指数律")

    order_desc = np.argsort(DD)[::-1]
    k_out = max(1, int(len(DD) * delta))
    outlier_idx = order_desc[:k_out]

    mask = np.ones(len(DD), dtype=bool)
    mask[outlier_idx] = False
    x = np.sort(DD[mask])

    uniq = np.unique(x)
    N_x = np.array([(x >= u).sum() for u in uniq], dtype=float)

    y = np.log(N_x + 1e-12)
    coeffs = np.polyfit(uniq, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    DDc = -1.0 / slope
    N0 = np.exp(intercept)

    return N0, DDc, outlier_idx, DD


# ====================================================================
# 3. 小波与 Regime 划分
# ====================================================================

def wavelet_regimes(log_price: np.ndarray,
                    wavelet_name: str = WAVELET,
                    level: int = WAVELET_LEVEL,
                    min_len: int = MIN_REGIME_LEN,
                    peak_order: int = PEAK_ORDER):
    """使用 DWT 和 detail 系数重构来划分 regime"""
    wave = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(log_price), wave.dec_len)
    if level > max_level:
        raise ValueError("数据长度不足以支持 level={} 的 {} 分解".format(
            level, wavelet_name))

    coeffs = pywt.wavedec(log_price, wave, level=level, mode='symmetric')

    arrs = [None] * (level + 1)
    arrs[0] = np.zeros_like(coeffs[0])
    for i in range(1, level):
        arrs[i] = np.zeros_like(coeffs[i])
    arrs[level] = coeffs[level]

    d_rec = pywt.waverec(arrs, wave, mode='symmetric')
    d_rec = d_rec[:len(log_price)]

    idx_peaks = argrelextrema(d_rec, np.greater_equal, order=peak_order)[0]

    cuts = [0] + sorted(idx_peaks.tolist()) + [len(log_price) - 1]

    regimes = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        length = b - a + 1
        if length > 1:
            regimes.append((a, b))

    return d_rec, regimes


# ====================================================================
# 4. LPPL 拟合与 TD9 信号
# ====================================================================

@dataclass
class LPPLParams:
    omega: float
    beta: float
    tc: float
    A: float
    B: float
    C1: float
    C2: float


def lppl_design_matrix(t: np.ndarray, omega: float, beta: float, tc: float) -> np.ndarray:
    tau = np.maximum(tc - t, 1e-9)
    f = tau ** beta
    X = np.column_stack([
        np.ones_like(t),
        f,
        f * np.cos(omega * np.log(tau)),
        f * np.sin(omega * np.log(tau)),
    ])
    return X


def fit_lppl_linearized(t: np.ndarray,
                        lnP: np.ndarray,
                        beta_bounds=BETA_BOUNDS,
                        omega_bounds=OMEGA_BOUNDS,
                        tc_bounds=None,
                        n_starts=20,
                        random_seed=42) -> Optional[LPPLParams]:
    """Filimonov & Sornette 的线性化 LPPL 校准方法"""
    rng = np.random.default_rng(random_seed)

    if tc_bounds is None:
        tc_bounds = (t[-1] + 1.0, t[-1] + 24.0 * 7)

    best = None
    best_sse = np.inf

    def residuals(theta):
        omega, beta, tc = theta
        X = lppl_design_matrix(t, omega, beta, tc)
        coeffs, _, _, _ = np.linalg.lstsq(X, lnP, rcond=None)
        fit = X @ coeffs
        return fit - lnP

    bounds = (
        [omega_bounds[0], beta_bounds[0], tc_bounds[0]],
        [omega_bounds[1], beta_bounds[1], tc_bounds[1]],
    )

    for _ in range(n_starts):
        x0 = np.array([
            rng.uniform(*omega_bounds),
            rng.uniform(*beta_bounds),
            rng.uniform(*tc_bounds),
        ])
        try:
            res = optimize.least_squares(
                residuals,
                x0=x0,
                bounds=bounds,
                max_nfev=20000,
            )
            if not res.success:
                continue
            omega, beta, tc = res.x
            X = lppl_design_matrix(t, omega, beta, tc)
            coeffs, _, _, _ = np.linalg.lstsq(X, lnP, rcond=None)
            fit = X @ coeffs
            sse = float(np.sum((fit - lnP) ** 2))
            if sse < best_sse and np.isfinite(sse):
                best_sse = sse
                A, B, C1, C2 = coeffs
                best = LPPLParams(
                    omega=float(omega),
                    beta=float(beta),
                    tc=float(tc),
                    A=float(A),
                    B=float(B),
                    C1=float(C1),
                    C2=float(C2),
                )
        except Exception:
            continue

    return best


def compute_td9_signals(close: np.ndarray):
    """简化版神奇九转（TD Sequential Setup）"""
    n = len(close)
    buy_setup = np.zeros(n, dtype=bool)
    sell_setup = np.zeros(n, dtype=bool)
    buy_count = 0
    sell_count = 0

    for i in range(4, n):
        if close[i] < close[i - 4]:
            buy_count += 1
            sell_count = 0
        elif close[i] > close[i - 4]:
            sell_count += 1
            buy_count = 0
        else:
            buy_count = 0
            sell_count = 0

        if buy_count == 9:
            buy_setup[i] = True
        if sell_count == 9:
            sell_setup[i] = True

    return buy_setup, sell_setup


def plot_all_bubbles(df, accepted_bubbles, fig, ax):
    """
    整体价格 + 泡沫区间 + 神奇九转 对照图
    """
    if df.empty: return

    # 全局价格曲线
    ax.plot(df['time'], df['close'], label='BTC price', linewidth=1.0)

    # ========== 神奇九转（TD9）信号 ==========
    close = df['close'].values
    buy_setup, sell_setup = compute_td9_signals(close)

    if buy_setup.any():
        ax.scatter(
            df['time'][buy_setup],
            df['close'][buy_setup],
            marker='v',
            s=40,
            color='green',
            label='TD9 buy'
        )

    if sell_setup.any():
        ax.scatter(
            df['time'][sell_setup],
            df['close'][sell_setup],
            marker='^',
            s=40,
            color='magenta',
            label='TD9 sell'
        )

    # ========== LPPL 泡沫区间与崩盘时间 ==========
    for idx, (ridx, a, b, params, tDD) in enumerate(accepted_bubbles):
        start_time = df['time'].iloc[a]
        end_time = df['time'].iloc[b]
        ax.axvspan(start_time, end_time, alpha=0.12, color='skyblue', label='bubble regime' if idx == 0 else None)

        # t_DD
        tdd_idx = int(tDD)
        tdd_idx = max(0, min(len(df) - 1, tdd_idx))
        tdd_time = df['time'].iloc[tdd_idx]
        ax.axvline(
            tdd_time,
            color='grey',
            linestyle=':',
            linewidth=0.8,
            label='$t_{DD}$' if idx == 0 else None
        )

        # t_c（LPPL 预测崩盘时间）
        tc_idx = int(round(params.tc))
        tc_idx = max(0, min(len(df) - 1, tc_idx))
        tc_time = df['time'].iloc[tc_idx]
        ax.axvline(
            tc_time,
            color='red',
            linestyle='--',
            linewidth=0.8,
            label='$t_c$' if idx == 0 else None
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("BTC Price with LPPL Bubbles vs TD Sequential")

    # --- 修复后的图例去重逻辑 ---
    handles, labels = ax.get_legend_handles_labels()

    # 使用字典存储唯一的 (label: handle) 对
    unique_items = {}
    for h, l in zip(handles, labels):
        if l not in unique_items:
            unique_items[l] = h

    # 将字典转换回 (handle, label) 对的列表
    unique = [(h, l) for l, h in unique_items.items()]

    # 传递解包后的唯一句柄和标签给 ax.legend
    if unique:
        ax.legend(*zip(*unique), loc='upper left')

    fig.tight_layout()


# ====================================================================
# 5. Streamlit 核心分析逻辑
# ====================================================================

@st.cache_data
def cached_load_data(path, start_dt, end_dt):
    """包装 load_price_csv 实现 Streamlit 缓存"""
    return load_price_csv(path, start_dt, end_dt)


def run_analysis(df, t_idx, params):
    """
    运行完整的 LPPL 泡沫识别和 TD9 分析流程。
    """

    st.header("📈 分析结果")

    # 1. 预处理
    df['ln_close'] = np.log(df['close'])
    df['log_ret'] = df['ln_close'].diff()
    lnP = df['ln_close'].values
    prices = df['close'].values

    # 2. ε-drawdown
    epsilon = epsilon_from_vol(df['log_ret'].dropna().values)
    st.info(f"数据总点数: {len(df)} | $\\epsilon$ (基于波动性) = **{epsilon:.5f}**")

    dd_records = find_drawdowns(prices, epsilon)
    if len(dd_records) < 10:
        st.warning("drawdown 事件太少（<10），无法进行指数律拟合。")
        return

    DD_values = [r[2] for r in dd_records]

    try:
        N0, DDc, outlier_idx, DD_filtered = fit_exponential_law(DD_values, delta=params['DELTA'])
    except RuntimeError as e:
        st.error(f"指数律拟合失败: {e}")
        return

    st.subheader("1. 指数律拟合结果 (确定 $t_{DD}$)")
    st.markdown(
        f"$N_0 \\approx **{N0:.2f}**, DD_c \\approx **{DDc:.4f}**$ (Top {params['DELTA'] * 100:.1f}% DDs considered outliers)")

    # 异常 drawdown 的 trough 时间索引作为 t_DD
    outlier_records = [dd_records[int(i)] for i in outlier_idx]
    tDD_list = sorted([r[1] for r in outlier_records])

    # 3. db4 小波分解，选 d10 划分 regimes
    d_rec, regimes = wavelet_regimes(
        lnP,
        wavelet_name=params['WAVELET'],
        level=params['WAVELET_LEVEL'],
        min_len=None,
        peak_order=params['PEAK_ORDER']
    )
    st.subheader("2. 小波分解 Regime 划分")
    st.markdown(f"找到 **{len(regimes)}** 个潜在泡沫区间 (窗口: {params['PEAK_ORDER']})")

    # 4. 对每个 regime 拟合 LPPL，并用 |t_c - t_DD| 过滤
    accepted_bubbles = []

    with st.expander("点击查看 LPPL 拟合与 $t_{DD}$ 校验过程", expanded=False):
        progress_bar = st.progress(0, text="LPPL 拟合中...")
        info_placeholder = st.empty()

        for ridx, (a, b) in enumerate(regimes):

            regime_len = b - a + 1
            if regime_len < params['MIN_REGIME_LEN']:
                progress_bar.progress((ridx + 1) / len(regimes),
                                      text=f"Skipping regime {ridx} (len {regime_len} < {params['MIN_REGIME_LEN']})...")
                continue

            t_reg = t_idx[a:b + 1]
            lnP_reg = lnP[a:b + 1]

            tc_bounds = (t_reg[-1] + 1.0, t_reg[-1] + 24.0 * 7)

            params_lppl = fit_lppl_linearized(
                t_reg,
                lnP_reg,
                beta_bounds=params['BETA_BOUNDS'],
                omega_bounds=params['OMEGA_BOUNDS'],
                tc_bounds=tc_bounds,
                n_starts=params['N_STARTS'],
            )

            progress_bar.progress((ridx + 1) / len(regimes), text=f"拟合 Regime {ridx} (len {regime_len})...")

            if params_lppl is None:
                info_placeholder.text(f"Regime {ridx}: LPPL 拟合失败。")
                continue

            tc = params_lppl.tc
            if not tDD_list:
                info_placeholder.text("没有有效的 t_DD 列表用于校验。")
                continue

            nearest_tDD = min(tDD_list, key=lambda x: abs(tc - x))
            t_diff = abs(tc - nearest_tDD)

            tc_idx = int(round(tc))
            tc_idx = max(0, min(len(df) - 1, tc_idx))

            if t_diff <= params['TACC_HOURS']:
                accepted_bubbles.append((ridx, a, b, params_lppl, nearest_tDD))

                info_placeholder.success(
                    f"[泡沫确认] Regime {ridx}: t_c ≈ {tc:.2f} ({df['time'].iloc[tc_idx].strftime('%Y-%m-%d %H:%M')}), "
                    f"最近 t_DD={nearest_tDD:.2f}。差值: {t_diff:.2f}h"
                )
            else:
                info_placeholder.text(
                    f"Regime {ridx}: t_c ≈ {tc:.2f}, 最近 t_DD={nearest_tDD:.2f}。 "
                    f"差值 {t_diff:.2f}h，不满足 $\\leq {params['TACC_HOURS']}h$ 的条件。"
                )

        progress_bar.empty()
        info_placeholder.empty()

    st.success(f"总共确认 **{len(accepted_bubbles)}** 个泡沫（通过 t_DD 校验）。")

    # 绘制图形
    st.subheader("3. 价格与泡沫识别结果图")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_all_bubbles(df, accepted_bubbles, fig, ax)
    st.pyplot(fig)

    # 输出泡沫概览
    st.subheader("4. 确认泡沫概览")

    bubble_summary_data = []
    n = len(df)
    for idx, (ridx, a, b, params_lppl, tDD) in enumerate(accepted_bubbles, 1):
        t_start = df['time'].iloc[a]
        t_end = df['time'].iloc[b]
        p_start = float(df['close'].iloc[a])
        p_end = float(df['close'].iloc[b])
        seg = df.iloc[a:b + 1]
        p_max = float(seg['close'].max())

        rise_start_to_max = (p_max / p_start - 1.0) if p_start > 0 else float('nan')
        fall_max_to_end = (p_end / p_max - 1.0) if p_max > 0 else float('nan')

        tc_idx = int(round(params_lppl.tc))
        tc_idx = max(0, min(n - 1, tc_idx))
        t_c_time = df['time'].iloc[tc_idx].strftime('%Y-%m-%d %H:%M')

        tdd_idx = int(tDD)
        tdd_idx = max(0, min(n - 1, tdd_idx))
        t_DD_time = df['time'].iloc[tdd_idx].strftime('%Y-%m-%d %H:%M')

        bubble_summary_data.append({
            'ID': idx,
            'Regime': ridx,
            '起始时间': t_start.strftime('%Y-%m-%d %H:%M'),
            '结束时间': t_end.strftime('%Y-%m-%d %H:%M'),
            'P_start': f"{p_start:.2f}",
            'P_max': f"{p_max:.2f}",
            'P_end': f"{p_end:.2f}",
            '涨幅(起点→峰值)': f"{rise_start_to_max * 100:.2f}%",
            '跌幅(峰值→结束)': f"{fall_max_to_end * 100:.2f}%",
            't_c (LPPL)': t_c_time,
            't_DD (观测)': t_DD_time,
        })

    if bubble_summary_data:
        st.dataframe(pd.DataFrame(bubble_summary_data))

    # 打印 TD9 信号
    st.subheader("5. 神奇九转 (TD9) 信号")
    close = prices
    buy_setup, sell_setup = compute_td9_signals(close)

    bubble_map = {}
    for idx, (ridx, a, b, params_lppl, tDD) in enumerate(accepted_bubbles, 1):
        for i in range(a, b + 1):
            bubble_map[i] = idx

    td9_data = []
    buy_indices = np.where(buy_setup)[0]
    for i in buy_indices:
        bubble_id = bubble_map.get(i, '无')
        td9_data.append({'类型': 'BUY 💚', '时间': df['time'].iloc[i].strftime('%Y-%m-%d %H:%M'),
                         '价格': f"{df['close'].iloc[i]:.2f}", 'Index': i, '所在泡沫ID': bubble_id})

    sell_indices = np.where(sell_setup)[0]
    for i in sell_indices:
        bubble_id = bubble_map.get(i, '无')
        td9_data.append({'类型': 'SELL 💔', '时间': df['time'].iloc[i].strftime('%Y-%m-%d %H:%M'),
                         '价格': f"{df['close'].iloc[i]:.2f}", 'Index': i, '所在泡沫ID': bubble_id})

    if td9_data:
        st.dataframe(pd.DataFrame(td9_data))
    else:
        st.info("未找到 TD9 买入或卖出信号。")


def st_main():
    """Streamlit 界面布局"""
    st.title("LPPL 泡沫识别与 TD9 信号分析")
    st.markdown("---")

    # 侧边栏：参数输入区
    st.sidebar.header("参数配置区")

    # --- 数据源选择 (新增 K 线类型) ---
    st.sidebar.subheader("数据源选择 (Data Range)")

    # K 线类型选择
    timeframe = st.sidebar.selectbox(
        "选择 K 线类型 (Timeframe)",
        ('30m', '1h', '1d'),
        index=0,
        help="请确保运行目录下有对应的文件，例如 btc_30m.csv, btc_1h.csv, btc_1d.csv"
    )

    # 根据选择构建文件名
    DATA_PATH = f"btc_{timeframe}.csv"
    st.sidebar.caption(f"当前尝试加载文件: **{DATA_PATH}**")

    default_end_date = datetime.date.today()
    default_start_date = default_end_date - datetime.timedelta(days=365)

    # 起始时间选择
    start_date = st.sidebar.date_input("起始日期", default_start_date)
    start_time = st.sidebar.time_input("起始时间 (UTC)", datetime.time(0, 0))

    # 结束时间选择
    end_date = st.sidebar.date_input("结束日期", default_end_date)
    end_time = st.sidebar.time_input("结束时间 (UTC)", datetime.time(23, 59))

    # 组合日期和时间
    start_dt = datetime.datetime.combine(start_date, start_time)
    end_dt = datetime.datetime.combine(end_date, end_time)

    if start_dt >= end_dt:
        st.sidebar.error("起始时间不能晚于或等于结束时间！")
        return

    # --- Drawdown & Filtering ---
    st.sidebar.subheader("1. Drawdown & 过滤")
    delta = st.sidebar.slider(
        "DELTA (δ): 异常回撤占比 (Top X%)",
        min_value=0.005, max_value=0.1, value=DELTA, step=0.005, format="%.3f"
    )

    # --- Wavelet & Regime ---
    st.sidebar.subheader("2. 小波与区间划分")
    wavelet = st.sidebar.selectbox("Wavelet (db/sym)", ('db4', 'db8', 'sym4'), index=0)
    wavelet_level = st.sidebar.slider("Wavelet Level (n)", 5, 12, WAVELET_LEVEL, 1)
    peak_order = st.sidebar.slider("Peak Order (窗口大小)", 10, 100, PEAK_ORDER, 1)
    min_regime_len = st.sidebar.slider("最小区间长度 (点数)", 50, 500, MIN_REGIME_LEN, 10)

    # --- LPPL Bounds & Validation ---
    st.sidebar.subheader("3. LPPL 拟合与校验")
    beta_min = st.sidebar.slider("Beta Min", 0.01, 0.5, BETA_BOUNDS[0], 0.01)
    beta_max = st.sidebar.slider("Beta Max", 0.5, 1.0, BETA_BOUNDS[1], 0.01)
    omega_min = st.sidebar.slider("Omega Min", 0.5, 5.0, OMEGA_BOUNDS[0], 0.1)
    omega_max = st.sidebar.slider("Omega Max", 5.0, 30.0, OMEGA_BOUNDS[1], 1.0)
    tacc_hours = st.sidebar.slider("t_c - t_DD 最大小时差", 12, 168, TACC_HOURS, 12)
    n_starts = st.sidebar.slider("拟合随机起始次数", 5, 100, 20, 5)

    # 将所有参数打包
    analysis_params = {
        'DELTA': delta,
        'WAVELET': wavelet,
        'WAVELET_LEVEL': wavelet_level,
        'PEAK_ORDER': peak_order,
        'MIN_REGIME_LEN': min_regime_len,
        'BETA_BOUNDS': (beta_min, beta_max),
        'OMEGA_BOUNDS': (omega_min, omega_max),
        'TACC_HOURS': tacc_hours,
        'N_STARTS': n_starts
    }

    # --- 数据加载 ---
    # 缓存函数依赖 DATA_PATH (即 K 线类型)
    df = cached_load_data(DATA_PATH, start_dt, end_dt)

    if not df.empty:
        t_idx = np.arange(len(df), dtype=float)

        # 运行分析
        run_analysis(df, t_idx, analysis_params)
    else:
        # 如果 df.empty，load_price_csv 已经打印了错误/警告信息
        pass


if __name__ == "__main__":
    st_main()