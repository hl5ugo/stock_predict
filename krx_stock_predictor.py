#!/usr/bin/env python3
"""
KRX 주가 예측 스크립트
=======================
데이터 수집: pykrx (KRX 공식 데이터, 로그인/인증키 불필요)
ML 모델   : RandomForest + GradientBoosting + LinearRegression 앙상블

설치:
  pip install pykrx pandas numpy matplotlib scikit-learn

사용법:
  python krx_stock_predictor.py 005930       ← 삼성전자, 5 영업일 후 예측
  python krx_stock_predictor.py 000660 10    ← SK하이닉스, 10 영업일 후 예측
"""

import argparse
import math
import re
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc

from pykrx import stock as krx

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 전역 경고 억제 제거 → sklearn 관련 경고만 선택적으로 억제
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ─────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────
def set_korean_font() -> None:
    candidates = ["NanumGothic", "NanumBarunGothic", "AppleGothic",
                  "Malgun Gothic", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rc("font", family=name)
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "default"

set_korean_font()


# ─────────────────────────────────────────────
# 종목코드 유효성 검증
# ─────────────────────────────────────────────
def validate_ticker(ticker: str) -> str:
    """
    종목코드가 숫자 6자리인지 검증합니다.
    - 숫자가 아닌 문자 포함 → 오류
    - 6자리 미만 → zfill(6) 패딩
    - 6자리 초과 → 오류
    """
    ticker = ticker.strip()
    if not re.fullmatch(r"\d{1,6}", ticker):
        print(f"❌ 잘못된 종목코드: '{ticker}'")
        print("   종목코드는 숫자 1~6자리여야 합니다. (예: 005930, 660)")
        sys.exit(1)
    ticker = ticker.zfill(6)
    return ticker


# ─────────────────────────────────────────────
# KRX 데이터 수집 (pykrx)
# ─────────────────────────────────────────────

# 시장 판별에 사용할 기준일 (최근 영업일 근사치로 오늘 사용)
_MARKETS = ["KOSPI", "KOSDAQ", "KONEX"]


def detect_market(ticker: str) -> str | None:
    """
    종목코드가 속한 시장(KOSPI / KOSDAQ / KONEX)을 자동으로 판별합니다.
    pykrx의 get_market_ticker_list로 각 시장 종목 목록을 조회해 비교합니다.
    판별 실패 시 None을 반환합니다 (호출부에서 사용자 입력을 받아야 합니다).
    """
    today = datetime.today().strftime("%Y%m%d")
    for market in _MARKETS:
        try:
            tickers = krx.get_market_ticker_list(today, market=market)
            if ticker in tickers:
                return market
        except Exception:
            continue
    return None


def get_stock_name(ticker: str) -> str:
    """종목명 조회. 실패 시 종목코드를 그대로 반환합니다."""
    try:
        name = krx.get_market_ticker_name(ticker)
        return name if name else ticker
    except Exception:
        return ticker


def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    pykrx로 OHLCV 데이터를 조회합니다.
    start / end : 'YYYYMMDD'
    반환 컬럼   : Date(datetime), Open, High, Low, Close, Volume
    """
    try:
        df = krx.get_market_ohlcv_by_date(start, end, ticker)
        if df is None or df.empty:
            return pd.DataFrame()

        col_map = {
            "시가": "Open", "고가": "High", "저가": "Low",
            "종가": "Close", "거래량": "Volume",
            "거래대금": "TradingValue", "등락률": "ChangeRate",
        }
        df = df.rename(columns=col_map)
        df.index.name = "Date"
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"❌ [OHLCV] {type(e).__name__}: {e}")
        return pd.DataFrame()


def get_fundamental(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    pykrx로 PER / PBR / EPS / BPS / DIV 데이터를 일자별로 조회합니다.
    start / end : 'YYYYMMDD'

    주의: KRX의 펀더멘털 데이터는 연간 사업보고서 제출 후
    약 1~2개월 검토 기간을 거쳐 입력됩니다.
    따라서 최신 실적 대비 다소 지연될 수 있습니다.
    """
    try:
        df = krx.get_market_fundamental(start, end, ticker)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = "Date"
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        # PER=0 은 적자 또는 미집계 → NaN 처리 (0으로 남기면 피처 왜곡)
        df["PER"] = df["PER"].replace(0, float("nan"))
        df["PBR"] = df["PBR"].replace(0, float("nan"))
        df["EPS"] = df["EPS"].replace(0, float("nan"))
        df["BPS"] = df["BPS"].replace(0, float("nan"))
        return df[["Date", "PER", "PBR", "EPS", "BPS", "DIV"]]
    except Exception as e:
        print(f"⚠️  [펀더멘털] {type(e).__name__}: {e}  → 피처에서 제외됩니다.")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 기술적 지표
# ─────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 기술적 지표 컬럼을 추가해 반환합니다.
    원본 DataFrame은 수정하지 않습니다 (copy 사용).
    """
    df = df.copy()
    c = df["Close"]

    # 이동평균 + 비율
    for w in [5, 10, 20, 60]:
        df[f"MA{w}"]       = c.rolling(w).mean()
        df[f"MA{w}_ratio"] = c / df[f"MA{w}"]

    # 볼린저 밴드
    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_mid"]   = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (bb_mid + 1e-9)
    df["BB_pos"]   = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # 스토캐스틱
    low14             = df["Low"].rolling(14).min()
    high14            = df["High"].rolling(14).max()
    df["Stoch_K"]     = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"]     = df["Stoch_K"].rolling(3).mean()

    # 거래량 비율
    df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / (df["Vol_MA20"] + 1e-9)

    # 수익률 — Return_1d를 먼저 명시적으로 계산 (Volatility가 의존)
    df["Return_1d"] = c.pct_change(1)
    for lag in [2, 3, 5, 10]:
        df[f"Return_{lag}d"] = c.pct_change(lag)

    # 변동성 (Return_1d 계산 후에 참조)
    df["Volatility_5"]  = df["Return_1d"].rolling(5).std()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # 고저 스프레드
    df["HL_spread"] = (df["High"] - df["Low"]) / (c + 1e-9)

    return df


def prepare_features(df: pd.DataFrame, target_days: int = 5) -> tuple:
    """
    기술적 지표를 피처로 변환합니다.
    target_days: 예측할 영업일 수 (달력일 아님)
    """
    df = add_technical_indicators(df)
    df["Target"] = df["Close"].shift(-target_days)

    # PER, PBR, EPS, BPS, DIV 는 피처로 포함 (skip 하지 않음)
    skip = {"Date", "Close", "Open", "High", "Low",
            "Volume", "TradingValue", "ChangeRate", "Target"}
    feature_cols = [col for col in df.columns
                    if col not in skip and df[col].dtype != "object"]

    df_clean = df.dropna(subset=feature_cols + ["Target"]).copy()
    return (
        df_clean[feature_cols].values,
        df_clean["Target"].values,
        df_clean["Date"].values,
        feature_cols,
        df_clean,
    )


# ─────────────────────────────────────────────
# ML 예측 모델
# ─────────────────────────────────────────────
class StockPredictor:
    """
    RandomForest / GradientBoosting / LinearRegression 앙상블 예측기.

    [수정 사항]
    - 데이터 누수 방지: MinMaxScaler를 Pipeline에 통합해
      각 CV 폴드의 train 구간에서만 fit합니다.
    - 트리 기반 모델(RF, GB)은 스케일링 불필요 → Pipeline 미적용
    - LinearRegression만 Pipeline(scaler + model) 사용
    - R² 음수 모델은 앙상블에서 제외 (평균값보다 나쁜 모델)
    - 예측값 음수 방지: max(price, 0) 처리
    """

    def __init__(self):
        # 트리 모델은 스케일링 없이, LinearRegression만 Pipeline으로 감쌈
        self.models: dict = {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=8,
                min_samples_split=5, random_state=42, n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, random_state=42,
            ),
            "LinearRegression": Pipeline([
                ("scaler", MinMaxScaler()),
                ("model",  LinearRegression()),
            ]),
        }
        self.weights: dict = {}
        self.metrics: dict = {}

    def train_evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        TimeSeriesSplit 교차검증으로 성능을 측정하고,
        MAE 역수 기반 가중치를 계산합니다.
        R² < 0 인 모델은 앙상블 가중치를 0으로 설정합니다.
        """
        tscv      = TimeSeriesSplit(n_splits=5)
        cv_preds  = {n: np.zeros(len(y)) for n in self.models}
        cv_counts = np.zeros(len(y))

        for tr, va in tscv.split(X):
            cv_counts[va] = 1
            for name, model in self.models.items():
                model.fit(X[tr], y[tr])
                cv_preds[name][va] = model.predict(X[va])

        # 전체 데이터로 최종 재학습
        for model in self.models.values():
            model.fit(X, y)

        # 성능 평가 및 가중치 계산
        mask    = cv_counts > 0
        inv_mae = {}
        for name, preds in cv_preds.items():
            mae  = mean_absolute_error(y[mask], preds[mask])
            rmse = np.sqrt(mean_squared_error(y[mask], preds[mask]))
            r2   = r2_score(y[mask], preds[mask])
            self.metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

            # R² < 0 이면 해당 모델은 앙상블에서 제외
            if r2 >= 0:
                inv_mae[name] = 1.0 / (mae + 1e-9)
            else:
                inv_mae[name] = 0.0
                print(f"   ⚠️  {name}: R²={r2:.4f} (음수) → 앙상블에서 제외")

        total = sum(inv_mae.values())
        if total == 0:
            # 모든 모델이 R² < 0 이면 균등 가중치로 fallback
            print("   ⚠️  모든 모델 R² 음수 → 균등 가중치 적용")
            self.weights = {k: 1 / len(self.models) for k in self.models}
        else:
            self.weights = {k: v / total for k, v in inv_mae.items()}

    def predict(self, X_last: np.ndarray) -> dict:
        """
        각 모델의 예측값과 앙상블 예측값을 반환합니다.
        예측값은 0 이상으로 클리핑합니다 (음수 주가 방지).
        """
        preds    = {}
        ensemble = 0.0
        for name, model in self.models.items():
            raw       = model.predict(X_last.reshape(1, -1))[0]
            price     = max(raw, 0.0)   # 음수 주가 방지
            preds[name] = price
            ensemble   += self.weights[name] * price
        preds["Ensemble"] = max(ensemble, 0.0)
        return preds


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def plot_results(
    df_full: pd.DataFrame,
    ticker: str,
    stock_name: str,
    predictions: dict,
    target_days: int,
    path: str,
) -> None:
    # ── 라이트 테마 컬러 팔레트 (배경 투명)
    C_PRICE   = "#1565C0"   # 종가선  — 진파랑
    C_MA20    = "#F57C00"   # MA20   — 주황
    C_MA60    = "#C62828"   # MA60   — 진빨강
    C_UP      = "#C62828"   # 상승   — 진빨강
    C_DOWN    = "#00695C"   # 하락   — 진초록
    C_RSI     = "#6A1B9A"   # RSI    — 보라
    C_MACD    = "#1565C0"   # MACD   — 진파랑
    C_SIGNAL  = "#E65100"   # Signal — 진주황
    C_GRID    = "#E0E0E0"   # 그리드 — 연회색
    C_SPINE   = "#9E9E9E"   # 축 테두리
    C_TEXT    = "#212121"   # 텍스트 — 거의 검정

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 16),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    # 배경 투명 처리
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C_SPINE)

    recent       = df_full.tail(120).copy()
    dates, close = recent["Date"], recent["Close"]

    # ── 주가 차트
    ax = axes[0]
    ax.plot(dates, close, color=C_PRICE, linewidth=1.8, label="종가")
    ax.fill_between(dates, close, close.min(), alpha=0.07, color=C_PRICE)
    for w, color in [(20, C_MA20), (60, C_MA60)]:
        ax.plot(dates, close.rolling(w).mean(), "--", lw=1.2,
                color=color, alpha=0.85, label=f"MA{w}")

    ep   = predictions["Ensemble"]
    cpct = (ep / close.iloc[-1] - 1) * 100
    ac   = C_UP if cpct >= 0 else C_DOWN

    # 예측 화살표: x축 범위를 고려해 차트 안에 표시
    chart_span_days = (dates.iloc[-1] - dates.iloc[0]).days
    offset_days     = max(int(chart_span_days * 0.05), target_days * 2)
    fd              = dates.iloc[-1] + timedelta(days=offset_days)

    ax.annotate(
        f"예측({target_days} 영업일 후)\n{ep:,.0f}원\n({cpct:+.2f}%)",
        xy=(fd, ep), xytext=(fd, close.iloc[-1]),
        fontsize=9, color=ac, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ac, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor=ac, alpha=0.85),
    )
    ax.axhline(ep, color=ac, linestyle=":", lw=1.2, alpha=0.6)
    ax.set_title(
        f"{stock_name} ({ticker})  주가 예측 분석",
        color=C_TEXT, fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_ylabel("주가 (원)", color=C_TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax.legend(facecolor="white", edgecolor=C_SPINE,
              labelcolor=C_TEXT, fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── RSI
    di  = add_technical_indicators(df_full).tail(120)
    ax2 = axes[1]
    ax2.plot(di["Date"], di["RSI"], color=C_RSI, lw=1.4)
    ax2.axhline(70, color=C_UP,   linestyle="--", lw=0.9, alpha=0.7, label="과매수(70)")
    ax2.axhline(30, color=C_DOWN, linestyle="--", lw=0.9, alpha=0.7, label="과매도(30)")
    ax2.fill_between(di["Date"], di["RSI"], 50, alpha=0.1, color=C_RSI)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color=C_TEXT)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax2.legend(facecolor="white", edgecolor=C_SPINE,
               labelcolor=C_TEXT, fontsize=8, framealpha=0.85)

    # ── MACD
    ax3 = axes[2]
    ax3.plot(di["Date"], di["MACD"],        color=C_MACD,   lw=1.2, label="MACD")
    ax3.plot(di["Date"], di["MACD_signal"], color=C_SIGNAL, lw=1.2,
             linestyle="--", label="Signal")
    hv = di["MACD_hist"].values
    ax3.bar(di["Date"], hv,
            color=[C_UP if v >= 0 else C_DOWN for v in hv],
            alpha=0.5, width=1)
    ax3.axhline(0, color=C_SPINE, lw=0.8, alpha=0.6)
    ax3.set_ylabel("MACD", color=C_TEXT)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax3.legend(facecolor="white", edgecolor=C_SPINE,
               labelcolor=C_TEXT, fontsize=8, framealpha=0.85)

    for ax in axes:
        ax.grid(axis="y", color=C_GRID, lw=0.6, linestyle="--")
        ax.tick_params(axis="x", colors=C_TEXT)
        ax.tick_params(axis="y", colors=C_TEXT)

    plt.tight_layout(pad=2.0)
    # transparent=True 로 배경 완전 투명 저장
    plt.savefig(path, dpi=130, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"📊 차트 저장: {path}")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def run_prediction(
    ticker: str,
    target_days: int = 5,
    lookback_years: int = 3,
    market: str | None = None,
) -> dict | None:
    print("=" * 60)
    print("  KRX 주가 예측 시스템  (pykrx)")
    print("=" * 60)

    # 여유 있게 요청 (pykrx가 영업일만 반환하므로 달력일 기준으로 넉넉히 설정)
    end_dt    = datetime.today()
    start_dt  = end_dt - timedelta(days=int(365 * lookback_years * 1.1))
    start_str = start_dt.strftime("%Y%m%d")
    end_str   = end_dt.strftime("%Y%m%d")

    print(f"\n🔍 종목코드  : {ticker}")
    print(f"📅 요청 기간 : {start_str} ~ {end_str}  (공휴일 제외 영업일만 수집)")
    print(f"🎯 예측 기간 : {target_days} 영업일 후 (달력 기준 약 {int(target_days * 1.4)}일)")

    # 종목명 + 시장 판별
    print("\n📡 종목 정보 조회 중...")
    stock_name = get_stock_name(ticker)

    if market:
        # 파라미터로 명시적으로 지정된 경우 그대로 사용
        market = market.upper()
        print(f"   → {stock_name}  [{market}] (파라미터 지정)")
    else:
        # 자동 판별 시도
        detected = detect_market(ticker)
        if detected:
            market = detected
            print(f"   → {stock_name}  [{market}] (자동 판별)")
        else:
            # 자동 판별 실패 → 사용자에게 직접 입력 요청
            print(f"   → {stock_name}")
            print("   ⚠️  시장을 자동으로 판별하지 못했습니다.")
            while True:
                ans = input("   시장을 입력하세요 (KOSPI / KOSDAQ / KONEX): ").strip().upper()
                if ans in _MARKETS:
                    market = ans
                    break
                print(f"   ❌ '{ans}'은 유효하지 않습니다. KOSPI / KOSDAQ / KONEX 중 하나를 입력하세요.")
            print(f"   → [{market}] 로 진행합니다.")

    # OHLCV — pykrx는 종목코드만으로 KOSPI/KOSDAQ 구분 없이 조회 가능
    print("\n📡 OHLCV 데이터 수집 중...")
    df = get_ohlcv(ticker, start_str, end_str)
    if df.empty:
        print("❌ 데이터를 가져오지 못했습니다.")
        print("   확인: 종목코드 6자리 / 네트워크 연결")
        return None

    print(f"   → {len(df)}개 영업일 로드 완료")
    print(f"   최근 종가: {df['Close'].iloc[-1]:,.0f}원  "
          f"({df['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    # 펀더멘털 데이터 (PER / PBR / EPS / BPS / DIV) 수집 및 병합
    print("\n📡 펀더멘털 데이터 수집 중 (PER / PBR / EPS / BPS)...")
    df_fund = get_fundamental(ticker, start_str, end_str)
    if not df_fund.empty:
        df = pd.merge(df, df_fund, on="Date", how="left")
        # 결측값은 직전 영업일 값으로 전방 채움 (PER/PBR은 실적 발표 전까지 변경 없음)
        for col in ["PER", "PBR", "EPS", "BPS", "DIV"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        per_last = df["PER"].dropna().iloc[-1] if "PER" in df.columns and not df["PER"].dropna().empty else None
        pbr_last = df["PBR"].dropna().iloc[-1] if "PBR" in df.columns and not df["PBR"].dropna().empty else None
        per_str = f"{per_last:.2f}배" if per_last is not None else "N/A"
        pbr_str = f"{pbr_last:.2f}배" if pbr_last is not None else "N/A"
        print(f"   → 최근 PER: {per_str},  PBR: {pbr_str}")
    else:
        print("   ⚠️  펀더멘털 데이터를 가져오지 못했습니다. PER/PBR 없이 진행합니다.")

    # 피처 준비 — 지표 계산은 여기서 한 번만 수행
    print("\n⚙️  피처 엔지니어링 중...")
    X, y, dates, feature_cols, df_clean = prepare_features(df, target_days)
    print(f"   → 피처 수: {len(feature_cols)},  학습 샘플: {len(X)}")

    if len(X) < 100:
        print("⚠️  학습 데이터 부족 (최소 100일). lookback_years를 늘려보세요.")
        return None

    # 모델 학습
    print("\n🤖 모델 학습 중 (RandomForest / GradientBoosting / LinearRegression)...")
    predictor = StockPredictor()
    predictor.train_evaluate(X, y)

    print("\n📈 교차검증 성능:")
    print(f"   {'모델':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'가중치':>8}")
    print("   " + "─" * 58)
    for name, m in predictor.metrics.items():
        w   = predictor.weights.get(name, 0)
        r2_warn = " ⚠️" if m["R2"] < 0 else ""
        print(f"   {name:<22} {m['MAE']:>10,.0f} {m['RMSE']:>10,.0f} "
              f"{m['R2']:>8.4f}{r2_warn}  {w:>7.3f}")

    # 예측 — df_clean에 이미 지표가 계산되어 있으므로 재계산 불필요
    last_row = df_clean[feature_cols + ["Close"]].iloc[-1]
    preds    = predictor.predict(last_row[feature_cols].values)
    cur      = df["Close"].iloc[-1]

    print("\n" + "=" * 60)
    print(f"  📌 현재가: {cur:>12,.0f} 원")
    print("  " + "─" * 58)
    print(f"  {'모델':<22} {'예측가':>12}  {'등락':>12}")
    print("  " + "─" * 58)
    for name, price in preds.items():
        diff  = price - cur
        pct   = diff / cur * 100
        tag   = "★ " if name == "Ensemble" else "  "
        sign  = "▲" if diff >= 0 else "▼"
        label = "(상승 예상)" if diff >= 0 else "(하락 예상)"
        print(f"  {tag}{name:<22} {price:>12,.0f}원  "
              f"{sign}{abs(pct):5.2f}%  {label}")
    print("=" * 60)

    # 차트
    out_img = f"krx_prediction_{ticker}.png"
    plot_results(df, ticker, stock_name, preds, target_days, out_img)

    # ── 기술적 분석 요약 (df_clean 마지막 행 재활용)
    last  = df_clean.iloc[-1]
    ens_pct = (preds["Ensemble"] / cur - 1) * 100

    rsi_now    = last["RSI"]
    macd_now   = last["MACD"]
    macd_sig   = last["MACD_signal"]
    bb_pos     = last["BB_pos"]        # 0=하단, 0.5=중간, 1=상단
    ma20_ratio = last["MA20_ratio"]    # 1.0 기준 (1보다 크면 MA 위)
    ma60_ratio = last["MA60_ratio"]
    vol_ratio  = last["Vol_ratio"]     # 1.0 기준 (1보다 크면 평균 이상)

    print("\n" + "=" * 60)
    print("  💡 초보 투자자를 위한 기술적 분석 요약")
    print("=" * 60)

    # ── 1) AI 예측 결과
    print("\n  【 AI 예측 결과 】")
    pred_arrow = "📈" if ens_pct >= 0 else "📉"
    print(f"  {pred_arrow} {target_days} 영업일 후 예상 주가: {preds['Ensemble']:,.0f}원")
    print(f"     현재가 {cur:,.0f}원 대비  {ens_pct:+.2f}%")
    if ens_pct >= 5:
        pred_comment = "강한 상승이 예상됩니다."
    elif ens_pct >= 2:
        pred_comment = "소폭 상승이 예상됩니다."
    elif ens_pct >= -2:
        pred_comment = "큰 변동 없이 횡보할 것으로 보입니다."
    elif ens_pct >= -5:
        pred_comment = "소폭 하락이 예상됩니다."
    else:
        pred_comment = "하락 가능성이 높습니다."
    print(f"  → {pred_comment}")

    # ── 2) RSI — 과열/침체 여부
    print("\n  【 RSI (과열·침체 지표) 】")
    print(f"  현재 RSI: {rsi_now:.1f}")
    print( "  ※ 0~30: 침체 구간 (많이 팔린 상태) / 70~100: 과열 구간 (많이 오른 상태)")
    if rsi_now >= 80:
        rsi_comment = "매우 과열 상태입니다. 단기 하락 조정이 올 수 있으니 신규 매수는 신중하게 하세요."
    elif rsi_now >= 70:
        rsi_comment = "과열 구간입니다. 추격 매수보다는 보유 중이라면 수익 실현을 고려해볼 수 있습니다."
    elif rsi_now >= 50:
        rsi_comment = "보통 수준입니다. 아직 추가 상승 여력이 있을 수 있습니다."
    elif rsi_now >= 30:
        rsi_comment = "다소 낮은 수준입니다. 매수를 고려해볼 수 있는 구간입니다."
    else:
        rsi_comment = "매우 침체된 상태입니다. 단기 반등 가능성이 있어 분할 매수를 고려할 수 있습니다."
    print(f"  → {rsi_comment}")

    # ── 3) MACD — 추세 방향
    print("\n  【 MACD (추세 방향 지표) 】")
    print( "  ※ MACD가 Signal선보다 위에 있으면 상승 추세, 아래이면 하락 추세")
    # MACD 갭 판별: 종가의 0.3%를 "확실한" 갭 임계치로 사용
    # abs(macd_now) 기반 임계치는 macd_now≈0 일 때 항상 False가 되는 버그 있음
    macd_threshold = cur * 0.003
    macd_gap = macd_now - macd_sig
    if macd_gap > macd_threshold:
        macd_comment = "MACD가 Signal선을 확실히 상회 중입니다. 상승 추세가 유지되고 있습니다."
    elif macd_gap > 0:
        macd_comment = "MACD가 Signal선을 막 넘어선 상태입니다. 상승 전환 초기 신호일 수 있습니다."
    elif macd_gap > -macd_threshold:
        macd_comment = "MACD가 Signal선 아래로 막 내려온 상태입니다. 하락 전환 초기 신호일 수 있습니다."
    else:
        macd_comment = "MACD가 Signal선 아래에 있습니다. 하락 추세가 이어지고 있습니다."
    print(f"  → {macd_comment}")

    # ── 4) 볼린저 밴드 — 현재 가격 위치
    print("\n  【 볼린저 밴드 (가격 위치 지표) 】")
    print( "  ※ 주가가 밴드 상단에 가까울수록 고점, 하단에 가까울수록 저점에 위치")
    bb_pct = bb_pos * 100
    print(f"  현재 밴드 내 위치: {bb_pct:.0f}%  (0%=하단 / 50%=중간 / 100%=상단)")
    if bb_pos >= 0.9:
        bb_comment = "밴드 상단에 매우 근접했습니다. 단기 과열로 조정 가능성이 있습니다."
    elif bb_pos >= 0.7:
        bb_comment = "밴드 상단 근처에 있습니다. 상승 모멘텀이 강하지만 추가 상승 여력은 제한적일 수 있습니다."
    elif bb_pos >= 0.3:
        bb_comment = "밴드 중간 구간에 있습니다. 방향성을 좀 더 지켜볼 필요가 있습니다."
    elif bb_pos >= 0.1:
        bb_comment = "밴드 하단 근처에 있습니다. 저점 매수를 고려해볼 수 있는 구간입니다."
    else:
        bb_comment = "밴드 하단을 이탈했습니다. 강한 하락 압력이 있으나 단기 반등도 가능합니다."
    print(f"  → {bb_comment}")

    # ── 5) 이동평균 — 중장기 추세
    print("\n  【 이동평균 (중장기 추세 지표) 】")
    print( "  ※ 주가가 이동평균선 위에 있으면 상승 추세, 아래이면 하락 추세")
    ma20_pct = (ma20_ratio - 1) * 100
    ma60_pct = (ma60_ratio - 1) * 100
    if ma20_ratio >= 1.0 and ma60_ratio >= 1.0:
        print(f"  단기(20일)·중기(60일) 이동평균 모두 위 (MA20 대비 {ma20_pct:+.1f}%, MA60 대비 {ma60_pct:+.1f}%)")
        print("  → 중장기 상승 추세가 지속되고 있습니다.")
    elif ma20_ratio >= 1.0 and ma60_ratio < 1.0:
        print(f"  단기(20일) 위, 중기(60일) 아래 (MA20 대비 {ma20_pct:+.1f}%, MA60 대비 {ma60_pct:+.1f}%)")
        print("  → 단기 반등 중이나 중기 추세는 아직 하락입니다. 추세 전환 확인이 필요합니다.")
    elif ma20_ratio < 1.0 and ma60_ratio >= 1.0:
        print(f"  단기(20일) 아래, 중기(60일) 위 (MA20 대비 {ma20_pct:+.1f}%, MA60 대비 {ma60_pct:+.1f}%)")
        print("  → 중기 추세는 살아있으나 단기 조정 중입니다. 지지 여부를 지켜보세요.")
    else:
        print(f"  단기·중기 이동평균 모두 아래 (MA20 대비 {ma20_pct:+.1f}%, MA60 대비 {ma60_pct:+.1f}%)")
        print("  → 하락 추세입니다. 신규 매수는 추세 전환 확인 후에 하는 것이 안전합니다.")

    # ── 6) 거래량 — 신뢰도 확인
    print("\n  【 거래량 (신뢰도 지표) 】")
    print( "  ※ 거래량이 평균보다 많을수록 현재 주가 움직임의 신뢰도가 높아집니다.")
    print(f"  최근 거래량: 20일 평균 대비 {vol_ratio:.1f}배")
    if vol_ratio >= 2.0:
        vol_comment = "거래량이 평균의 2배 이상입니다. 강한 매매 신호가 나타나고 있습니다."
    elif vol_ratio >= 1.3:
        vol_comment = "거래량이 평균보다 많습니다. 현재 주가 흐름의 신뢰도가 높습니다."
    elif vol_ratio >= 0.7:
        vol_comment = "거래량이 평균 수준입니다."
    else:
        vol_comment = "거래량이 평균보다 적습니다. 현재 주가 흐름을 그대로 믿기보다 좀 더 지켜보는 것이 좋습니다."
    print(f"  → {vol_comment}")

    # ── 7) PER / PBR (가치 지표)
    per_val = df_clean["PER"].iloc[-1] if "PER" in df_clean.columns else float("nan")
    pbr_val = df_clean["PBR"].iloc[-1] if "PBR" in df_clean.columns else float("nan")
    eps_val = df_clean["EPS"].iloc[-1] if "EPS" in df_clean.columns else float("nan")
    bps_val = df_clean["BPS"].iloc[-1] if "BPS" in df_clean.columns else float("nan")
    div_val = df_clean["DIV"].iloc[-1] if "DIV" in df_clean.columns else float("nan")

    per_valid = not pd.isna(per_val)
    pbr_valid = not pd.isna(pbr_val)

    print("\n  【 PER / PBR (가치 지표) 】")
    print("  ※ PER(주가수익비율): 현재 주가가 1년치 이익의 몇 배인지를 나타냅니다.")
    print("     낮을수록 이익 대비 저평가, 높을수록 고평가입니다.")
    print("     ⚠️  업종마다 적정 PER이 다릅니다. 반드시 동종 업종 평균과 비교해야 합니다.")
    print("  ※ PBR(주가순자산비율): 현재 주가가 순자산의 몇 배인지를 나타냅니다.")
    print("     1배 미만이면 장부상 자산보다 싸게 거래되고 있는 것입니다.")
    print("  ※ KRX 펀더멘털 데이터는 사업보고서 검토 후 반영되어 최신 실적과 차이가 있을 수 있습니다.")

    if per_valid:
        print(f"  현재 PER: {per_val:.2f}배", end="")
        if not pd.isna(eps_val):
            print(f"  (EPS: {eps_val:,.0f}원)", end="")
        print()
        if per_val < 0:
            per_comment = "현재 적자 상태입니다. 실적 개선 여부를 반드시 확인하세요."
        elif per_val < 10:
            per_comment = "PER이 낮아 이익 대비 저평가 구간입니다. 가치투자 관점에서 매력적일 수 있습니다."
        elif per_val < 20:
            per_comment = "PER이 적정 수준입니다. 시장 평균과 비교해 판단하세요."
        elif per_val < 40:
            per_comment = "PER이 다소 높습니다. 향후 실적 성장이 뒷받침되어야 합니다."
        else:
            per_comment = "PER이 매우 높습니다. 고성장 기대감이 반영된 것으로, 실적 둔화 시 급락 위험이 있습니다."
        print(f"  → {per_comment}")
    else:
        print("  PER: 데이터 없음 (적자 기업이거나 미집계 상태)")

    if pbr_valid:
        print(f"  현재 PBR: {pbr_val:.2f}배", end="")
        if not pd.isna(bps_val):
            print(f"  (BPS: {bps_val:,.0f}원)", end="")
        print()
        if pbr_val < 0.5:
            pbr_comment = "PBR이 매우 낮아 자산 대비 크게 저평가 상태입니다. 해당 자산의 질(부채 등)도 함께 확인하세요."
        elif pbr_val < 1.0:
            pbr_comment = "PBR이 1배 미만으로 장부가보다 저렴하게 거래되고 있습니다. 저평가 신호일 수 있습니다."
        elif pbr_val < 2.0:
            pbr_comment = "PBR이 적정 수준입니다."
        elif pbr_val < 4.0:
            pbr_comment = "PBR이 다소 높습니다. 브랜드·기술력 등 무형자산 가치가 반영된 것일 수 있습니다."
        else:
            pbr_comment = "PBR이 매우 높습니다. 성장 기대감이 높거나 과열 상태일 수 있습니다."
        print(f"  → {pbr_comment}")
    else:
        print("  PBR: 데이터 없음")

    if not pd.isna(div_val) and div_val > 0:
        print(f"  배당수익률: {div_val:.2f}%  → ", end="")
        if div_val >= 4:
            print("높은 배당 수익률입니다. 배당 투자 관점에서 매력적입니다.")
        elif div_val >= 2:
            print("양호한 배당 수익률입니다.")
        else:
            print("낮은 배당 수익률입니다.")

    # ── 8) 종합 의견 (PER/PBR 점수 반영)
    # 펀더멘털 데이터 유무에 따라 기준 점수를 조정해 공정성 유지
    # PER/PBR 각각 최대 +2점 → 둘 다 없으면 임계치를 4씩 낮춤
    fundamental_bonus_max = (2 if per_valid else 0) + (2 if pbr_valid else 0)
    print("\n  【 종합 의견 】")
    score = 0
    if ens_pct >= 2:   score += 2
    elif ens_pct >= 0: score += 1
    elif ens_pct >= -2: score -= 1
    else:              score -= 2
    if rsi_now < 50:   score += 1
    if rsi_now > 70:   score -= 1
    if macd_now > macd_sig: score += 1
    else:                   score -= 1
    if bb_pos < 0.5:   score += 1
    if bb_pos > 0.8:   score -= 1
    if ma20_ratio >= 1.0: score += 1
    if ma60_ratio >= 1.0: score += 1
    # PER 점수 반영 (유효한 경우에만)
    if per_valid and per_val > 0:
        if per_val < 10:   score += 2
        elif per_val < 20: score += 1
        elif per_val > 40: score -= 1
    # PBR 점수 반영 (유효한 경우에만)
    if pbr_valid and pbr_val > 0:
        if pbr_val < 1.0:  score += 2
        elif pbr_val < 2.0: score += 1
        elif pbr_val > 4.0: score -= 1

    # 기준 점수 = 펀더멘털 최대 가산점만큼 임계치 이동
    t = fundamental_bonus_max
    if score >= 7 - (4 - t):
        overall = "🟢 매수 우위  — 기술적·가치 지표 모두 긍정적입니다. 분할 매수를 고려해볼 수 있습니다."
    elif score >= 4 - (4 - t):
        overall = "🔵 중립 (약 상승)  — 일부 긍정적인 신호가 있습니다. 소량 매수 후 추이를 지켜보세요."
    elif score >= 1 - (4 - t):
        overall = "⚪ 중립  — 뚜렷한 방향성이 없습니다. 관망하며 추가 신호를 기다리는 것이 낫습니다."
    elif score >= -2 - (4 - t):
        overall = "🟡 중립 (약 하락)  — 일부 부정적 신호가 있습니다. 신규 매수보다 보유 물량 관리에 집중하세요."
    else:
        overall = "🔴 매도 우위  — 여러 지표가 부정적입니다. 손절 기준을 점검하고 비중 축소를 고려하세요."

    print(f"  {overall}")
    print("\n" + "=" * 60)
    print("  ⚠️  본 분석은 참고용이며 투자 손실에 대한 책임을 지지 않습니다.")
    print("  ⚠️  모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.")
    print("=" * 60)
    return preds


# ─────────────────────────────────────────────
# 엔트리 포인트
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KRX 주가 예측 시스템",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "사용 예시:\n"
            "  python krx_stock_predictor.py 005930             # 삼성전자 (시장 자동 판별)\n"
            "  python krx_stock_predictor.py 035720 --market KOSDAQ  # 카카오 (시장 명시)\n"
            "  python krx_stock_predictor.py 000660 --days 10   # SK하이닉스, 10 영업일 예측\n"
        ),
    )
    parser.add_argument("ticker",
                        help="종목코드 (예: 005930)")
    parser.add_argument("--market", "-m",
                        choices=["KOSPI", "KOSDAQ", "KONEX"],
                        default=None,
                        help="시장 구분 (미지정 시 자동 판별)")
    parser.add_argument("--days", "-d",
                        type=int, default=5,
                        help="예측 영업일 수 (기본값: 5, 1~60)")
    parser.add_argument("--years", "-y",
                        type=int, default=3,
                        help="학습 데이터 연수 (기본값: 3, 1~10)")

    args = parser.parse_args()

    # --days / --years 범위 검증
    if not (1 <= args.days <= 60):
        parser.error(f"--days는 1~60 사이여야 합니다. (입력값: {args.days})")
    if not (1 <= args.years <= 10):
        parser.error(f"--years는 1~10 사이여야 합니다. (입력값: {args.years})")

    TICKER = validate_ticker(args.ticker)
    run_prediction(
        TICKER,
        target_days=args.days,
        lookback_years=args.years,
        market=args.market,
    )
