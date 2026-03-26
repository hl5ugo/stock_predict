#!/usr/bin/env python3
"""
KRX 주가 예측 FastAPI 서버
==================================================
설치: pip install fastapi uvicorn
실행: uvicorn fastapi_server:app --reload --port 8000

※ krx_stock_predictor.py 와 같은 폴더에 위치해야 합니다.
"""
import math, os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from krx_stock_predictor import (
    get_stock_name, detect_market,
    get_ohlcv, get_fundamental,
    prepare_features, add_technical_indicators,
    StockPredictor,
)

app = FastAPI(title="KRX 주가 예측 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── 인메모리 캐시 ────────────────────────────────
# { cache_key: {"result": {...}, "ts": float} }
_cache: dict = {}
CACHE_TTL = 3600  # 1시간 (초)

def _cache_key(ticker, market, days, years):
    raw = f"{ticker}:{market}:{days}:{years}"
    return hashlib.md5(raw.encode()).hexdigest()

def _get_cache(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["result"]
    return None

def _set_cache(key, result):
    _cache[key] = {"result": result, "ts": time.time()}

# ── index.html 서비스 ────────────────────────────
@app.get("/")
def root():
    """브라우저에서 Render 주소로 접속 시 index.html 반환"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "KRX 주가 예측 API 서버 정상 작동 중. index.html 파일을 같은 폴더에 넣어주세요."}

class PredictRequest(BaseModel):
    ticker: str
    market: Optional[str] = None
    days:   int = 10
    years:  int = 3

# ── 유틸 ────────────────────────────────────────
def sf(v, d=4):
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, d)
    except: return None

def si(v):
    try:
        f = float(v)
        return None if math.isnan(f) else int(f)
    except: return None

# ── 헬스 체크 ───────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ── 예측 엔드포인트 ─────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest):
    # 종목코드 검증
    # 일반 종목: 숫자 6자리 (예: 005930) → zfill 패딩
    # ETF 등 특수 종목: 영숫자 혼합 6자리 (예: 0163Y0, 069500) → 그대로 사용
    raw = req.ticker.strip().upper()
    import re as _re
    if not _re.fullmatch(r"[A-Z0-9]{1,6}", raw):
        raise HTTPException(400, "종목코드는 영숫자 1~6자리여야 합니다. (예: 005930, 0163Y0)")
    # 순수 숫자인 경우에만 6자리로 패딩
    ticker = raw.zfill(6) if raw.isdigit() else raw.ljust(6, "0") if len(raw) < 6 else raw

    days  = max(1, min(60, req.days))
    years = max(1, min(10, req.years))

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=int(365 * years * 1.1))
    s = start_dt.strftime("%Y%m%d")
    e = end_dt.strftime("%Y%m%d")

    # 종목명 / 시장
    stock_name = get_stock_name(ticker)
    mkt = (req.market or "").upper() or detect_market(ticker) or "KOSPI"

    # OHLCV
    df = get_ohlcv(ticker, s, e)
    if df.empty:
        raise HTTPException(404, "OHLCV 데이터를 가져오지 못했습니다. 종목코드를 확인하세요.")

    # 펀더멘털 병합 — 긴 기간 실패 시 단계적으로 기간 축소하여 재시도
    def _try_fundamental(t, start, end):
        """start~end 조회 실패 시 6개월 → 3개월 → 1개월 순으로 재시도"""
        candidates = [(start, end)]
        for months in [6, 3, 1]:
            fb = (end_dt - timedelta(days=30 * months)).strftime("%Y%m%d")
            candidates.append((fb, end))
        for fs, fe in candidates:
            try:
                fd = get_fundamental(t, fs, fe)
                if not fd.empty:
                    # PER/PBR 유효값(0 제외) 확인
                    has_valid = any(
                        (fd[c].replace(0, float("nan")).dropna().shape[0] > 0)
                        for c in ["PER", "PBR"] if c in fd.columns
                    )
                    if has_valid:
                        return fd
            except Exception:
                continue
        return pd.DataFrame()

    df_fund = _try_fundamental(ticker, s, e)
    if not df_fund.empty:
        df = pd.merge(df, df_fund, on="Date", how="left")
        for c in ["PER", "PBR", "EPS", "BPS", "DIV"]:
            if c in df.columns:
                df[c] = df[c].ffill()

    # 피처 엔지니어링 + 모델 학습
    try:
        X, y, _, feat_cols, df_clean = prepare_features(df, days)
    except Exception as ex:
        raise HTTPException(500, f"피처 엔지니어링 오류: {ex}")

    if len(X) < 100:
        raise HTTPException(400, "학습 데이터 부족 (영업일 최소 100일 필요).")

    pred_model = StockPredictor()
    pred_model.train_evaluate(X, y)
    preds = pred_model.predict(df_clean[feat_cols].iloc[-1].values)

    cur     = float(df["Close"].iloc[-1])
    ens     = preds["Ensemble"]
    ens_pct = (ens / cur - 1) * 100

    # 기술적 지표
    di   = add_technical_indicators(df)
    last = di.iloc[-1]
    g    = lambda c: sf(last.get(c))

    rsi  = g("RSI") or 50.0
    macd = g("MACD") or 0.0
    msig = g("MACD_signal") or 0.0
    bbp  = g("BB_pos") or 0.5
    m20r = g("MA20_ratio") or 1.0
    m60r = g("MA60_ratio") or 1.0
    volr = g("Vol_ratio") or 1.0

    # ── 분석 코멘트 생성 ────────────────────────
    pc = ("강한 상승 예상"    if ens_pct >= 5  else
          "소폭 상승 예상"    if ens_pct >= 2  else
          "횡보 예상"         if ens_pct >= -2 else
          "소폭 하락 예상"    if ens_pct >= -5 else "하락 가능성 높음")

    rc = ("매우 과열 — 신규 매수 신중"            if rsi >= 80 else
          "과열 구간 — 수익 실현 고려"             if rsi >= 70 else
          "보통 수준 — 추가 상승 여력 있음"        if rsi >= 50 else
          "다소 낮음 — 매수 고려 구간"             if rsi >= 30 else
          "매우 침체 — 단기 반등 가능, 분할 매수")

    thr = cur * 0.003; gap = macd - msig
    mc = ("MACD Signal 확실히 상회 — 상승 추세 유지"     if gap >  thr  else
          "MACD Signal 근접 상회 — 상승 전환 초기 신호"   if gap >  0    else
          "MACD Signal 근접 하회 — 하락 전환 초기 신호"   if gap > -thr  else
          "MACD Signal 아래 — 하락 추세 지속")

    bp = bbp * 100
    bc = (f"밴드 내 {bp:.0f}% — 상단 근접, 단기 과열 주의"   if bbp >= 0.9 else
          f"밴드 내 {bp:.0f}% — 상단 근처, 추가 여력 제한적"  if bbp >= 0.7 else
          f"밴드 내 {bp:.0f}% — 중간 구간, 방향성 관망"       if bbp >= 0.3 else
          f"밴드 내 {bp:.0f}% — 하단 근처, 저점 매수 고려"    if bbp >= 0.1 else
          f"밴드 내 {bp:.0f}% — 하단 이탈, 강한 하락 압력")

    m20p = (m20r - 1) * 100; m60p = (m60r - 1) * 100
    mac = (f"MA20 {m20p:+.1f}% / MA60 {m60p:+.1f}% — 중장기 상승 추세 지속"    if m20r >= 1 and m60r >= 1 else
           f"MA20 {m20p:+.1f}% / MA60 {m60p:+.1f}% — 단기 반등 중, 중기 하락"  if m20r >= 1 else
           f"MA20 {m20p:+.1f}% / MA60 {m60p:+.1f}% — 중기 추세 유효, 단기 조정" if m60r >= 1 else
           f"MA20 {m20p:+.1f}% / MA60 {m60p:+.1f}% — 하락 추세, 추세 전환 후 매수")

    vc = (f"20일 평균 {volr:.1f}배 — 강한 매매 신호"      if volr >= 2.0 else
          f"20일 평균 {volr:.1f}배 — 신뢰도 높음"          if volr >= 1.3 else
          f"20일 평균 {volr:.1f}배 — 평균 수준"             if volr >= 0.7 else
          f"20일 평균 {volr:.1f}배 — 거래량 부족, 관망 권장")

    # 종합 점수
    per = sf(df_clean["PER"].iloc[-1]) if "PER" in df_clean.columns else None
    pbr = sf(df_clean["PBR"].iloc[-1]) if "PBR" in df_clean.columns else None
    pok = per is not None; bok = pbr is not None

    sc = 0
    sc += 2 if ens_pct >= 2 else (1 if ens_pct >= 0 else (-1 if ens_pct >= -2 else -2))
    sc += 1 if rsi < 50  else (-1 if rsi > 70 else 0)
    sc += 1 if macd > msig else -1
    sc += 1 if bbp < 0.5 else (-1 if bbp > 0.8 else 0)
    sc += 1 if m20r >= 1 else 0
    sc += 1 if m60r >= 1 else 0
    if pok and per and per > 0:
        sc += 2 if per < 10 else (1 if per < 20 else (-1 if per > 40 else 0))
    if bok and pbr and pbr > 0:
        sc += 2 if pbr < 1 else (1 if pbr < 2 else (-1 if pbr > 4 else 0))

    t = (2 if pok else 0) + (2 if bok else 0)
    if   sc >= 7 - (4 - t): ov = "🟢 매수 우위 — 기술적·가치 지표 모두 긍정적, 분할 매수 고려"
    elif sc >= 4 - (4 - t): ov = "🔵 중립 (약 상승) — 일부 긍정 신호, 소량 매수 후 관망"
    elif sc >= 1 - (4 - t): ov = "⚪ 중립 — 방향성 불명확, 추가 신호 대기"
    elif sc >=-2 - (4 - t): ov = "🟡 중립 (약 하락) — 일부 부정 신호, 보유 물량 관리 집중"
    else:                    ov = "🔴 매도 우위 — 다수 지표 부정적, 비중 축소 고려"

    # OHLCV (최근 120일 + 지표)
    dc = add_technical_indicators(df).tail(120)
    ohlcv = [{
        "date":     r["Date"].strftime("%m/%d"),
        "open":     si(r.get("Open")),
        "high":     si(r.get("High")),
        "low":      si(r.get("Low")),
        "close":    si(r.get("Close")),
        "volume":   si(r.get("Volume")),
        "ma5":      si(r.get("MA5")),
        "ma20":     si(r.get("MA20")),
        "ma60":     si(r.get("MA60")),
        "bb_upper": si(r.get("BB_upper")),
        "bb_lower": si(r.get("BB_lower")),
    } for _, r in dc.iterrows()]

    # 앙상블 MAE 기반 신뢰구간
    ens_mae = sum(
        m["MAE"] * pred_model.weights.get(n, 0)
        for n, m in pred_model.metrics.items()
    )

    response = {
        "ticker":      ticker,
        "stock_name":  stock_name,
        "market":      mkt,
        "current_price": si(cur),
        "last_date":   df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "target_days": days,
        "predictions": {k: si(v) for k, v in preds.items()},
        "ensemble_pct": round(ens_pct, 2),
        "confidence_band": {
            "upper": si(ens + ens_mae),
            "lower": si(max(ens - ens_mae, 0)),
        },
        "metrics": {
            n: {
                "MAE":    si(m["MAE"]),
                "RMSE":   si(m["RMSE"]),
                "R2":     sf(m["R2"]),
                "weight": sf(pred_model.weights.get(n, 0)),
            }
            for n, m in pred_model.metrics.items()
        },
        "technical": {
            "rsi":        sf(rsi),
            "macd":       sf(macd),
            "macd_signal":sf(msig),
            "macd_hist":  g("MACD_hist"),
            "bb_pos":     sf(bbp),
            "bb_upper":   si(g("BB_upper")),
            "bb_lower":   si(g("BB_lower")),
            "bb_mid":     si(g("BB_mid")),
            "ma5":        si(g("MA5")),
            "ma20":       si(g("MA20")),
            "ma60":       si(g("MA60")),
            "ma5_ratio":  sf(g("MA5_ratio")),
            "ma20_ratio": sf(m20r),
            "ma60_ratio": sf(m60r),
            "vol_ratio":  sf(volr),
            "per": per, "pbr": pbr,
            "eps": sf(df_clean["EPS"].iloc[-1]) if "EPS" in df_clean.columns else None,
            "bps": sf(df_clean["BPS"].iloc[-1]) if "BPS" in df_clean.columns else None,
            "div": sf(df_clean["DIV"].iloc[-1]) if "DIV" in df_clean.columns else None,
        },
        "analysis": {
            "prediction": pc, "rsi": rc, "macd": mc,
            "bb": bc, "ma": mac, "volume": vc,
            "overall": ov, "score": sc,
        },
        "ohlcv": ohlcv,
        "_cached": False,
    }

    # 캐시 저장 후 반환
    cache_key_val = _cache_key(ticker, req.market or "", days, years)
    _set_cache(cache_key_val, response)
    return response
