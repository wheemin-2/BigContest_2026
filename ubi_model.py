"""
UBI 모형: 강수 취약 업종 판정 (2026-10-24 팀 확정판, 2026-10-26 효과 표시 기준 10mm)
=================================================================
지역 × 업종마다
    ln(일 매출_t) = α + Σγ_m 월_m(8~12월, 기준 7월) + Σδ_k 요일_k(화~일, 기준 월)
                    + β1 명절연휴 + β2 명절당일 + β3 일반공휴일 + β4 공휴일전날 + β5 공휴일다음날
                    + θ1 폭염_t + θ2 ln(1 + 일 강수량_t) + ε_t
    추정: OLS, Newey-West 표준오차(시차 7, 소표본 보정), t분포

데이터
    card data: 신한카드 업종별 매출에서 상단, 하단 이상치 셀을 뺀 데이터 (기존 이름 D4)
      - 기본 입력은 <project>/preprocessed_data/SH_card_data1_preprocessed.csv
      - master1.csv를 넣어도 됩니다. out_low, out_high 열이 있으면 표시된 셀을 빼서 같은 card data를 만듭니다
    날씨: <project>/preprocessed_data/weather_event.csv (date, region, feels, rain), 기상청 AWS 관측
      - 폭염 = 일 최고 체감온도 33℃ 이상인 날, 비 = ln(1 + 일 강수량), 자연로그 (강수량을 구간이나 기준선으로 나누지 않음)

판정 순서
    0. 매출이 있는 날이 100일 이상인 업종만 분석 (min_rows = 100) → 강남 84개, 춘천 77개
    1. 강수 감소 업종: 지역 × 변수 안에서 BH 다중검정 보정 → p < 0.05 이고 q < 0.10 이고 계수 < 0
    2. 수정 결정계수(adjusted R²)가 0.3 미만인 업종 제외 (모형이 하루 매출을 거의 설명하지 못하는 업종)
    → 강수 취약 업종: 강남 13개, 춘천 22개 (결과 표의 열 이름은 다른 코드가 불러 쓰므로 "폭우취약업종" 그대로 둠)

효과 표시 (판정에는 쓰지 않음)
    하루 10mm 비가 온 날, 비 안 온 날 대비 매출 변화 % = (1 + 10)^θ - 1 과 95% 구간 (두 지역 같은 기준)
    10mm로 정한 이유: 강수 구간별로 보면 처음 10mm에서 매출이 뚜렷이 줄고 그 위로 늘어나는 효과는 작음
    참고로 20, 40, 80mm에서의 %도 함께 저장합니다
    판정(p, q, 수정 R²), 피해액, 회복 유형은 기준 mm와 관계없이 같습니다

공휴일 (공휴일 목록만 주면 전날, 다음날은 자동 생성)
    명절연휴 = 추석 연휴 + 대체공휴일, 명절당일 = 추석, 일반공휴일 = 그 외 법정공휴일
    전날 = 공휴일이 아닌 날 중 다음 날이 공휴일인 날, 다음날 = 공휴일이 아닌 날 중 전날이 공휴일인 날

사용
    python ubi_model.py                  # <project>/preprocessed_data/SH_card_data1_preprocessed.csv
    python ubi_model.py 경로/master1.csv  # 원본을 넣으면 이상치 셀을 빼고 실행
출력
    <project>/outputs/ubi_model_results.csv (업종별 전체 결과), ubi_model_vulnerable.csv (강수 취약 업종만)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                # project root (this file lives in <root>/UBI/)
DATA_DIR = ROOT / "preprocessed_data"             # inputs: card data, weather_event.csv, forecast/
OUT_DIR = ROOT / "outputs"                        # all results
OUT_DIR.mkdir(exist_ok=True)
CARD_DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "SH_card_data1_preprocessed.csv"
WEATHER = DATA_DIR / "weather_event.csv"
OUT_ALL = OUT_DIR / "ubi_model_results.csv"
OUT_VUL = OUT_DIR / "ubi_model_vulnerable.csv"

MIN_ROWS = 100      # 최소 관측일 (매출이 있는 날 수)
HEAT_C = 33         # 폭염: 일 최고 체감온도 33℃ 이상
EFFECT_MM = 10      # 효과 표시 기준 강수량 (하루 10mm 비, 두 지역 같음). 판정에는 쓰지 않음
Q_MAX, P_MAX = 0.10, 0.05
ADJ_R2_MIN = 0.3    # 수정 결정계수 기준
HAC = dict(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": True}, use_t=True)
REGION = {"서울 강남구": "강남", "강원 춘천시": "춘천"}


# ---------------------------------------------------------------
# 1. 공휴일 정의
# ---------------------------------------------------------------
def dates(*ds):
    return pd.to_datetime(list(ds))


def build_holidays(festival_days, festival_main_days, other_holidays, next_year_first_holiday=None):
    """공휴일 목록 → 전날/다음날 자동 생성"""
    holidays = festival_days.union(other_holidays)
    all_days = pd.date_range(holidays.min() - pd.Timedelta(days=1), holidays.max() + pd.Timedelta(days=1))
    is_hol = all_days.isin(holidays)
    prev_ = pd.DatetimeIndex([d for d, h in zip(all_days, is_hol) if not h and (d + pd.Timedelta(days=1)) in holidays])
    next_ = pd.DatetimeIndex([d for d, h in zip(all_days, is_hol) if not h and (d - pd.Timedelta(days=1)) in holidays])
    if next_year_first_holiday is not None:          # 데이터 범위 밖 공휴일(신정)의 전날
        prev_ = prev_.union(dates(next_year_first_holiday) - pd.Timedelta(days=1))
    return {"명절연휴": festival_days, "명절당일": festival_main_days, "일반공휴일": other_holidays, "전날": prev_, "다음날": next_}


HOLIDAYS = build_holidays(
    festival_days=pd.date_range("2025-10-05", "2025-10-08"),          # 추석 연휴 + 대체공휴일
    festival_main_days=dates("2025-10-06"),                           # 추석 당일
    other_holidays=dates("2025-08-15", "2025-10-03", "2025-10-09", "2025-12-25"),
    next_year_first_holiday="2026-01-01",
)
VAR_EN = {"명절연휴": "fest", "명절당일": "fest_day", "일반공휴일": "hol", "전날": "eve", "다음날": "after"}


# ---------------------------------------------------------------
# 2. card data → 지역 × 업종 × 날짜 일 매출 + 날씨
# ---------------------------------------------------------------
def load_daily(path):
    header = pd.read_csv(path, nrows=0).columns
    use = [c for c in ["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT", "out_low", "out_high"] if c in header]
    m = pd.read_csv(path, usecols=use)
    if {"out_low", "out_high"} <= set(m.columns):                     # 원본(master1.csv)을 넣은 경우: 상단, 하단 이상치 셀 제거
        flag = m.out_low.astype(str).str.lower().eq("true") | m.out_high.astype(str).str.lower().eq("true")
        print(f"이상치 셀 제거: {int(flag.sum())}개 (전체 {len(m)}개)")
        m = m[~flag]
    m["date"] = pd.to_datetime(m.TA_YMD.astype(str))
    m["region"] = m.MCT_SGG_CD.map(REGION)
    d = (m.groupby(["region", "MCT_RY_CD", "date"], as_index=False).TS_AT.sum()
         .rename(columns={"MCT_RY_CD": "industry", "TS_AT": "sales"}))
    d = d[d.sales > 0]

    w = pd.read_csv(WEATHER, parse_dates=["date"])[["date", "region", "feels", "rain"]]
    d = d.merge(w, on=["region", "date"], how="left", validate="m:1")
    assert d[["feels", "rain"]].notna().all().all(), "날씨가 없는 날이 있음"
    d["heat"] = (d.feels >= HEAT_C).astype(int)
    d["ln_rain"] = np.log1p(d.rain)                                    # ln(1 + 일 강수량), 자연로그
    d["month"], d["dow"] = d.date.dt.month, d.date.dt.dayofweek
    d["log_sales"] = np.log(d.sales)
    for k, v in HOLIDAYS.items():
        d[VAR_EN[k]] = d.date.isin(v).astype(int)
    return d, w


# ---------------------------------------------------------------
# 3. 회귀와 판정
# ---------------------------------------------------------------
def formula_for(d):
    hol = [v for v in VAR_EN.values() if d[v].sum() > 0]
    rhs = ["C(month)", "C(dow)"] + hol + (["heat"] if d.heat.sum() > 0 else []) + ["ln_rain"]
    return "log_sales ~ " + " + ".join(rhs)


def bh(p):
    """BH 보정 q값"""
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = ~np.isnan(p)
    if ok.any(): q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def main():
    daily, w = load_daily(CARD_DATA)
    L = np.log1p(EFFECT_MM)                                            # 효과 표시 기준: 일 강수 10mm (두 지역 같음)
    pct = lambda b, x=1.0: (np.exp(b * x) - 1) * 100

    rows = []
    for (region, industry), d in daily.groupby(["region", "industry"]):
        if len(d) < MIN_ROWS:
            continue
        d = d.sort_values("date")
        res = smf.ols(formula_for(d), d).fit(**HAC)
        ci = res.conf_int()
        b_r = res.params["ln_rain"]
        rows.append({
            "지역": region, "업종": industry, "관측일": len(d), "비온날": int((d.rain > 0).sum()), "폭염일": int(d.heat.sum()),
            "AIC": res.aic, "BIC": res.bic, "R2": res.rsquared, "adj_R2": res.rsquared_adj,
            "폭염_계수": res.params.get("heat", np.nan), "폭염_p": res.pvalues.get("heat", np.nan),
            "폭염_효과pct": pct(res.params.get("heat", np.nan)),
            "호우_계수": b_r, "호우_p": res.pvalues["ln_rain"],
            "효과기준_mm": EFFECT_MM, "호우_효과pct": pct(b_r, L),
            "호우_효과_lo": pct(ci.loc["ln_rain", 0], L), "호우_효과_hi": pct(ci.loc["ln_rain", 1], L),
            "호우_20mm_pct": pct(b_r, np.log1p(20)), "호우_40mm_pct": pct(b_r, np.log1p(40)), "호우_80mm_pct": pct(b_r, np.log1p(80)),
            **{f"{k}_계수": res.params.get(VAR_EN[k], np.nan) for k in VAR_EN},
        })
    out = pd.DataFrame(rows)

    # 1단계: 지역 × 변수 안에서 BH → 감소 = p < 0.05 이고 q < 0.10 이고 계수 < 0
    for nm in ["호우", "폭염"]:
        out[f"{nm}_q"] = np.nan
        for region, idx in out.groupby("지역").groups.items():
            out.loc[idx, f"{nm}_q"] = bh(out.loc[idx, f"{nm}_p"])
        out[f"{nm}_감소"] = (out[f"{nm}_q"] < Q_MAX) & (out[f"{nm}_p"] < P_MAX) & (out[f"{nm}_계수"] < 0)
    # 2단계: 수정 결정계수 0.3 미만 제외
    out["R2기준통과"] = out.adj_R2 >= ADJ_R2_MIN
    out["폭우취약업종"] = out.호우_감소 & out.R2기준통과                 # 강수 취약 업종 (열 이름은 다른 코드와 맞추려고 유지)

    out = out.sort_values(["지역", "폭우취약업종", "호우_효과pct"], ascending=[True, False, True])
    out.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    vul = out[out.폭우취약업종][["지역", "업종", "관측일", "비온날", "호우_효과pct", "호우_효과_lo", "호우_효과_hi", "호우_p", "호우_q", "adj_R2",
                                "호우_20mm_pct", "호우_40mm_pct", "호우_80mm_pct"]]
    vul.to_csv(OUT_VUL, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    print("입력:", CARD_DATA)
    res = main()
    print(f"\n0. 분석 업종 (관측 {MIN_ROWS}일 이상):", res.groupby("지역").size().to_dict())
    print(f"1. 강수 감소 업종 (p < {P_MAX}, q < {Q_MAX}):", res.groupby("지역").호우_감소.sum().astype(int).to_dict(),
          "| 폭염 감소 업종:", res.groupby("지역").폭염_감소.sum().astype(int).to_dict())
    drop = res[res.호우_감소 & ~res.R2기준통과]
    print(f"2. 수정 R² {ADJ_R2_MIN} 미만이라 제외:", ", ".join(f"{a} {b} ({r:.3f})" for a, b, r in zip(drop.지역, drop.업종, drop.adj_R2)) or "없음")
    vul = res[res.폭우취약업종]
    print(f"\n효과 표시: 일 강수 {EFFECT_MM}mm 비 기준")
    print("강수 취약 업종:", vul.groupby("지역").size().to_dict())
    for region, g in vul.groupby("지역"):
        print(f"- {region} {len(g)}: " + ", ".join(g.업종))
    print("\n저장:", OUT_ALL, "|", OUT_VUL)
