"""
UBI 대시보드용 사전 계산 테이블 (백테스트, 누출 없음)
=================================================================
지역 x 업종 x 날짜마다 아래 값을 미리 계산해 두면, 대시보드는 예보 강수량만 받아 즉시 계산 가능

    예상 피해(F mm) = 기준매출 x (1 - (1 + F) ^ theta_fc)          <- (1+F)^theta = exp(theta * ln(1+F))

기준 매출과 강수 계수를 따로 만든다
    기준매출  : 직전 N주 같은 요일의 실제 매출을, 관측 입력 UBI 모형의 효과로 날짜 t의 조건에 맞춰 평균
                (기준일의 비 효과를 theta_obs로 걷어내고, 기준일의 월/공휴일/폭염 효과를 빼고 t의 효과를 더함)
    theta_fc  : 예보 강수(ln(1+예보 일 강수량))로 적합한 같은 UBI 모형의 강수 계수 (예보 입력에는 이 계수가 맞음)
    theta_obs : 관측 강수로 적합한 계수 (기준일의 비를 걷어낼 때, 사후 확인 피해를 구할 때 사용)

계산 순서 (업종마다)
    1. 날짜 t가 속한 주(월~일)를 뺀 데이터로 UBI 모형을 두 번 적합 (GAP_DAYS > 0이면 앞뒤 GAP_DAYS일도 제외)
         - 관측 입력: theta_obs, 월/공휴일/폭염 계수
         - 예보 입력: theta_fc (NW 표준오차로 95% 구간)
    2. t 이전 N_REF_WEEKS주 안의 같은 요일 날들을 고름
    3. 각 날의 실제 매출에서 비/월/공휴일/폭염 효과를 걷어내고 t의 조건(월/공휴일, 예보 폭염)으로 맞춤
    4. 평균(또는 중앙값) = 기준매출

블록 정의
    ISO 주(월~일) 단위로 검정할 주 하나만 빼고 적합한다 (GAP_DAYS = 0). 7/1~12/31은 부분 주 2개를 포함해 27개 블록이다.
    NW 시차(7일)를 고려해 앞뒤 이웃 날까지 빼고 싶으면 GAP_DAYS = 7 (총 21일 제외)로 바꾼다.

기준 일수/집계 방식
    N_REF_WEEKS, REF_AGG는 근거 있는 값이 아니라 선택해야 하는 설정이다.
    python ubi_dashboard_precompute.py <card_data> --tune   으로 후보별 오차를 비교해 정한다 (비 오지 않은 날의 기준매출 오차)

필요 파일: UBI/ubi_model.py, preprocessed_data/SH_card_data1_preprocessed.csv (또는 master1.csv),
                       preprocessed_data/weather_event.csv, preprocessed_data/forecast/short17_reg_{gangnam,chuncheon}_d1.csv,
                       ubi_model_results.csv (없으면 ubi_model.py를 먼저 실행해 만든다)
출력: ubi_dashboard_table.csv  (--tune이면 ubi_reference_tuning.csv)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
TUNE = "--tune" in sys.argv
sys.argv = sys.argv[:1]
import ubi_model as G
INPUT = ARGS[0] if ARGS else G.CARD_DATA                         # default: <project>/preprocessed_data/SH_card_data1_preprocessed.csv
G.CARD_DATA = Path(INPUT)                                         # so that G.main() (run when results are missing) reads the same file

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
FORECAST = {"강남": G.DATA_DIR / "forecast" / "short17_reg_gangnam_d1.csv", "춘천": G.DATA_DIR / "forecast" / "short17_reg_chuncheon_d1.csv"}

N_REF_WEEKS = 4                                  # 기준일: t 이전 N주 안의 같은 요일 (--tune으로 확인)
REF_AGG = "mean"                                 # "mean" | "median"
GAP_DAYS = 0                                     # 검정 주(월~일) 앞뒤로 더 빼는 날 수. 0이면 해당 주만 제외 (NW 시차와 맞추려면 7)
MIN_TRAIN = 60                                   # 학습 일수가 이보다 적으면 그 블록은 계산하지 않음
HOL_VARS = list(G.VAR_EN.values())               # fest, fest_day, hol, eve, after
EFFECT_VARS = HOL_VARS + ["heat"]


# ---------------------------------------------------------------
# 1. 자료: 일 매출 + 관측 날씨 + 예보
# ---------------------------------------------------------------
def load_data(path):
    daily, _ = G.load_daily(path)
    parts = []
    for region, f in FORECAST.items():
        fc = pd.read_csv(f, parse_dates=["date"])[["date", "app_max", "heat_app33", "rain_sum", "rain_log1p"]]
        parts.append(fc.assign(region=region))
    fc = pd.concat(parts).rename(columns={"heat_app33": "heat_fc", "rain_sum": "rain_fc", "rain_log1p": "ln_rain_fc"})
    d = daily.merge(fc[["region", "date", "heat_fc", "rain_fc", "ln_rain_fc"]], on=["region", "date"], how="left")
    d["heat_fc"] = d.heat_fc.fillna(0).astype(int)                     # 예보가 없는 날은 폭염 아님으로 처리
    d["week"] = d.date.dt.to_period("W-SUN")                           # 월~일 주
    return d


def as_forecast_input(d):
    """같은 식(formula_for)을 쓰기 위해 열 이름만 바꾼 복사본: heat <- 예보 폭염, ln_rain <- 예보 ln(1+강수)"""
    return d.assign(heat=d.heat_fc, ln_rain=d.ln_rain_fc)


# ---------------------------------------------------------------
# 2. 효과와 기준 매출
# ---------------------------------------------------------------
def calendar_effect(params, row, heat):
    """월 + 공휴일 + 폭염 효과 (요일은 같은 요일끼리 비교하므로 제외). heat는 값을 직접 받는다."""
    e = params.get(f"C(month)[T.{row['month']}]", 0.0)
    for v in HOL_VARS:
        e += params.get(v, 0.0) * row[v]
    return e + params.get("heat", 0.0) * heat


def unestimable(params, row):
    """학습에 없던 공휴일 더미(그 날의 공휴일 효과를 추정할 수 없음)가 이 날에 켜져 있는지"""
    return any(row[v] == 1 and v not in params.index for v in HOL_VARS)


def baseline(d, t, p, heat_t, n_weeks=None, agg=None):
    """직전 N주 같은 요일의 매출을 t의 조건으로 맞춘 값의 평균. 반환: (기준매출, 쓴 기준일 수)"""
    n_weeks = n_weeks or N_REF_WEEKS
    lo = t["date"] - pd.Timedelta(days=7 * n_weeks)
    ref = d[(d["dow"] == t["dow"]) & (d["date"] < t["date"]) & (d["date"] >= lo)]
    th, ce_t = p.get("ln_rain", 0.0), calendar_effect(p, t, heat_t)
    adj = []
    for _, r in ref.iterrows():
        if unestimable(p, r):                                          # 효과를 추정할 수 없는 기준일은 쓰지 않음
            continue
        adj.append(r["sales"] * np.exp(-th * r["ln_rain"] - calendar_effect(p, r, r["heat"]) + ce_t))
    if not adj:
        return np.nan, 0
    return float(np.median(adj) if (agg or REF_AGG) == "median" else np.mean(adj)), len(adj)


def damage_rate(F, theta):
    return 1 - (1 + F) ** theta


# ---------------------------------------------------------------
# 3. 블록별 적합 (검정 주와 앞뒤 GAP_DAYS를 뺀 학습)
# ---------------------------------------------------------------
def fit_blocks(d):
    """반환: {주: (관측 적합 params, 예보 적합 params, theta_fc 95% 구간)}"""
    fitted = {}
    for wk, blk in d.groupby("week"):
        s, e = blk["date"].min() - pd.Timedelta(days=GAP_DAYS), blk["date"].max() + pd.Timedelta(days=GAP_DAYS)
        tr = d[(d["date"] < s) | (d["date"] > e)]
        if len(tr) < MIN_TRAIN:
            fitted[wk] = None
            continue
        po = smf.ols(G.formula_for(tr), tr).fit(**G.HAC).params
        trf = as_forecast_input(tr)
        rf = smf.ols(G.formula_for(trf), trf).fit(**G.HAC)
        ci = rf.conf_int().loc["ln_rain"] if "ln_rain" in rf.params.index else (np.nan, np.nan)
        fitted[wk] = (po, rf.params, (ci[0], ci[1]))
    return fitted


def industry_table(d):
    d = d.sort_values("date").reset_index(drop=True)
    po_full = smf.ols(G.formula_for(d), d).fit(**G.HAC).params
    df_ = as_forecast_input(d)
    rf_full = smf.ols(G.formula_for(df_), df_).fit(**G.HAC)
    th_obs_full, th_fc_full = po_full.get("ln_rain", np.nan), rf_full.params.get("ln_rain", np.nan)
    blocks = fit_blocks(d)
    rows = []
    for _, t in d.iterrows():
        fb = blocks[t["week"]]
        row = {"date": t["date"].date(), "요일": t["dow"], "theta_obs_전체": th_obs_full, "theta_fc_전체": th_fc_full,
               "예보강수mm": t["rain_fc"], "예보폭염": t["heat_fc"], "실제강수mm": t["rain"], "실제폭염": t["heat"], "실제매출": t["sales"]}
        for tag, p, th_fc, ci in [("백테스트", *( (fb[0], fb[1].get("ln_rain", np.nan), fb[2]) if fb else (None, np.nan, (np.nan, np.nan)) )),
                                  ("전체", po_full, th_fc_full, (np.nan, np.nan))]:
            if p is None:
                continue
            th_obs = p.get("ln_rain", np.nan)
            b_fc, n_ref = baseline(d, t, p, t["heat_fc"])                # 예측 시점: t의 폭염은 예보
            b_ob, _ = baseline(d, t, p, t["heat"])                       # 사후 확인: t의 폭염은 관측
            r_fc, r_ob = damage_rate(t["rain_fc"], th_fc), damage_rate(t["rain"], th_obs)
            row.update({f"theta_obs_{tag}": th_obs, f"theta_fc_{tag}": th_fc, f"기준일수_{tag}": n_ref,
                        f"기준매출_{tag}": b_fc, f"피해율_예보_{tag}": r_fc, f"예상피해_예보_{tag}": b_fc * r_fc,
                        f"기준매출_관측폭염_{tag}": b_ob, f"피해율_관측_{tag}": r_ob, f"예상피해_관측_{tag}": b_ob * r_ob})
            if tag == "백테스트":
                lo_rate = max(0.0, damage_rate(t["rain_fc"], min(ci[1], 0.0)))   # theta가 덜 음수인 쪽이 피해 하한
                hi_rate = damage_rate(t["rain_fc"], ci[0])
                row.update({"theta_fc_lo": ci[0], "theta_fc_hi": ci[1], "피해율_예보_lo": lo_rate, "피해율_예보_hi": hi_rate,
                            "예상피해_예보_lo": b_fc * lo_rate, "예상피해_예보_hi": b_fc * hi_rate,
                            "공휴일효과_미추정": bool(unestimable(p, t))})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 4. 기준일 수/집계 방식 비교 (--tune)
# ---------------------------------------------------------------
def tune_reference(daily, targets, n_grid=(2, 3, 4, 6, 8), aggs=("mean", "median")):
    """비 오지 않은 날(관측 1mm 미만)에는 기준매출이 실제 매출과 같아야 한다고 보고, 후보별 오차를 비교한다.
    공통 평가 구간(가장 긴 N주 이력이 있는 날부터), 공휴일 더미가 켜진 날 제외, 업종별 평균 절대 오차의 중앙값"""
    res = []
    start_min = daily["date"].min() + pd.Timedelta(days=7 * max(n_grid))
    for (region, industry), d in daily.groupby(["region", "industry"]):
        if (region, industry) not in targets:
            continue
        d = d.sort_values("date").reset_index(drop=True)
        blocks = fit_blocks(d)
        ev = d[(d["date"] >= start_min) & (d["rain"] < 1) & (d[HOL_VARS].sum(axis=1) == 0)]
        for n in n_grid:
            for agg in aggs:
                err = []
                for _, t in ev.iterrows():
                    fb = blocks[t["week"]]
                    if fb is None:
                        continue
                    b, k = baseline(d, t, fb[0], t["heat"], n, agg)       # 사후 조건(관측 폭염)으로 비교
                    if k >= max(2, n // 2):
                        err.append(b / t["sales"] - 1)
                if err:
                    err = np.array(err)
                    res.append({"지역": region, "업종": industry, "N주": n, "집계": agg, "MAPE": np.mean(np.abs(err)) * 100,
                                "편향": np.mean(err) * 100, "일수": len(err)})
    return pd.DataFrame(res)


def target_set():
    out = G.OUT_ALL
    if not out.exists():
        G.main()
    res = pd.read_csv(out)
    return {(r, i) for r, i, v in zip(res["지역"], res["업종"], res["폭우취약업종"]) if v}


def main():
    daily = load_data(INPUT)
    targets = target_set()
    if TUNE:
        t = tune_reference(daily, targets)
        t.to_csv(G.OUT_DIR / "ubi_reference_tuning.csv", index=False, encoding="utf-8-sig")
        s = t.groupby(["N주", "집계"])[["MAPE", "편향"]].median().round(2)
        print(s)
        return t
    out = []
    for (region, industry), d in daily.groupby(["region", "industry"]):
        if len(d) < G.MIN_ROWS:
            continue
        out.append(industry_table(d).assign(지역=region, 업종=industry, 폭우취약업종=(region, industry) in targets))
    tab = pd.concat(out)
    cols = ["지역", "업종", "폭우취약업종", "date"] + [c for c in tab.columns if c not in ("지역", "업종", "폭우취약업종", "date")]
    tab[cols].to_csv(G.OUT_DIR / "ubi_dashboard_table.csv", index=False, encoding="utf-8-sig")
    return tab


if __name__ == "__main__":
    t = main()
    if not TUNE:
        print(t.shape, "기준일 없는 비율(백테스트):", round(t["기준매출_백테스트"].isna().mean(), 3))
