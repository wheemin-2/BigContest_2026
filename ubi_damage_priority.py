"""
ubi_ubi_damage_priority.py -- damage size + priority scoring with closed-form intervals

The UBI model is imported (ubi_model.py is reused as is: data loading, holidays, formula, HAC,
vulnerable-industry selection). The damage calculation and the priority scoring are done in this
file. All intervals are closed form (no random numbers).

UBI model (from ubi_model.py)
    log(daily sales) ~ month + weekday + holiday dummies + heat wave + ln(1 + rain mm)
    OLS with Newey-West (HAC, lag 7). The vulnerable industries come from ubi_model_results.csv;
    if that file does not exist, the UBI model is run first (G.main()) and creates it.

Damage size (per selected industry)
    baseline_t   = exp(X_t b | ln_rain = 0) * smear          smear = mean(exp(resid))  (Duan smearing)
    damage rate  d(R) = 1 - (1 + R)^beta
    damage_t     = baseline_t * d(R_t)
    period total = sum_t damage_t
    ratio        = period total / sum of actual sales (same industry, same period)

Intervals: closed form, no random numbers
    beta_L = beta_hat - 1.96 * se,  beta_U = beta_hat + 1.96 * se      (se: HAC standard error)
    d(R) rises as beta falls, so the interval ends of beta map directly to interval ends of
        - the 10mm damage rate        1 - 11^beta
        - the period total damage     sum_t baseline_t * (1 - (1 + R_t)^beta)
        - the ratio to actual sales   total / sum of actual sales
    The baseline (calendar/heat coefficients and smear) is held at its estimate, so these
    intervals reflect the uncertainty of beta only.

Priority score (inside each region, rank 1 = highest priority)
    scale score        = mean baseline * 10mm rate * confidence     confidence = 1 - p (rain coefficient)
    conservative score = mean baseline * 10mm rate lower bound
    scale axis         = (rank of scale score + rank of conservative score) / 2
                         ties are broken by scale-score rank, then conservative rank
    relative axis      = rank of the ratio to actual sales
    final score        = (scale axis + relative axis) / 2           (used for sorting/cumulative share)
    group              = top TOP_N on each axis -> both / scale only / relative only / none
    boundary flag      = 5th/6th place of an axis (or tied with them)

Usage (inputs in <project>/preprocessed_data, results in <project>/outputs)
    python ubi_damage_priority.py [card_data.csv] [--out=outdir]   # default: <project>/preprocessed_data/SH_card_data1_preprocessed.csv
Outputs (in --out, default: folder of this file)
    ubi_damage_priority.csv       damage size, intervals, scores, ranks, groups
    ubi_damage_by_day.csv         baseline and damage per industry x day
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ubi_model reads the card data path from sys.argv[1] at import time, so pass only that
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
_opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
sys.argv = [sys.argv[0]] + _pos[:1]
import ubi_model as G

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUTDIR = Path(_opt["out"]) if "out" in _opt else G.OUT_DIR

REF_MM = 10             # reference rain for the damage rate (10 mm)
LN_REF = np.log1p(REF_MM)
Z95 = 1.959964          # normal quantile for the 95% interval of beta
TOP_N = 5               # top-N per axis defines the groups
REL_RATE_MIN = 10.0     # "relatively highly vulnerable" flag: 10mm damage rate >= this (%)


def vulnerable_list():
    """(region, industry) pairs selected by the UBI model. Runs the model first if its result file does not exist."""
    res = pd.read_csv(G.OUT_ALL) if G.OUT_ALL.exists() else G.main()
    return res[res["폭우취약업종"]][["지역", "업종"]]


# ---------------------------------------------------------------
# 4. Damage size with closed-form intervals
# ---------------------------------------------------------------
def fit_industry(d):
    """Refit one industry and keep what the damage calculation needs."""
    d = d.sort_values("date").reset_index(drop=True)
    res = smf.ols(G.formula_for(d), d).fit(**G.HAC)
    names = list(res.model.exog_names)
    i = names.index("ln_rain")
    X = np.asarray(res.model.exog, float)             # design matrix exactly as fitted
    Xcf = X.copy()
    Xcf[:, i] = 0.0                                   # counterfactual input: rain effect removed
    cov = np.asarray(res.cov_params(), float)
    return dict(
        d=d, b=np.asarray(res.params, float), se=float(np.sqrt(cov[i, i])), i=i, Xcf=Xcf,
        smear=float(np.mean(np.exp(np.asarray(res.resid, float)))),
        p=float(res.pvalues["ln_rain"]), r2adj=float(res.rsquared_adj),
    )


def damage_for(m):
    d, b, i, Xcf, smear, se = m["d"], m["b"], m["i"], m["Xcf"], m["smear"], m["se"]
    rain = d.rain.to_numpy()
    r = np.log1p(rain)
    actual = d.sales.to_numpy()
    tot = actual.sum()

    base = np.exp(Xcf @ b) * smear                    # baseline sales (rain effect removed, Duan smearing)
    beta = b[i]
    beta_L, beta_U = beta - Z95 * se, beta + Z95 * se  # beta_L: larger damage, beta_U: smaller damage

    def total(bt):                                    # period total damage for a given beta, baseline fixed
        return float((base * (1 - np.exp(bt * r))).sum())

    def rate10(bt):                                   # 10mm damage rate for a given beta
        return 1 - np.exp(bt * LN_REF)

    dam = base * (1 - np.exp(beta * r))               # daily damage
    mean_base = base.mean()
    rate, rate_lo, rate_hi = rate10(beta), rate10(beta_U), rate10(beta_L)
    T, T_lo, T_hi = dam.sum(), total(beta_U), total(beta_L)
    conf = 1 - m["p"]

    out = dict(
        관측일=len(d), 비온날_1mm=int((rain >= 1).sum()), 폭우계수=beta, 폭우_se=se, 폭우_p=m["p"], adj_R2=m["r2adj"],
        일평균매출_만원=mean_base / 1e4,
        피해율10_pct=100 * rate, 피해율10_하한_pct=100 * rate_lo, 피해율10_상한_pct=100 * rate_hi,
        일피해10_만원=mean_base * rate / 1e4,
        일피해10_하한_만원=mean_base * rate_lo / 1e4, 일피해10_상한_만원=mean_base * rate_hi / 1e4,
        기간총피해_억=T / 1e8, 기간총피해_하한_억=T_lo / 1e8, 기간총피해_상한_억=T_hi / 1e8,
        기간매출_억=tot / 1e8,
        매출대비총피해_pct=100 * T / tot, 매출대비_하한_pct=100 * T_lo / tot, 매출대비_상한_pct=100 * T_hi / tot,
        확신도=conf,
        규모점수_만원=mean_base * rate * conf / 1e4,
        보수점수_만원=mean_base * rate_lo / 1e4,
    )
    day = pd.DataFrame(dict(date=d.date, 강수mm=rain, 실제매출=actual, 기준매출=base, 피해액=dam))
    return out, day


# ---------------------------------------------------------------
# 5. Priority scoring, groups, boundary flags
# ---------------------------------------------------------------
def add_scores(T):
    g = T.groupby("지역")
    T["r_규모"] = g["규모점수_만원"].rank(ascending=False, method="min")
    T["r_보수"] = g["보수점수_만원"].rank(ascending=False, method="min")
    T["r_상대"] = g["매출대비총피해_pct"].rank(ascending=False, method="min")
    T["규모축"] = (T.r_규모 + T.r_보수) / 2
    T["최종점수"] = (T.규모축 + T.r_상대) / 2
    # scale-axis rank with explicit tie-break: axis value, then scale-score rank, then conservative rank
    T["규모축순위"] = 0
    for _, idx in T.groupby("지역").groups.items():
        o = T.loc[idx].sort_values(["규모축", "r_규모", "r_보수"]).index
        T.loc[o, "규모축순위"] = np.arange(1, len(o) + 1)
    g = T.groupby("지역")
    T["최종순위"] = g["최종점수"].rank(method="min").astype(int)
    T["상대축순위"] = g["r_상대"].rank(method="min").astype(int)
    big, rel = T.규모축순위 <= TOP_N, T.상대축순위 <= TOP_N
    T["집단"] = np.select([big & rel, big, rel], ["둘 다", "규모만", "상대만"], default="해당 없음")
    # boundary flag: 5th/6th place of an axis, or tied with them on the axis value
    flag = pd.Series("", index=T.index)
    for _, g2 in T.groupby("지역"):
        for name, col, rk in [("규모", "규모축", "규모축순위"), ("상대", "r_상대", "상대축순위")]:
            cut_vals = g2.loc[g2[rk].isin([TOP_N, TOP_N + 1]), col].unique()
            hit = g2.index[g2[col].isin(cut_vals)]
            flag.loc[hit] = flag.loc[hit] + name + " "
    T["경계"] = flag.str.strip().replace("", "-")
    T["상대고취약"] = T.피해율10_pct >= REL_RATE_MIN
    share = T.groupby("지역")["기간총피해_억"].transform(lambda s: 100 * s / s.sum())
    T["총피해비중_pct"] = share
    T["누적비중_pct"] = T.assign(s=share).sort_values(["지역", "최종순위"]).groupby("지역")["s"].cumsum().reindex(T.index)
    return T


# ---------------------------------------------------------------
# 6. Run everything
# ---------------------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    daily, _ = G.load_daily(G.CARD_DATA)
    targets = set(map(tuple, vulnerable_list().to_numpy()))

    rows, days = [], []
    for (region, industry), d in daily.groupby(["region", "industry"]):
        if (region, industry) not in targets:
            continue
        out, day = damage_for(fit_industry(d))
        rows.append(dict(지역=region, 업종=industry, **out))
        days.append(day.assign(지역=region, 업종=industry))
    T = add_scores(pd.DataFrame(rows)).sort_values(["지역", "최종순위", "업종"]).reset_index(drop=True)
    T.to_csv(OUTDIR / "ubi_damage_priority.csv", index=False, encoding="utf-8-sig")
    pd.concat(days).to_csv(OUTDIR / "ubi_damage_by_day.csv", index=False, encoding="utf-8-sig")
    return T


if __name__ == "__main__":
    print("card data:", G.CARD_DATA)
    T = main()
    print("selected industries:", T.groupby("지역").size().to_dict())
    pd.set_option("display.width", 250)
    cols = ["업종", "피해율10_pct", "기간총피해_억", "기간총피해_하한_억", "기간총피해_상한_억", "매출대비총피해_pct",
            "규모축순위", "상대축순위", "최종순위", "집단", "경계"]
    for region, t in T.groupby("지역"):
        print(f"\n[{region}] {len(t)} industries, total {t.기간총피해_억.sum():.1f}억 ({100 * t.기간총피해_억.sum() / t.기간매출_억.sum():.2f}% of sales)")
        print(t[cols].round(2).to_string(index=False))
    print("\nsaved to", OUTDIR)
