"""
UBI 대시보드 페이지 생성
=================================================================
ubi_damage_priority.py, ubi_dashboard_precompute.py 결과를 합쳐 data.json과 단일 HTML 페이지(index.html)를 만든다.

입력 (<project>/outputs, <project>/preprocessed_data, 이 폴더)
    ubi_damage_priority.csv        우선순위, 집단, 업종별 피해규모 (ubi_damage_priority.py)
    ubi_dashboard_table.csv        지역 x 업종 x 날짜별 예상 피해 (ubi_dashboard_precompute.py)
    preprocessed_data/forecast/short17_reg_*_d1.csv   강수확률(pop_max)용
    ubi_dashboard_template.html    화면 틀 (__DATA__ 자리에 데이터를 넣음)
    ubi_model.py           공휴일 목록(G.HOLIDAYS)만 가져다 씀
출력
    outputs/data.json, outputs/index.html

화면에 쓰는 값
    일별 예상 피해        ubi_dashboard_table.csv의 _전체 컬럼 (전체 자료로 적합한 시연용 값)
    일별 기준 매출(bs)    ubi_dashboard_table.csv의 기준매출_전체 (업종별, 사업자 화면의 "매출 감소 예상 %" 분모)
    알림 단계별 과거 성적 _백테스트 컬럼 (해당 주를 뺀 적합) 중 CASE_FROM 이후, 공휴일 주 제외, 공휴일 효과 미추정일 제외
    사례일               위와 같은 후보일에서 유형별 예보 강수량 중앙값인 날 (사전 규칙, 사후 선택 아님)
알림 단계 (이 프로젝트 자료에서 정한 기준이며 기상청 특보 기준이 아님)
    3단계 피해 예상: 예보 10mm 이상
    2단계 주의    : 예보 1~10mm, 또는 강수확률 60% 이상
    1단계 비 없음 : 그 외
    오경보 사례   : 예보 10mm 이상인데 관측 FALSE_ALARM_OBS_MM 미만

실행: python ubi_build_dashboard.py          (이어서 index.html을 게시하거나 열면 됨)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import ubi_model as G

HERE = Path(__file__).resolve().parent
START = pd.Timestamp("2025-07-29")       # 화면에 쓰는 첫 날 (기준일 4주가 확보되는 날)
CASE_FROM = pd.Timestamp("2025-08-26")   # 과거 성적, 사례일 후보의 첫 날 (백테스트 기준 매출이 모든 업종에 있는 날부터)
FALSE_ALARM_OBS_MM = 5                   # 오경보: 예보 10mm 이상, 관측이 이 값 미만
SHORT = {"강남": "gangnam", "춘천": "chuncheon"}
GROUP = {"둘 다": "both", "규모만": "scale", "상대만": "rel", "해당 없음": "none"}


def level(fc, pop):
    return np.where(fc >= 10, 2, np.where((fc >= 1) | (pop >= 60), 1, 0))


def holiday_weeks():
    """공휴일(전날·다음날 포함)이 있는 ISO 주 목록"""
    days = set().union(*[set(v) for v in G.HOLIDAYS.values()])
    return {tuple(d.isocalendar()[:2]) for d in days}


def main():
    T = pd.read_csv(G.OUT_DIR / "ubi_dashboard_table.csv", parse_dates=["date"])
    T = T[T["폭우취약업종"]].copy()
    DP = pd.read_csv(G.OUT_DIR / "ubi_damage_priority.csv", encoding="utf-8-sig")
    POP = pd.concat([
        pd.read_csv(G.DATA_DIR / "forecast" / f"short17_reg_{f}_d1.csv", parse_dates=["date"], encoding="utf-8-sig").assign(지역=k)[["지역", "date", "pop_max"]]
        for k, f in SHORT.items()])

    out = {"regions": {}}
    V = T[T.date >= START]
    for reg in ["강남", "춘천"]:
        t = V[V.지역 == reg]
        inds = sorted(t.업종.unique())
        pop = POP[POP.지역 == reg].set_index("date").pop_max
        days = []
        for dt, g in t.groupby("date"):
            g = g.set_index("업종").reindex(inds)
            man = lambda c: [0 if np.isnan(v) else round(v / 1e4) for v in g[c]]          # 원 -> 만원
            days.append({"d": dt.strftime("%Y-%m-%d"), "rf": round(float(g.예보강수mm.iloc[0]), 1), "pop": int(round(pop.get(dt, 0))),
                         "ro": round(float(g.실제강수mm.iloc[0]), 1), "pf": man("예상피해_예보_전체"), "po": man("예상피해_관측_전체"),
                         "base": round(float(np.nansum(g.기준매출_전체)) / 1e8, 2),
                         "bs": man("기준매출_전체"),                                                         # same-day baseline sales per industry (10k KRW, order = inds)
                         "hd": bool(g.공휴일효과_미추정.any()), "nb": int(g.기준매출_전체.isna().sum())})
        out["regions"][reg] = {"inds": inds, "days": days}

    out["rank"] = [dict(reg=r.지역, ind=r.업종, rank=int(r.최종순위), sr=int(r.규모축순위), rr=int(r.상대축순위), grp=GROUP[r.집단],
                        r10=round(r.피해율10_pct, 2), rel=bool(r.상대고취약), tot=round(r.기간총피해_억, 2), rv=round(r.매출대비총피해_pct, 2))
                   for r in DP.itertuples()]
    out["dmg"] = [dict(reg=r.지역, ind=r.업종, sales=round(r.일평균매출_만원, 0), r10=round(r.피해율10_pct, 2), lo=round(r.피해율10_하한_pct, 2),
                       hi=round(r.피해율10_상한_pct, 2), d10=round(r.일피해10_만원, 0), tot=round(r.기간총피해_억, 2), tlo=round(r.기간총피해_하한_억, 2),
                       thi=round(r.기간총피해_상한_억, 2), rev=round(r.기간매출_억, 1), rainy=int(r.비온날_1mm), r2=round(r.adj_R2, 3))
                  for r in DP.itertuples()]

    # 날짜별 합계 (백테스트): 알림 단계별 과거 성적, 사례일
    hw = holiday_weeks()
    T["hw"] = T.date.map(lambda d: tuple(d.isocalendar()[:2]) in hw)
    day = (T.groupby(["지역", "date"])
            .agg(fc=("예보강수mm", "first"), ob=("실제강수mm", "first"), hw=("hw", "first"), n=("업종", "count"),
                 nbase=("기준일수_백테스트", lambda s: int((s >= 3).sum())), un=("공휴일효과_미추정", "sum"),
                 base=("기준매출_백테스트", "sum"), pred=("예상피해_예보_백테스트", "sum"), lo=("예상피해_예보_lo", "sum"),
                 hi=("예상피해_예보_hi", "sum"), post=("예상피해_관측_백테스트", "sum"))
            .reset_index().merge(POP, on=["지역", "date"]))
    e = day[(day.date >= CASE_FROM) & (~day.hw) & (day.nbase == day.n) & (day.un == 0)].copy()
    e["l"] = level(e.fc, e.pop_max)

    total_post = e.groupby("지역").post.sum()
    out["alert"] = [dict(reg=reg, l=int(l), n=len(t), h10=round(float((t.ob >= 10).mean()), 3), pred=round(t.pred.mean() / 1e8, 2),
                         real=round(t.post.mean() / 1e8, 2), share=round(100 * t.post.sum() / total_post[reg], 1))
                    for (reg, l), t in e.groupby(["지역", "l"])]

    e["false"] = (e.fc >= 10) & (e.ob < FALSE_ALARM_OBS_MM)
    cases = []
    for reg, t in e.groupby("지역"):
        for name, m in [("피해예상", (t.fc >= 10) & ~t["false"]), ("주의", (t.fc >= 1) & (t.fc < 10)), ("오경보", t["false"])]:
            c = t[m].sort_values("fc")
            if len(c):
                r = c.iloc[len(c) // 2]                                               # 예보 강수량 중앙값인 날
                cases.append(dict(reg=reg, typ=name, d=r.date.strftime("%Y-%m-%d"), rf=round(r.fc, 1), ro=round(r.ob, 1), base=round(r.base / 1e8, 1),
                                  pred=round(r.pred / 1e8, 1), lo=round(r.lo / 1e8, 1), hi=round(r.hi / 1e8, 1), post=round(r.post / 1e8, 1),
                                  pct=round(100 * r.pred / r.base, 1), candidates=len(c)))
    out["cases"] = cases

    (G.OUT_DIR / "data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tpl = (HERE / "ubi_dashboard_template.html").read_text(encoding="utf-8")
    (G.OUT_DIR / "index.html").write_text(tpl.replace("__DATA__", json.dumps(out, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print("저장: data.json, index.html")
    print("알림 단계별 과거 성적:", len(out["alert"]), "행 | 사례일:", [(c["reg"], c["typ"], c["d"]) for c in cases])


if __name__ == "__main__":
    main()
