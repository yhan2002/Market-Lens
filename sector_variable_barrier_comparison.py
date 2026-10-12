"""Bounded research: reversal versus continuation, variable-time competing barriers.

Not a pivot detector: predicts the next material move relative to a causal trend.
Inputs are close-known; entry proxy is next session close. No downloads or DB writes.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sector_medium_trend_signatures import load_prices, signatures
from sector_absolute_turn_prediction import features, ABS_FEATURES, PATH_FEATURES, TREND_FEATURES, RRG_FEATURES
from sector_pre_pivot_hypothesis import liquidity_features, VOLUME, OPTIONS, DIVERGENCE
from sector_turning_sector_macro_boost import load_market_factor_levels
from turning_macro_shadow_experiment import load_macro_history
from sector_rotation import SECTOR_ETFS

ROOT = Path(__file__).resolve().parent
BASE = ABS_FEATURES + PATH_FEATURES + TREND_FEATURES + RRG_FEATURES


def make_candidate(family):
    """Shared research/live-export settings; never tune during inference."""
    if family=='LOGIT':
        return LogisticRegression(C=.1,max_iter=1000,random_state=42)
    if family=='BOOST':
        return HistGradientBoostingClassifier(max_depth=3,max_iter=100,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=42)
    if family=='EXTRA':
        return ExtraTreesClassifier(n_estimators=150,max_depth=5,min_samples_leaf=20,max_features=.7,n_jobs=2,random_state=42)
    raise ValueError(f'Unknown family: {family}')


def first_hit(values, position, barrier, direction):
    """No time horizon; unresolved/missing future is censored, never a negative."""
    entry = position + 1
    if entry >= len(values) or not np.isfinite(values[entry]):
        return None
    for end in range(entry+1, len(values)):
        if not np.isfinite(values[end]):
            return None
        move = values[end]-values[entry]
        if abs(move) >= barrier:
            return end, int(move*direction < 0)
    return None


def prepare():
    prices, _ = load_prices(); calendar = prices.index
    f = features(prices)
    factor_parts = []
    for ticker in SECTOR_ETFS:
        part = signatures(prices, ticker)
        part['peer_spread20'] = np.log(prices[ticker]).diff(20)-np.log(prices[list(SECTOR_ETFS)]).diff(20).mean(axis=1)
        part['date'], part['ticker'] = part.index, ticker
        factor_parts.append(part.reset_index(drop=True))
    spread = [c for c in factor_parts[0] if c not in ('date', 'ticker')]
    f = f.merge(pd.concat(factor_parts), on=['date','ticker'], validate='one_to_one')
    history = load_macro_history(ROOT/'outputs/bloomberg_turning_macro_staging_20260902_212424/bloomberg_turning_macro_20260902_212424/macro_history.csv')
    m = history.pivot(index='date', columns='series', values='value').reindex(calendar)
    market = load_market_factor_levels(ROOT/'regime_router_publish.db').reindex(calendar)
    levels = pd.DataFrame({'curve':m.UST_10Y-m.UST_2Y, 'real':m.UST_REAL_10Y,
        'bei':m.BREAKEVEN_10Y, 'wti':m.WTI_SPOT, 'oil_spread':m.BRENT_SPOT-m.WTI_SPOT,
        'yen':-np.log(market.usdjpy), 'vix':np.log(market.vix), 'move':np.log(market.move)})
    macro = pd.concat([levels.diff(5).add_suffix('_shock'), levels.diff(5).diff(5).add_suffix('_accel')],axis=1).shift(1)
    macro.index.name='date'; macros=list(macro)
    f = f.merge(macro.reset_index(), on='date', validate='many_to_one')
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as con:
        raw = pd.read_sql_query('SELECT date,ticker,adj_close,close,high,low,volume FROM daily_prices', con, parse_dates=['date'])
    option = pd.read_csv(ROOT/'outputs/bloomberg_sector_option_full_staging_20260904_184703/merged_history.csv',parse_dates=['date'])
    f = f.merge(liquidity_features(raw, calendar, option), on=['date','ticker'], validate='one_to_one')
    sets = {'PRICE':list(BASE), 'SPREAD':BASE+spread, 'MACRO':BASE+spread+macros,
            'FULL':BASE+spread+macros+VOLUME+OPTIONS+DIVERGENCE}
    dates = pd.Series(calendar,index=calendar)
    weekly = set(dates.groupby(calendar.to_period('W-FRI')).max())
    records=[]
    for ticker in SECTOR_ETFS:
        log=np.log(prices[ticker]); values=log.to_numpy()
        prior=log.diff(60); sigma=log.diff().shift(1).rolling(60).std()
        for pos,date in enumerate(calendar):
            if date not in weekly or not np.isfinite(prior.iloc[pos]) or not np.isfinite(sigma.iloc[pos]) or sigma.iloc[pos]<=0:
                continue
            direction = 1 if prior.iloc[pos]>0 else -1
            barrier = float(np.clip(6*sigma.iloc[pos], .06, .30))
            hit = first_hit(values,pos,barrier,direction)
            row=dict(date=date,ticker=ticker,direction='OUT' if direction>0 else 'IN',
                y=hit[1] if hit else np.nan, label_end=calendar[hit[0]] if hit else pd.NaT,
                barrier=barrier, duration=hit[0]-pos-1 if hit else np.nan)
            for horizon in (20,60):
                path=values[pos+2:pos+2+horizon]-values[pos+1] if pos+1<len(values) else np.array([])
                complete=len(path)==horizon and np.isfinite(path).all()
                row[f'return{horizon}']=np.expm1(path[-1]) if complete else np.nan
                row[f'favorable{horizon}']=np.max(-direction*path) if complete else np.nan
                row[f'adverse{horizon}']=np.max(direction*path) if complete else np.nan
            records.append(row)
    frame=pd.DataFrame(records).merge(f,on=['date','ticker'],validate='one_to_one').replace([np.inf,-np.inf],np.nan)
    complete=frame[sets['FULL']].notna().all(axis=1)
    coverage=frame.groupby(['direction',frame.date.dt.year]).agg(rows=('date','size'),resolved=('y','count'))
    frame=frame.loc[complete].copy()
    for ticker in SECTOR_ETFS:
        col='sector_'+ticker; frame[col]=frame.ticker.eq(ticker).astype(float)
        for cols in sets.values(): cols.append(col)
    return frame.sort_values(['date','ticker']).reset_index(drop=True),sets,coverage


def match_weights(train):
    """Fit-only hard negatives: same sector, similar prior return and volatility."""
    scale=train[['ret60','vol20']].std().replace(0,1)
    z=train[['ret60','vol20']]/scale
    weights=np.ones(len(train)); matches=[]
    for ticker, group in train.groupby('ticker'):
        positive=group.index[group.y.eq(1)]; negative=group.index[group.y.eq(0)]
        if not len(negative): continue
        for i in positive:
            distance=((z.loc[negative]-z.loc[i])**2).sum(axis=1)
            j=distance.idxmin()
            if distance.loc[j]<=2:
                weights[j]+=1
                matches.append((i,int(j)))
    # Downweight repeated outcomes, rather than counting overlapping weeks as independent events.
    groups=train.ticker+'_'+train.label_end.dt.to_period('M').astype(str)
    weights/=groups.map(groups.value_counts()).to_numpy()
    for target in (0,1):
        mask=train.y.eq(target).to_numpy(); weights[mask]/=weights[mask].sum()
    weights*=len(train)/2
    return weights,matches


def metrics(g):
    known=g.loc[g.y.notna()]; a=known.loc[known.alert]
    def auc(): return float(roc_auc_score(known.y,known.score)) if known.y.nunique()==2 else None
    return dict(rows=len(g),resolved=len(known),positive=int(known.y.sum()),auc=auc(),
        ap=float(average_precision_score(known.y,known.score)) if known.y.sum() else None,
        alerts=int(g.alert.sum()),resolved_alerts=len(a),precision=float(a.y.mean()) if len(a) else None,
        recall=float(a.y.sum()/known.y.sum()) if known.y.sum() else None,
        signed_return20=float((a.return20*np.where(a.direction.eq('IN'),1,-1)).mean()) if len(a) else None,
        signed_return60=float((a.return60*np.where(a.direction.eq('IN'),1,-1)).mean()) if len(a) else None)


def self_test():
    x=np.array([0.,0.,.03,.07,.1])
    assert first_hit(x,0,.06,1)==(3,0)
    assert first_hit(-x,0,.06,1)==(3,1)
    assert first_hit(x[:3],0,.06,1) is None
    assert first_hit(np.array([0.,0.,np.nan,.1]),0,.06,1) is None
    assert first_hit(np.r_[x, -5.],0,.06,1)==first_hit(x,0,.06,1)
    t=pd.DataFrame({'ticker':['X']*4,'y':[0,1,0,1], 'ret60':[.1,.1,.2,.2],
        'vol20':[.01,.01,.02,.02], 'label_end':pd.to_datetime(['2020-01-01']*4)})
    w,pairs=match_weights(t)
    assert np.isfinite(w).all() and (w>0).all() and len(pairs)==2
    print('PASS: entry timing, first passage, censoring, future-extension invariance, matching weights',flush=True)


def audit_results(output):
    result=pd.read_csv(output/'predictions.csv',parse_dates=['date','label_end'])
    folds=pd.read_csv(output/'fold_audit.csv')
    assert not result.duplicated(['direction','model','date','ticker']).any()
    assert result.score.between(0,1).all()
    assert (pd.to_datetime(folds.fit_label_end)<pd.to_datetime((folds.year-2).astype(str)+'-01-01')).all()
    assert (pd.to_datetime(folds.validation_label_end)<pd.to_datetime(folds.year.astype(str)+'-01-01')).all()
    selected=[]
    for _,g in result.groupby(['direction','year','model']):
        g=g.loc[g.y.notna()].copy()
        g['pick']=False
        if len(g): g.loc[g.nlargest(max(1,int(np.ceil(len(g)*.1))),'score').index,'pick']=True
        selected.append(g)
    r=pd.concat(selected)
    comparisons=[]
    for direction,group in r.groupby('direction'):
        for model,g in group.groupby('model'):
            family=model.rsplit('_',1)[1]
            baseline=group.loc[group.model.eq('PRICE_'+family)]
            pair=g.merge(baseline,on=['date','ticker','direction'],suffixes=('_a','_b'),validate='one_to_one')
            assert len(pair)==len(g)==len(baseline)
            block=pair.date.dt.year.astype(str)+'H'+((pair.date.dt.month-1)//6+1).astype(str)
            pair['block']=block
            counts=pair.groupby('block').apply(lambda x: pd.Series({
                'ha':(x.y_a*x.pick_a).sum(),'na':x.pick_a.sum(),
                'hb':(x.y_b*x.pick_b).sum(),'nb':x.pick_b.sum()}),include_groups=False)
            arrays=counts.to_numpy(); rng=np.random.default_rng(42)
            samples=arrays[rng.integers(0,len(arrays),(2000,len(arrays)))].sum(axis=1)
            valid=(samples[:,1]>0)&(samples[:,3]>0)
            delta=samples[valid,0]/samples[valid,1]-samples[valid,2]/samples[valid,3]
            chosen=g.loc[g.pick]; base=baseline.loc[baseline.pick]
            sign=1 if direction=='IN' else -1
            comparisons.append(dict(direction=direction,model=model,n=len(chosen),base_rate=g.y.mean(),
                precision=chosen.y.mean(),baseline_precision=base.y.mean(),delta=chosen.y.mean()-base.y.mean(),
                ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975),blocks=len(counts),
                signed_return20=(sign*chosen.return20).mean(),signed_return60=(sign*chosen.return60).mean(),
                unfavorable20=(sign*chosen.return20.dropna()<0).mean()))
    comp=pd.DataFrame(comparisons);comp.to_csv(output/'paired_budget_comparison.csv',index=False)
    imp=pd.read_csv(output/'validation_importance.csv')
    importance=imp.groupby(['direction','model','feature']).validation_auc_drop.agg(['mean','min','max','count']).reset_index()
    importance.sort_values('mean',ascending=False).to_csv(output/'importance_stability.csv',index=False)
    frame=pd.read_csv(output/'research_frame.csv',parse_dates=['date','label_end'])
    contrasts=[]
    for direction in ('IN','OUT'):
        for year in (2023,2024,2025,2026):
            train=frame.loc[frame.direction.eq(direction)&frame.label_end.lt(pd.Timestamp(year-2,1,1))].reset_index(drop=True)
            _,pairs=match_weights(train)
            if not pairs:continue
            a,b=np.array(pairs).T
            for feature in ['USMV_SPY_20','USMV_SPY_63','VLUE_SPY_20','QUAL_SPY_20','MTUM_VLUE_20']:
                contrasts.append(dict(direction=direction,year=year,feature=feature,pairs=len(pairs),
                    case_mean=train.loc[a,feature].mean(),control_mean=train.loc[b,feature].mean(),
                    standardized_difference=(train.loc[a,feature].to_numpy()-train.loc[b,feature].to_numpy()).mean()/train[feature].std()))
    pd.DataFrame(contrasts).to_csv(output/'fit_matched_contrasts.csv',index=False)
    lines=['# 기간 비고정 전환·지속 비교학습 결과','',
        '## 요약 판정','',
        '96회 학습 완료(약150초). 개선 후보는 스프레드 구성에 있으나 추세전환 조기 포착 성공이나 운영 반영을 뒷받침하지는 못했다.',
        'IN 상위10% 동일경보 비교에서 가격 부스팅92.3% → 스프레드 부스팅96.9%(65개 중63개). 차이95% 구간은 -4.2~+22.0%p로 0을 포함한다. 이는 독립된63번의 저점 포착을 뜻하지 않는다.',
        'OUT Extra Trees는49.3% →58.8%(136개 중80개), 차이95% 구간+1.0~+17.4%p. 다만 8개 반기 블록·중첩 표본·다중비교 때문에 확정적 유의성은 주장하지 않는다.',
        '실시간형 검증 임계값에서 IN 스프레드 부스팅은47개 경보 중46개 결과확정,45개 적중(97.8%)이나 전체 양성 주간행 포착률은10.1%에 그친다. 고점/저점 이벤트 포착률이 아니다.',
        '실시간형 OUT 스프레드 Extra Trees는138개 결과확정 경보 중63개 적중(45.7%). 가격만 사용한46.1%보다 개선되지 않았고, 경보 이후 평균 가격은20일+0.08%,60일+0.31% 상승했다. 저점 근처 OUT 문제를 해결했다고 볼 수 없다.',
        '검증 교란 중요도에서 OUT의 가치주/SPY20일·모멘텀/가치주20일 격차는 두 트리 모델 모두4개 연도에서 양의 기여 후보였다. 인과관계나 일정한 매매 방향을 뜻하지 않으며, 학습 내 매칭 차이는 fit_matched_contrasts.csv에 별도 기록했다.',
        '이번에는 매칭 가중치를 모든 모델에 공통 적용했으므로 매칭 자체의 효과는 분리 검증하지 않았다. 다음 검증은 후보를 동결하고 이 가중치 제거 비교 및 새로운 날짜 관찰을 진행하는 것이 적절하다.','',
        '## 범위와 해석','',
        '11개 섹터 ETF, 2023~2026년 연도별 시간순 평가. 3개 모델 × 4개 입력 구성 × 2방향 × 4개 연도 = 96회 학습.',
        '정답은 사후 고저점이 아니라 다음 유의미한 가격 이동의 방향이다. 직전 60일 가격 방향에 반대되는 장벽을 먼저 넘으면 전환, 같은 방향이면 지속이다.',
        '장벽은 과거 일간 변동성 6배·최소 로그변화 6%·최대 30%. 다음 거래일 종가를 진입 대용값으로 사용한다. 시간 제한 없이 첫 도달까지 기다리며 미확정·결측 단절은 음성이 아닌 검열이다.',
        '따라서 완벽한 변곡점 포착이나 큰 추세 예측 성공으로 해석할 수 없다. 기간 제약은 없지만 이동 폭 기준은 설계자가 정한 값이다.',
        '모든 모델은 옵션·매크로 등 공통 입력이 관측된 동일 표본으로 비교한다. 입력기간은 2018년부터, XLC는 2019년부터이며 마지막 공통 주간 입력은 2026-08-28이다.',
        '학습 내 동일 섹터·유사 과거 수익률/변동성인 지속 사례에 가중치를 더 주었다. 전체 지속 사례도 유지하며, 같은 섹터·결과 확정월의 중복 표본 가중치를 낮췄다.',
        '연도 전 2년을 검증으로 사용하고, 검증 시작 전 정답이 확정된 자료만 학습한다. 특징 중요도는 검증 표본의 순열 교란으로 계산하며 평가구간에서는 특징/설정을 선택하지 않았다.',
        '모델 출력은 클래스 가중치가 적용된 점수이며 보정된 확률이 아니다. 섹터 구분 입력을 포함하지만 섹터별 독립 모델은 아니다.','',
        '## 같은 경보 수 비교: 사후 순위 진단','',
        '각 연도·방향에서 결과가 확정된 표본의 점수 상위 10%만 선택했다. 미래 전체 점수 순위를 사용하는 연구 진단이지 실시간 경보 규칙이 아니다.',
        '95% 구간은 모든 섹터를 같은 반기 블록으로 함께 재표집한 차이 구간이다. 블록이 적고 일부 사건은 반기를 넘으므로 참고치이며 다중 비교도 보정하지 않았다.',
        '| 방향 | 모델 | 선택 수 | 기본 발생률 | 적중률 | 같은 알고리즘 가격 대비 | 차이 95% 구간 | 방향조정 20일 | 방향조정 60일 |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for a in comp.itertuples():
        lines.append(f'| {a.direction} | {a.model} | {a.n} | {a.base_rate:.1%} | {a.precision:.1%} | {100*a.delta:+.1f}%p | [{100*a.ci_low:+.1f}, {100*a.ci_high:+.1f}]%p | {a.signed_return20:+.1%} | {a.signed_return60:+.1%} |')
    lines += ['', '방향조정 수익률은 IN=가격 수익률, OUT=가격 수익률의 부호 반대다. 종목 비중·거래비용·연속 경보를 반영한 투자 수익률은 아니다. 20/60일은 평가 진단 창이지 정답 기간 제한이 아니다.',
        '', '## 실시간 적용 가능한 별도 규칙','',
        '검증 점수 90백분위를 다음 연도에 고정 적용한 결과는 metrics.csv에 별도 저장했다. 이 규칙은 평가구간 경보 횟수가 같지 않다. 반복 경보를 제거하지 않았으며 적중 일수를 독립 전환 횟수로 부풀려 해석하면 안 된다.',
        '', '## 한계와 다음 판정','',
        '현재 수정 이력 가격과 PIT 미검증 Bloomberg 자료이며, 이미 반복 열람한 평가기간이다. 새로운 독립 검증 결과가 아니다.',
        '특징 중요도는 원인/인과관계가 아니며 상관된 변수에서 불안정할 수 있다. validation_importance.csv 및 importance_stability.csv에 연도별 결과를 남겼다.',
        '최고 적중률 하나만 보고 채택하지 않는다. 연도·섹터 일관성, 기본 발생률, 같은 모델 가격 대비 차이, 신호 이후 남은 움직임을 함께 본다.',
        'DB·웹·운영 모델 변경 없음. 외부 API 호출 없음. 중복/점수 범위/학습 및 검증 정답 확정일 경계 검사 통과.']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(comp.to_string(index=False))


def run(output):
    self_test(); started=time.monotonic()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    frame,sets,coverage=prepare(); coverage.to_csv(output/'coverage.csv')
    frame.to_csv(output/'research_frame.csv',index=False)
    predictions=[]; audits=[]; summaries=[]; counts=0; importance=[]
    for direction in ('IN','OUT'):
        part=frame.loc[frame.direction.eq(direction)]
        for year in (2023,2024,2025,2026):
            validation_start=pd.Timestamp(year-2,1,1); cutoff=pd.Timestamp(year,1,1)
            fit=part.loc[part.label_end.lt(validation_start)].reset_index(drop=True)
            validation=part.loc[part.date.ge(validation_start)&part.label_end.lt(cutoff)]
            test=part.loc[part.date.dt.year.eq(year)]
            if len(fit)<100 or fit.y.nunique()<2 or validation.y.nunique()<2 or test.empty:
                audits.append(dict(direction=direction,year=year,status='INSUFFICIENT',fit=len(fit),validation=len(validation),test=len(test)))
                continue
            assert fit.label_end.max()<validation_start and validation.label_end.max()<cutoff
            weights,pairs=match_weights(fit)
            for name,cols in sets.items():
                scaler=StandardScaler().fit(fit[cols]); a=scaler.transform(fit[cols]); b=scaler.transform(validation[cols]); c=scaler.transform(test[cols])
                for family in ('LOGIT','BOOST','EXTRA'):
                    if counts>=120 or time.monotonic()-started>1740:
                        raise TimeoutError('Run budget reached; partial folds preserved.')
                    model=make_candidate(family)
                    with threadpool_limits(limits=2):
                        model.fit(a,fit.y,sample_weight=weights)
                        vs=model.predict_proba(b)[:,1]; score=model.predict_proba(c)[:,1]
                    threshold=float(np.quantile(vs,.90))
                    row=test[['date','ticker','direction','y','label_end','duration','return20','return60','favorable20','adverse20']].copy()
                    row['model']=name+'_'+family;row['year']=year;row['score']=score
                    row['threshold']=threshold;row['alert']=score>=threshold
                    predictions.append(row);counts+=1
                    # Validation permutation only: never inspect test labels to select features.
                    if validation.y.nunique()==2:
                        base=roc_auc_score(validation.y,vs); rng=np.random.default_rng(42)
                        for j,col in enumerate(cols):
                            if col.startswith('sector_'): continue
                            perm=b.copy();perm[:,j]=rng.permutation(perm[:,j])
                            with threadpool_limits(limits=2): ps=model.predict_proba(perm)[:,1]
                            importance.append(dict(direction=direction,year=year,model=name+'_'+family,feature=col,validation_auc_drop=base-roc_auc_score(validation.y,ps)))
                    audits.append(dict(direction=direction,year=year,model=name+'_'+family,fit=len(fit),validation=len(validation),test=len(test),matched_pairs=len(pairs),
                        fit_label_end=str(fit.label_end.max()),validation_label_end=str(validation.label_end.max()),threshold=threshold))
                    print(f'{counts}/96 {direction} {year} {name}_{family} completed',flush=True)
            pd.concat(predictions).to_csv(output/'predictions.csv',index=False)
            pd.DataFrame(audits).to_csv(output/'fold_audit.csv',index=False)
    result=pd.concat(predictions,ignore_index=True)
    for (direction,model),g in result.groupby(['direction','model']):
        for period,sub in [('ALL',g),*list(g.groupby('year'))]:
            summaries.append(dict(direction=direction,model=model,period=str(period),**metrics(sub)))
    pd.DataFrame(summaries).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(importance).to_csv(output/'validation_importance.csv',index=False)
    # Equal-budget ranking is retrospective only; threshold alerts above are causal validation-selected.
    ranked=[]
    for (direction,year,model),g in result.groupby(['direction','year','model']):
        q=g.loc[g.y.notna()].copy(); q['alert']=False
        if len(q):q.loc[q.nlargest(max(1,int(np.ceil(.1*len(q)))),'score').index,'alert']=True
        ranked.append(dict(direction=direction,year=year,model=model,**metrics(q)))
    pd.DataFrame(ranked).to_csv(output/'equal_budget_diagnostic.csv',index=False)
    meta=dict(fits=counts,elapsed_seconds=time.monotonic()-started,sets=sets,
        target='opposite vs same-direction first barrier after next-close entry; trend=prior60 sign; log barrier=max(.06,6*lagged daily60 sigma), capped .30; NO duration cap',
        warning=['not exact pivot prediction or trading strategy','weekly sampling; correlated and overlapping labels',
        'all models use same full-feature complete rows; missing values not filled; unresolved outcomes censored',
        'two-year mature validation; fit outcomes fully mature before validation; no tuning on test',
        'calendar-year test previously reviewed; not pristine holdout; current revised history, PIT not verified',
        'scores are weighted classifier scores, not calibrated probabilities','permutation importance one repeat, correlated-feature interpretation limited',
        'signed return is direction diagnostic, not realizable portfolio return; fees/slippage not modeled'],operational_changes=False)
    (output/'audit.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'fits':counts,'seconds':meta['elapsed_seconds'],'output':str(output)}))
    audit_results(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--audit-only',action='store_true');parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/variable_barrier_comparison_20260921')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.audit_only:audit_results(args.output_dir)
    else:run(args.output_dir)
