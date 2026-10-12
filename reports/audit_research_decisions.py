"""Descriptive ledger of saved corrected predictive comparisons; no refitting."""
import pandas as pd
from research_leading_improvements import OUT


def run():
    names = ['A_tests','B_tests','E_tests','F_tests','J_tests','K_tests','L_tests',
             'J_rank_tests','K_rank_tests','L_rank_tests','P_predictive_tests',
             'Q_tests','R_tests','S_tests','T_tests']
    rows=[]
    for name in names:
        d=pd.read_csv(OUT/f'{name}.csv')
        if name=='A_tests':d=d[d.period.eq('historical_confirmation')]
        effect=next(c for c in ['brier_improvement','mse_improvement','delta_ic','improvement'] if c in d)
        corrected=next(c for c in d if c.startswith('p_holm'))
        assert d[corrected].notna().all() and d[corrected].between(0,1).all()
        assert d[effect].notna().all()
        rows.append({'file':name+'.csv','comparisons':len(d),
                     'positive_effects':int(d[effect].gt(0).sum()),
                     'positive_and_family_p_lt_05':int((d[effect].gt(0)&d[corrected].lt(.05)).sum()),
                     'negative_and_family_p_lt_05':int((d[effect].lt(0)&d[corrected].lt(.05)).sum()),
                     'effect_column':effect,'corrected_p_column':corrected})
    result=pd.DataFrame(rows);result.to_csv(OUT/'U_comparison_inventory.csv',index=False)
    print(result.to_string(index=False));print(result.select_dtypes('number').sum().to_dict())


if __name__=='__main__':run()
