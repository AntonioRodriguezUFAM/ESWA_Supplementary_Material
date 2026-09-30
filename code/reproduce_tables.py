#!/usr/bin/env python3
"""reproduce_tables.py — prints Table 3 (per-mission replay errors with the logged production reference), Table 4
(sensitivity to the frequency key dimension), Table 5 (ablation: retention x key partition), the leave-one-mission-out
range, the exact two-sided sign test and the descriptive bootstrap, from the result files."""
import os, math, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
C = pd.read_csv(os.path.join(R, 'wema_all_cells.csv'))
PM = C.groupby(['source', 'mission'])[['reset', 'retain', 'logged_err_power']].mean(); PM['diff'] = PM.retain - PM.reset
print('== Table 3: early-return power MAE (W), mean over 8 returning cells per mission (deployed key)\n', PM.round(4).to_string())
v = PM['diff'].values; rng = np.random.default_rng(5); boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
loo = [np.delete(v, i).mean() for i in range(len(v))]; neg = int((v < 0).sum()); n = len(v)
p = 2 * min(sum(math.comb(n, k) for k in range(0, neg + 1)), sum(math.comb(n, k) for k in range(neg, n + 1))) / 2 ** n
print(f'\n   mean {v.mean():+.4f} W | leave-one-out [{min(loo):+.4f}, {max(loo):+.4f}] | negative {neg}/{n} (two-sided sign p={min(1,p):.3f}) | cells {(C.K_retain_minus_reset<0).sum()}/{len(C)} | descriptive bootstrap 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}] | reduction {100*(1-PM.retain.mean()/PM.reset.mean()):.1f}%')
print(f'   fidelity: mean |replay per-cell MAE - production per-cell MAE| = {(C.retain - C.logged_err_power).abs().mean():.4f} W')
S = pd.read_csv(os.path.join(R, 'wema_freqkey_sensitivity.csv'))
print('\n== Table 4 / Table 5: key variants (rows) x cleared/retained (columns)')
for var, name in [('freqkey', 'with frequency partition (as deployed)'), ('nofreq', 'without frequency partition')]:
    X = S[S.variant == var]; M = X.groupby('mission')[['reset', 'retain']].mean(); d = (M.retain - M.reset).values; l = [np.delete(d, i).mean() for i in range(len(d))]
    print(f'   {name:40s} cleared {M.reset.mean():.4f}  retained {M.retain.mean():.4f}  effect {d.mean():+.4f}  LOO [{min(l):+.4f}, {max(l):+.4f}]  missions {(d<0).sum()}/{len(d)}  cells {(X["diff"]<0).sum()}/{len(X)}  reduction {100*(1-M.retain.mean()/M.reset.mean()):.1f}%')
A = S.groupby(['variant'])[['reset', 'retain']].mean()
print(f'   partition effect: cleared {A.loc["freqkey","reset"]-A.loc["nofreq","reset"]:+.4f} W, retained {A.loc["freqkey","retain"]-A.loc["nofreq","retain"]:+.4f} W')
print('\n== Figure 1 aggregates (from the recordings; reproduced here from the stored intermediates)')
print(pd.read_csv(os.path.join(R, 'e1_cap_medians.csv')).pivot_table(index='runtime_cpu_max_freq_khz', columns='profile', values='median').round(0).to_string())
print(pd.read_csv(os.path.join(R, 'e1_temporal_check.csv')).to_string(index=False))
