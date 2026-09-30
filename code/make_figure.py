#!/usr/bin/env python3
"""Regenerates Figure 2 and prints the Table 1 interval from wema_all_cells.csv."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'results', 'wema_all_cells.csv'))
PM = C.groupby(['source','mission'])[['reset','retain']].mean(); PM['K'] = PM.retain - PM.reset
rng = np.random.default_rng(5); v = PM.K.values
boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
print(f'mean {v.mean():+.4f} W, 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], '
      f'missions improved {(v<0).sum()}/{len(v)}, cells improved {(C.K_retain_minus_reset<0).sum()}/{len(C)}, '
      f'reduction {100*(1-PM.retain.mean()/PM.reset.mean()):.1f}%')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(PM)); w = 0.36
ax[0].bar(x-w/2, PM.reset.values, w, label='memory cleared at return', color='#b03a2e')
ax[0].bar(x+w/2, PM.retain.values, w, label='memory retained (deployed)', color='#1f618d')
ax[0].set_xticks(x); ax[0].set_xticklabels([m.replace('_live_18Sep',' (live)') for _, m in PM.index], fontsize=8)
ax[0].set_ylabel('early-visit power MAE [W]'); ax[0].set_title('(a) per mission, mean of 8 returning cells'); ax[0].legend(fontsize=9)
ax[1].plot([0,1], [C.reset.values, C.retain.values], color='grey', alpha=.45, lw=.9)
ax[1].scatter(np.zeros(len(C)), C.reset.values, s=22, color='#b03a2e', label='cleared', zorder=3)
ax[1].scatter(np.ones(len(C)), C.retain.values, s=22, color='#1f618d', label='retained', zorder=3)
ax[1].set_xlim(-.3,1.3); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['cleared','retained'])
ax[1].set_ylabel('early-visit power MAE [W]'); ax[1].set_title('(b) 32 returning cells, paired'); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'results', 'fig_retention.png'), dpi=170); print('wrote fig_retention.png')
