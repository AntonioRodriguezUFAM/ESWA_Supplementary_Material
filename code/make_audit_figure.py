#!/usr/bin/env python3
"""make_audit_figure.py — regenerates Figure 1 (audit findings E1 and E3) and the intermediate aggregates from the
three archived mission recordings (metrics_*, dynamic_adaptation_trace_*, actions_* CSVs of the Family-G missions).
Without the recordings, the intermediate aggregates written here (e1_cap_medians.csv, e3_phase_counts.csv,
e1_temporal_check.csv) document what the figure shows. Usage: python3 make_audit_figure.py <recordings_dir> <out_dir>"""
import sys, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); caps = []; phases = []; temporal = []
for prof, c, mk in [('ORIGINAL', '#534AB7', 'o'), ('REVISED1', '#0F6E56', 's'), ('REVISED2', '#993C1D', '^')]:
    M = pd.read_csv(glob.glob(f'{D}/metrics_*{prof}*.csv')[0], usecols=['Timestamp', 'SoC_CPU1_Freq', 'SoC_CPU2_Freq', 'SoC_CPU3_Freq', 'SoC_CPU4_Freq'])
    ts = pd.to_datetime(M.Timestamp); M['el'] = (ts - ts.iloc[0]).dt.total_seconds()      # device clock, mission-relative
    M['f'] = M[['SoC_CPU1_Freq', 'SoC_CPU2_Freq', 'SoC_CPU3_Freq', 'SoC_CPU4_Freq']].mean(axis=1)
    T = pd.read_csv(glob.glob(f'{D}/dynamic_adaptation_trace_*{prof}*.csv')[0], usecols=['mission_elapsed_s', 'runtime_cpu_max_freq_khz']).sort_values('mission_elapsed_s')
    T = T[T.runtime_cpu_max_freq_khz > 0]
    J = pd.merge_asof(M.sort_values('el'), T, left_on='el', right_on='mission_elapsed_s', direction='backward').dropna(subset=['runtime_cpu_max_freq_khz', 'f'])
    g = J.groupby('runtime_cpu_max_freq_khz')['f'].agg(median='median', p5=lambda x: np.percentile(x, 5), p95=lambda x: np.percentile(x, 95), n='count').reset_index(); g['profile'] = prof; caps.append(g)
    off = {'ORIGINAL': -15, 'REVISED1': 0, 'REVISED2': 15}[prof]
    ax[0].errorbar(g.runtime_cpu_max_freq_khz / 1000 + off, g['median'], yerr=[g['median'] - g.p5, g.p95 - g['median']], fmt=mk, color=c, label=prof, capsize=3, ms=6, alpha=.85)
    # temporal check: holds >= 60 s at caps < 1479 MHz, records >= 30 s after the change
    ch = T[T.runtime_cpu_max_freq_khz.ne(T.runtime_cpu_max_freq_khz.shift())].reset_index(drop=True); ch['end'] = ch.mission_elapsed_s.shift(-1).fillna(T.mission_elapsed_s.max())
    for _, r in ch[(ch.end - ch.mission_elapsed_s >= 60) & (ch.runtime_cpu_max_freq_khz < 1479000)].iterrows():
        m = M[(M.el >= r.mission_elapsed_s + 30) & (M.el < r.end)]
        if len(m): temporal.append(dict(profile=prof, cap_khz=r.runtime_cpu_max_freq_khz, hold_s=r.end - r.mission_elapsed_s, records_after_30s=len(m), median_mhz=m.f.median(), min_mhz=m.f.min()))
ax[0].plot([0, 1500], [0, 1500], ls='--', color='grey', lw=0.8, label='requested = reported'); ax[0].set_xlim(0, 1550); ax[0].set_ylim(0, 1650)
ax[0].set_xlabel('requested CPU cap [MHz] (runtime state)'); ax[0].set_ylabel('reported core frequency [MHz], median (P5–P95)'); ax[0].set_title('(a) E1: requested cap vs reported frequency'); ax[0].legend(fontsize=9, loc='lower right')
for prof, c in [('REVISED1', '#0F6E56'), ('REVISED2', '#993C1D')]:
    A = pd.read_csv(glob.glob(f'{D}/actions_*{prof}*.csv')[0], usecols=['generation', 'prefer_gpu', 'runtime_enable_gpu'])
    A['phase'] = (np.arange(len(A)) * 0.8 // 1800).astype(int)     # 800-ms control period, action order
    g = A.groupby('phase').agg(preferred=('prefer_gpu', 'sum'), enabled=('runtime_enable_gpu', 'sum')).reset_index(); g['profile'] = prof; phases.append(g)
    ax[1].plot(g.phase * 0.5, g.preferred, marker='o', color=c, label=f'{prof}: GPU preferred'); ax[1].plot(g.phase * 0.5, g.enabled, marker='x', ls='--', color=c, label=f'{prof}: GPU enabled')
ax[1].set_xlabel('mission time [h] (phases of 1800 s, action order at 800 ms)'); ax[1].set_ylabel('action records per phase'); ax[1].set_title('(b) E3: GPU preferred vs GPU enabled (action records)'); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_audit_e1_e3.png', dpi=170)
pd.concat(caps).to_csv(f'{OUT}/e1_cap_medians.csv', index=False); pd.concat(phases).to_csv(f'{OUT}/e3_phase_counts.csv', index=False); pd.DataFrame(temporal).to_csv(f'{OUT}/e1_temporal_check_holds.csv', index=False)
print('wrote fig_audit_e1_e3.png, e1_cap_medians.csv, e3_phase_counts.csv, e1_temporal_check_holds.csv')
