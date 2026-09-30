#!/usr/bin/env python3
"""wema_arms.py — decision-level evaluation of the DEPLOYED workload memory (W-EMA).

Reimplements the production SurrogateModel bucket estimator exactly as specified in the
architecture manuscript (five frequency anchors; key k(a)=1000*b_f+100*g+10*round(4*split)+b_c;
alpha = 0.60 on first observation, 0.30 thereafter; exact-bucket hit, else inverse-distance
interpolation over up to k=3 populated neighbours in the same workload; analytic prior otherwise),
and replays it over a collector event log in two arms:

    retain : memory persists across workload returns          (what the system actually does)
    reset  : the returning workload's bucket map is cleared at its entry (matched cold start)

Predictions are issued BEFORE the outcome is released, scored against the recorded outcome, then
admitted. This is a reimplementation of the deployed estimator, disclosed as such; the recorded
W-EMA predictions in the log are reported alongside as a fidelity check of the reimplementation.
"""
import json, re, sys, math, csv

from collections import defaultdict

ANCHORS = [102000, 460800, 921600, 1190400, 1479000]
TARGETS = ['power_w', 'fps', 'latency_ms', 'temp_c']

NO_FREQ_KEY = False
def bucket_key(freq_khz, gpu, split, conc):
    bf = 0 if NO_FREQ_KEY else min(range(5), key=lambda i: abs(ANCHORS[i] - freq_khz))
    bc = max(1, min(4, int(conc)))
    s = max(0.0, min(1.0, split))
    return 1000*bf + 100*int(gpu) + 10*int(round(4*s)) + bc

def prior(gpu):
    return {'power_w': 6.5 if gpu else 5.0, 'fps': 30.0, 'latency_ms': 5.0 if gpu else 10.0, 'temp_c': 45.0}

class WEMA:
    """Per-workload bucket maps; shared support counter per bucket, as in the deployed model."""
    def __init__(self, alpha=0.30, alpha0=0.60, k=3):
        self.banks = defaultdict(dict); self.alpha = alpha; self.alpha0 = alpha0; self.k = k
    def reset_workload(self, w):
        self.banks[w] = {}
    def predict(self, w, key, gpu):
        bank = self.banks[w]
        if key in bank:
            mu, n = bank[key]
            return dict(mu), 'exact', n
        if bank:
            neigh = sorted(bank.items(), key=lambda kv: abs(kv[0] - key))[:self.k]
            num = {t: 0.0 for t in TARGETS}; den = 0.0
            for kk, (mu, n) in neigh:
                wgt = 1.0 / (1e-3 + abs(kk - key))
                den += wgt
                for t in TARGETS: num[t] += wgt * mu[t]
            return {t: num[t]/den for t in TARGETS}, 'knn', len(neigh)
        return prior(gpu), 'prior', 0
    def observe(self, w, key, gpu, outcome, valid):
        bank = self.banks[w]
        if key not in bank:
            bank[key] = (prior(gpu), 0)
        mu, n = bank[key]; mu = dict(mu)
        a = self.alpha0 if n == 0 else self.alpha
        any_valid = False
        for t in TARGETS:
            if valid.get(t):
                mu[t] = (1 - a) * mu[t] + a * outcome[t]; any_valid = True
        bank[key] = (mu, n + (1 if any_valid else 0))

def load_queries(path):
    q = {}; order = []; 
    for line in open(path):
        try: e = json.loads(re.sub(r'(?<=[:,\[])(-?nan|-?inf)(?=[,\]}])', 'null', line.rsplit(',"crc32"', 1)[0] + '}'))
        except Exception: continue
        ev = e.get('event'); qid = e.get('query_id', 0)
        if not qid: continue
        r = q.setdefault(qid, {})
        if ev == 'context_sample':
            r['wl'] = e.get('workload'); r['wep'] = e.get('workload_epoch'); r['ae'] = e.get('action_epoch'); order.append(qid)
        elif ev == 'prediction_issue':
            v = e['prediction'].get('value') or []
            r['logged'] = {t: (v[i] if i < len(v) and v[i] is not None else float('nan')) for i, t in enumerate(TARGETS)}
        elif ev == 'action_apply':
            c = e.get('config_requested', {})
            r['cfg'] = (c.get('freq_cap_khz', 0), c.get('gpu_enable', 0), c.get('split', 0.0), c.get('concurrency', 1))
        elif ev == 'outcome_available':
            o = e['outcome']; val = o.get('value') or []; vd = o.get('valid') or []
            r['out'] = {t: (val[i] if i < len(val) and val[i] is not None else float('nan')) for i, t in enumerate(TARGETS)}
            r['valid'] = {t: bool(vd[i]) and isinstance(val[i], (int, float)) and math.isfinite(val[i] or float('nan')) for i, t in enumerate(TARGETS)}
    return q, order

def run(path, arm):
    q, order = load_queries(path)
    m = WEMA(); visits = defaultdict(int); cur = None; rows = []
    pos = 0
    for qid in order:
        r = q[qid]
        if 'cfg' not in r or 'wl' not in r: continue
        if r['wep'] != cur:
            cur = r['wep']; visits[r['wl']] += 1; pos = 0
            if arm == 'reset' and visits[r['wl']] > 1:
                m.reset_workload(r['wl'])
        key = bucket_key(*r['cfg'])
        pred, src, sup = m.predict(r['wl'], key, r['cfg'][1])
        if 'out' in r:
            err = {t: (abs(pred[t] - r['out'][t]) if r['valid'].get(t) else float('nan')) for t in TARGETS}
            lg = r.get('logged', {})
            lerr = {t: (abs(lg[t] - r['out'][t]) if r['valid'].get(t) and t in lg and math.isfinite(lg[t]) else float('nan')) for t in TARGETS}
            rows.append(dict(qid=qid, wl=r['wl'], wep=r['wep'], visit=visits[r['wl']], pos=pos, src=src, support=sup,
                             err_power=err['power_w'], err_fps=err['fps'], logged_err_power=lerr['power_w'],
                             pred_power=pred['power_w'], out_power=r['out']['power_w'], valid_power=int(bool(r['valid'].get('power_w')))))
            m.observe(r['wl'], key, r['cfg'][1], r['out'], r['valid'])
        pos += 1
    return rows

if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == '--no-freq-key': globals()['NO_FREQ_KEY'] = True
    suffix = '_nofreq' if NO_FREQ_KEY else ''
    for arm in ('retain', 'reset'):
        rows = run(path, arm)
        with open(f'{out}/wema_{arm}{suffix}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f'{arm}: {len(rows)} scored rows')
