#!/usr/bin/env bash
# Reproduces what this package can reproduce without the mission recordings: Tables 3-5, Figure 2, and the Figure 1
# aggregates. Rerunning the replay (code/wema_arms.py) and regenerating Figure 1 (code/make_audit_figure.py) need the
# recordings held by the authors.
set -e; cd "$(dirname "$0")"
echo "== Tables 3-5 and the Figure 1 aggregates"; python3 code/reproduce_tables.py
echo "== Figure 2"; python3 code/make_figure.py
