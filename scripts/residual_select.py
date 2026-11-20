"""Pick alpha* from selection-block episodes (pre-registration §5). REFUSED until approved.

    python scripts/residual_select.py --episodes selection.csv --failed failed.json --out DIR

``selection.csv`` has the columns described in ``farc.residual.analysis``; ``failed.json``
lists runs stopped by the rollback rule as [[config, alpha, network_seed], ...] (they enter as
C). No episodes are run here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from farc.residual.analysis import select_alpha, substitute_failed
from farc.residual.training import CONFIG_PATH, check_approved, load_plan_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", required=True)
    p.add_argument("--failed", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = load_plan_config(CONFIG_PATH)
    check_approved(plan)
    df = pd.read_csv(a.episodes)
    failed = json.loads(Path(a.failed).read_text()) if a.failed else []
    df = substitute_failed(df, [tuple(f) for f in failed])
    sel = select_alpha(df, plan["alphas"], plan["nominal_noninferiority"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel.table.to_csv(out / "selection_table.csv", index=False)
    (out / "selection.json").write_text(
        json.dumps({"alpha_star": sel.alpha, "null_result": sel.alpha is None}, indent=2)
    )
    print(sel.table.to_string(index=False), "\nalpha* =", sel.alpha)


if __name__ == "__main__":
    main()
