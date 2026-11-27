"""Pre-written development analysis (pre-registration §6). REFUSED until approved.

    python scripts/residual_analysis.py --episodes evaluation.csv --failed failed.json --out DIR

Fixed-sequence gatekeeping, the fixed claim wording, Holm secondaries and the safety counts,
computed by ``farc.residual.analysis.development_analysis``. No episodes are run here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from farc.residual.analysis import development_analysis, substitute_failed
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
    res = development_analysis(
        df,
        plan["bootstrap_resamples"],
        plan["bootstrap_seed"],
        ni_margin=plan["nominal_noninferiority"],
        practical=plan["practical_threshold"],
        require_sign_consistency=plan["per_network_sign_consistency"],
    )
    res["inference_note"] = "conditional on the 3 trained networks per arm (amendment R-A1, STAT-2)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res["safety"].to_csv(out / "harmed_rescued.csv", index=False)
    res["safety"] = "harmed_rescued.csv"
    (out / "analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
