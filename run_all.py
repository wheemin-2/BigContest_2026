"""
Run the whole UBI pipeline: model fit -> damage/priority -> dashboard table -> dashboard page.

    python run_all.py [path/to/card_data.csv] [--steps=1,2,3,4]

Without a path, the card data is read from preprocessed_data/SH_card_data1_preprocessed.csv.
Scripts are in UBI/, inputs in preprocessed_data/, results in outputs/.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UBI = ROOT / "UBI"
DEFAULT_CARD = ROOT / "preprocessed_data" / "SH_card_data1_preprocessed.csv"

STEPS = [  # (number, script, takes card data path)
    ("1", "ubi_model.py",                True),    # model fit, vulnerable industries
    ("2", "ubi_damage_priority.py",      True),    # damage size, intervals, priority
    ("3", "ubi_dashboard_precompute.py", True),    # per day forecast damage (hybrid baseline)
    ("4", "ubi_build_dashboard.py",      False),   # data.json, index.html
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    card = Path(args[0]).resolve() if args else DEFAULT_CARD
    if not card.exists():
        sys.exit(f"card data not found: {card}")
    run = set(opts.get("steps", "1,2,3,4").split(","))
    for num, script, needs_card in STEPS:
        if num in run:
            print(f"\n=== step {num}: {script}", flush=True)
            subprocess.run([sys.executable, str(UBI / script)] + ([str(card)] if needs_card else []), check=True)
    print("\nresults ->", ROOT / "outputs")


if __name__ == "__main__":
    main()
