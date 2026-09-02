"""
Evaluatiescript — rapporteert de vier metrieken uit de meetlat van het ML-plan.

Globale R² is een misleidende maat voor dit probleem: hij wordt gedomineerd
door het onderscheid tussen rampzalige en oke layouts, terwijl de optimizer
juist góéde layouts onderling moet kunnen rangschikken. Dit script zet daarom
vier getallen naast elkaar:

  1. Gevalideerde top-1   — beste layout na echte headless validatie (px)
  2. Kalibratiefout       — voorspelt het model te optimistisch aan de zoekgrens?
  3. Spearman top-deciel  — rangschikking binnen de 10% beste layouts
  4. R² schone holdout    — out-of-fold, gededupliceerd, seed-gewogen

Gebruik:
    python3 evaluate.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from train_surrogate import extract_features_from_list, top_decile_spearman

HERE = Path(__file__).parent
ROOT = HERE.parent

OOF_FILE   = ROOT / "surrogate-oof.json"
MODEL_FILE = ROOT / "surrogate_model.pkl"
VAL_FILE   = ROOT / "validation-results.json"


def _features(tables, feat_len):
    """Zelfde feature-pad als de optimizer, zodat de voorspelling vergelijkbaar is."""
    var = [t for t in tables if t["size"] != "custom"]
    var.sort(key=lambda t: (-t["w"], t["x"], t["y"]))
    row = extract_features_from_list(var)
    row = row + [0.0] * max(0, feat_len - len(row))
    return np.array([row], dtype=np.float32)


def _line(label, value, target, ok, note=""):
    mark = "OK " if ok else "-- "
    print(f"  {mark}{label:<26} {value:>16}   doel {target:<12} {note}")


def report():
    print("\nMeetlat — ML-verbeterplan fase 2")
    print("=" * 78)

    # ── 1 + 2: gevalideerde layouts ──────────────────────────────────────────
    if VAL_FILE.exists():
        val   = json.load(open(VAL_FILE))
        saved = joblib.load(MODEL_FILE)
        model, feat_len = saved["model"], saved["feat_len"]

        best_actual = min(v["actual_dist"] for v in val)
        _line("Gevalideerde top-1", f"{best_actual:,.0f} px", "< 240.000 px",
              best_actual < 240_000, f"(n={len(val)})")

        errs = []
        for v in val:
            if "tables" not in v:
                continue
            pred = float(model.predict(_features(v["tables"], feat_len))[0])
            errs.append((pred - v["actual_dist"]) / v["actual_dist"])
        if errs:
            calib = float(np.mean(errs)) * 100
            _line("Kalibratiefout (nu)", f"{calib:+.1f}%", "< 3%",
                  abs(calib) < 3, f"(n={len(errs)})")

        old = [(v["predicted_dist"] - v["actual_dist"]) / v["actual_dist"] for v in val]
        print(f"     ter vergelijking: het opgeslagen model voorspelde destijds "
              f"{np.mean(old)*100:+.1f}%")
    else:
        print(f"  (geen {VAL_FILE.name} — draai eerst validate_headless.js)")

    # ── 3 + 4: out-of-fold ───────────────────────────────────────────────────
    if OOF_FILE.exists():
        oof    = json.load(open(OOF_FILE))
        y_true = np.array(oof["y_true"])
        y_pred = np.array(oof["y_pred"])

        from sklearn.metrics import mean_absolute_error, r2_score
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rho  = float(spearmanr(y_true, y_pred).statistic)
        rtop = top_decile_spearman(y_true, y_pred)

        _line("Spearman top-deciel", f"{rtop:.3f}", "> 0,70", rtop > 0.70)
        _line("R² schone holdout", f"{r2:.3f}", "> 0,80", r2 > 0.80,
              f"({oof['model']}, {len(y_true)} layouts)")
        print(f"\n     MAE {mae:,.0f} px   |   Spearman totaal {rho:.3f}")
        print(f"     Ruisplafond uit de per-seed spreiding: R² ~ 0,99")
    else:
        print(f"  (geen {OOF_FILE.name} — draai eerst train_surrogate.py)")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    report()
