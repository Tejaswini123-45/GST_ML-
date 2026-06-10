"""
Explainable AI engine for GST Anomaly Detection.

For each flagged invoice, compares its feature vector against the
population distribution and generates ranked, human-readable explanations.

Design Principle
----------------
Every explanation must be:
  1. Specific     — include actual numbers, not generic phrases
  2. Causal       — explain WHY the feature is suspicious
  3. Actionable   — give the CFO a clear next step implied by the finding
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ml.explainer")


# ── Threshold constants ────────────────────────────────────────────────────
ZSCORE_CRITICAL  = 2.5   # standard deviations
ZSCORE_HIGH      = 1.5
TAX_RATE_NORMAL  = 0.18  # 18% standard GST rate
TAX_RATE_TOL     = 0.06  # ±6% acceptable deviation
DISCREPANCY_HIGH = 0.10  # 10% value mismatch threshold
LATE_RATE_HIGH   = 0.50  # 50% late-filing threshold


# ═══════════════════════════════════════════════════════════════════════════
# Population statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_population_stats(features_df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute mean, std, and percentiles for every feature column.
    Used as the reference distribution for per-invoice explanation.
    """
    stats: dict[str, dict] = {}
    for col in features_df.columns:
        series = features_df[col].dropna()
        if len(series) == 0:
            stats[col] = {"mean": 0, "std": 1, "p25": 0, "p50": 0, "p75": 0, "p90": 0}
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std":  float(series.std()) if series.std() > 0 else 1.0,
            "p25":  float(np.percentile(series, 25)),
            "p50":  float(np.percentile(series, 50)),
            "p75":  float(np.percentile(series, 75)),
            "p90":  float(np.percentile(series, 90)),
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Per-feature explanation generators
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_inr(value: float) -> str:
    """Format a rupee value with comma separators."""
    return f"₹{value:,.0f}"


def _explain_invoice_value_zscore(feat: dict, stats: dict) -> str | None:
    z = feat.get("invoice_value_zscore", 0.0)
    seller_mean = feat.get("seller_avg_value", 0.0)
    inv_value   = feat.get("invoice_value", 0.0)
    if abs(z) < ZSCORE_HIGH:
        return None
    ratio = inv_value / seller_mean if seller_mean > 0 else 0
    direction = "higher" if z > 0 else "lower"
    severity  = "critically" if abs(z) >= ZSCORE_CRITICAL else "significantly"
    return (
        f"Invoice value {_fmt_inr(inv_value)} is {severity} {direction} than seller's "
        f"average of {_fmt_inr(seller_mean)} "
        f"(z-score: {z:+.2f}, ratio: {ratio:.1f}x) — potential invoice stuffing or value inflation."
    )


def _explain_effective_tax_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("effective_tax_rate", 0.0)
    gstr1_filed = feat.get("gstr1_filed", 0)
    if gstr1_filed == 0:
        return None  # GSTR1 not filed — separate explanation covers this
    if rate == 0.0:
        return None
    deviation = abs(rate - TAX_RATE_NORMAL)
    if deviation <= TAX_RATE_TOL:
        return None
    pct = rate * 100
    if rate < TAX_RATE_NORMAL - TAX_RATE_TOL:
        return (
            f"Effective tax rate is unusually low at {pct:.1f}% "
            f"(expected ~{TAX_RATE_NORMAL*100:.0f}% GST) — "
            "potential under-reporting of tax liability."
        )
    else:
        return (
            f"Effective tax rate is unusually high at {pct:.1f}% "
            f"(expected ~{TAX_RATE_NORMAL*100:.0f}% GST) — "
            "check for rate mis-classification."
        )


def _explain_gstr1_not_filed(feat: dict, stats: dict) -> str | None:
    if feat.get("gstr1_filed", 1) == 0:
        return (
            "GSTR1 not filed by seller for this invoice — "
            "ITC claim is invalid until supplier files. Hold payment."
        )
    return None


def _explain_gstr1_late(feat: dict, stats: dict) -> str | None:
    status_enc = feat.get("gstr1_status_encoded", 0)
    days = feat.get("days_to_file", 0)
    if status_enc == 1:  # LATE
        return f"GSTR1 was filed late ({int(days)} days after invoice date). Late penalty applicable; ITC availability delayed."
    if status_enc == 2:  # PENDING
        return "GSTR1 filing status is PENDING — ITC cannot be claimed until filing is completed."
    return None


def _explain_days_to_file(feat: dict, stats: dict) -> str | None:
    days = feat.get("days_to_file", 0)
    if feat.get("gstr1_filed", 0) == 0:
        return None
    pop_mean = stats.get("days_to_file", {}).get("p90", 90)
    if days > pop_mean * 1.5 and days > 60:
        return (
            f"GSTR1 filing took {int(days)} days after invoice date — "
            "unusually long lag may indicate backdated invoice creation."
        )
    return None


def _explain_itc_blocked(feat: dict, stats: dict) -> str | None:
    elig = feat.get("itc_eligible", 1.0)
    if elig < 0.5:
        return (
            "ITC marked INELIGIBLE in GSTR2B — "
            "government has blocked the Input Tax Credit on this invoice. "
            "Do not treat as deductible."
        )
    return None


def _explain_gstr2b_discrepancy(feat: dict, stats: dict) -> str | None:
    disc = feat.get("gstr2b_value_discrepancy", 0.0)
    inv_val = feat.get("invoice_value", 0.0)
    if disc <= DISCREPANCY_HIGH:
        return None
    return (
        f"GSTR2B value differs from Invoice value by {disc*100:.1f}% "
        f"({_fmt_inr(inv_val)} declared vs different amount auto-populated in GSTR2B) — "
        "reconciliation required before ITC claim."
    )


def _explain_pr_discrepancy(feat: dict, stats: dict) -> str | None:
    disc = feat.get("pr_value_discrepancy", 0.0)
    if disc <= DISCREPANCY_HIGH:
        return None
    return (
        f"Purchase Register claimed value differs from Invoice by {disc*100:.1f}% — "
        "potential over-claiming of ITC. Reconcile before filing."
    )


def _explain_pr_tax_discrepancy(feat: dict, stats: dict) -> str | None:
    disc = feat.get("pr_tax_discrepancy", 0.0)
    if disc <= DISCREPANCY_HIGH:
        return None
    return (
        f"Tax claimed in Purchase Register differs from GSTR1 tax by {disc*100:.1f}% — "
        "tax amount mismatch between buyer's claim and seller's declaration."
    )


def _explain_missing_ewaybill(feat: dict, stats: dict) -> str | None:
    if feat.get("has_ewaybill", 1) == 0:
        inv_val = feat.get("invoice_value", 0.0)
        if inv_val >= 50000:
            return (
                f"E-Way Bill missing for invoice of {_fmt_inr(inv_val)} — "
                "mandatory for goods movement above ₹50,000. "
                "Possible fictitious transaction or transport compliance gap."
            )
        return (
            "E-Way Bill absent — no documentary evidence of physical goods movement."
        )
    return None


def _explain_seller_filing_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("seller_filing_rate", 1.0)
    if rate >= 0.80:
        return None
    pct = rate * 100
    return (
        f"Seller's GSTR1 filing rate is only {pct:.0f}% across all invoices — "
        "high-risk supplier with chronic non-compliance history."
    )


def _explain_seller_late_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("seller_late_rate", 0.0)
    if rate < LATE_RATE_HIGH:
        return None
    pct = rate * 100
    return (
        f"Seller files {pct:.0f}% of returns late — "
        "pattern of delayed compliance increases ITC reversal risk."
    )


def _explain_seller_payment_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("seller_gstr3b_payment_rate", 1.0)
    if rate >= 0.70:
        return None
    pct = rate * 100
    return (
        f"Seller has only {pct:.0f}% GSTR3B payment confirmation rate — "
        "seller collects GST from buyers but has poor record of remitting to government."
    )


def _explain_seller_itc_block_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("seller_itc_block_rate", 0.0)
    if rate < 0.40:
        return None
    pct = rate * 100
    return (
        f"Seller has {pct:.0f}% ITC block rate across their invoices — "
        "majority of transactions with this supplier have been flagged by GSTN."
    )


def _explain_seller_high_risk(feat: dict, stats: dict) -> str | None:
    enc = feat.get("seller_risk_encoded", 0)
    if enc >= 2:
        return (
            "Seller is categorised as HIGH risk — "
            "associated with non-compliance patterns or flagged by tax authorities."
        )
    return None


def _explain_buyer_itc_block_rate(feat: dict, stats: dict) -> str | None:
    rate = feat.get("buyer_itc_block_rate", 0.0)
    if rate < 0.40:
        return None
    pct = rate * 100
    return (
        f"Buyer's overall ITC block rate is {pct:.0f}% — "
        "buyer is frequently transacting with non-compliant suppliers."
    )


# ── Ordered list of explanation generators ────────────────────────────────
_GENERATORS = [
    _explain_invoice_value_zscore,
    _explain_effective_tax_rate,
    _explain_gstr1_not_filed,
    _explain_gstr1_late,
    _explain_days_to_file,
    _explain_itc_blocked,
    _explain_gstr2b_discrepancy,
    _explain_pr_discrepancy,
    _explain_pr_tax_discrepancy,
    _explain_missing_ewaybill,
    _explain_seller_filing_rate,
    _explain_seller_late_rate,
    _explain_seller_payment_rate,
    _explain_seller_itc_block_rate,
    _explain_seller_high_risk,
    _explain_buyer_itc_block_rate,
]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(
    feat: dict,
    pop_stats: dict[str, dict],
    max_reasons: int = 5,
) -> list[str]:
    """
    Given a feature dict for one invoice and the population statistics,
    produce a ranked list of human-readable anomaly reasons.

    Reasons are ordered from highest to lowest severity.
    Returns ["No anomaly signals detected."] when invoice looks clean.
    """
    reasons: list[str] = []
    for gen in _GENERATORS:
        try:
            result = gen(feat, pop_stats)
            if result:
                reasons.append(result)
        except Exception as exc:
            logger.debug("Explainer generator %s failed: %s", gen.__name__, exc)

    if not reasons:
        return ["No significant anomaly signals detected — invoice appears compliant."]

    return reasons[:max_reasons]


def rank_features_by_anomaly(
    feat: dict,
    pop_stats: dict[str, dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Return the top-N features with the largest z-score deviations from population.
    Used for feature importance visualization on the dashboard.
    """
    deviations = []
    for col, pstat in pop_stats.items():
        val = feat.get(col)
        if val is None:
            continue
        mean = pstat.get("mean", 0)
        std  = pstat.get("std", 1) or 1
        z    = abs((val - mean) / std)
        deviations.append({"feature": col, "z_score": round(z, 3), "value": val, "pop_mean": mean})

    deviations.sort(key=lambda x: x["z_score"], reverse=True)
    return deviations[:top_n]
