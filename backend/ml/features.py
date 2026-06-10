"""
GST Feature Engineering Pipeline — Phase 1.

Reads all 7 MongoDB collections and produces a rich 26-feature matrix,
one row per invoice, suitable for Isolation Forest anomaly detection.

Feature Design Rationale
------------------------
Each feature encodes a distinct GST compliance signal:

CATEGORY A — Invoice-level value signals
  invoice_value          : Raw declared amount. Large outliers = potential fraud.
  log_invoice_value      : Log₁₊ transform compresses heavy tail distribution.
  effective_tax_rate     : tax / value. GST is 5–28%; extreme values = manipulation.

CATEGORY B — Filing compliance signals (from GSTR1)
  gstr1_filed            : Binary. If the seller did not file, buyer's ITC is at risk.
  gstr1_status_encoded   : FILED=0, LATE=1, PENDING=2, MISSING=3 (ordinal severity).
  days_to_file           : Days between invoice date and GSTR1 filing. Long lag = suspicious.

CATEGORY C — ITC and reconciliation signals (from GSTR2B + Purchase Register)
  itc_eligible           : 1=YES, 0=NO/MISSING. Direct ITC risk indicator.
  gstr2b_value_discrepancy: |GSTR2B_value − Invoice_value| / Invoice_value.
  pr_value_discrepancy   : |Claimed_value − Invoice_value| / Invoice_value.
  pr_tax_discrepancy     : |Claimed_tax − GSTR1_tax| / GSTR1_tax.

CATEGORY D — Transport compliance (from EWayBill)
  has_ewaybill           : Binary. Missing EWB for >₹50K goods = legal violation.
  ewb_value_discrepancy  : |EWB_value − Invoice_value| / Invoice_value.

CATEGORY E — Seller aggregate behaviour features
  seller_invoice_count       : Volume. Very high or very low counts are anomalous.
  seller_avg_value           : Reference baseline for value comparisons.
  seller_value_std           : Spread. Very low std + one spike = invoice stuffing.
  invoice_value_zscore       : (value − seller_mean) / seller_std. Key outlier signal.
  seller_filing_rate         : Filed GSTR1 / Total invoices. Low rate = serial violator.
  seller_late_rate           : LATE / Filed. Chronic late filer = compliance risk.
  seller_gstr3b_payment_rate : Payment_Confirmed_Y / Total GSTR3B rows. Tax evasion.
  seller_itc_block_rate      : Blocked ITC / Total GSTR2B rows from seller.
  seller_risk_encoded        : HIGH=2, MEDIUM=1, LOW=0.

CATEGORY F — Buyer aggregate behaviour features
  buyer_invoice_count        : Buyer volume signal.
  buyer_avg_value            : Reference point for buyer-side analysis.
  buyer_itc_block_rate       : How much of buyer's ITC is blocked overall.
  buyer_risk_encoded         : HIGH=2, MEDIUM=1, LOW=0.

CATEGORY G — Temporal features
  invoice_day     : Day of month. End-of-month clustering is a known fraud pattern.
  invoice_month   : Month of year. Seasonal anomalies detectable with enough data.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ml.features")

# ── Ordered feature columns fed to the model ──────────────────────────────
FEATURE_COLS: list[str] = [
    # A — Value
    "invoice_value",
    "log_invoice_value",
    "effective_tax_rate",
    # B — Filing compliance
    "gstr1_filed",
    "gstr1_status_encoded",
    "days_to_file",
    # C — ITC / reconciliation
    "itc_eligible",
    "gstr2b_value_discrepancy",
    "pr_value_discrepancy",
    "pr_tax_discrepancy",
    # D — Transport
    "has_ewaybill",
    "ewb_value_discrepancy",
    # E — Seller aggregates
    "seller_invoice_count",
    "seller_avg_value",
    "seller_value_std",
    "invoice_value_zscore",
    "seller_filing_rate",
    "seller_late_rate",
    "seller_gstr3b_payment_rate",
    "seller_itc_block_rate",
    "seller_risk_encoded",
    # F — Buyer aggregates
    "buyer_invoice_count",
    "buyer_avg_value",
    "buyer_itc_block_rate",
    "buyer_risk_encoded",
    # G — Temporal
    "invoice_day",
    "invoice_month",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert CSV-sourced string/numeric to float, returns default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _risk_to_int(cat: str) -> int:
    """Encode risk category string → ordinal integer."""
    return {"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(str(cat).upper(), 0)


def _parse_date(date_str: Any) -> datetime | None:
    """Parse YYYY-MM-DD string to datetime. Returns None on failure."""
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _collection_to_df(docs: list[dict]) -> pd.DataFrame:
    """Convert a list of MongoDB dicts to a clean DataFrame (no internal fields)."""
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame([
        {k: v for k, v in d.items() if not k.startswith("_")}
        for d in docs
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

async def build_feature_matrix(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all GST collections and return:
        features_df  — [N_invoices × 28 features], indexed by Invoice_ID
        metadata_df  — display-friendly columns (names, dates, statuses)

    Raises ValueError if the Invoices collection is empty.
    """
    # ── 1. Fetch collections ──────────────────────────────────────────────
    inv_docs = await db.Invoices.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    g1_docs  = await db.GSTR1.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    g2b_docs = await db.GSTR2B.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    g3b_docs = await db.GSTR3B.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    ewb_docs = await db.EWayBill.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    pr_docs  = await db.Purchase_Register.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)
    tp_docs  = await db.Taxpayers.find({}, {"_id": 0, "_source_file": 0, "_source_row": 0}).to_list(length=10000)

    inv_df = _collection_to_df(inv_docs)
    g1_df  = _collection_to_df(g1_docs)
    g2b_df = _collection_to_df(g2b_docs)
    g3b_df = _collection_to_df(g3b_docs)
    ewb_df = _collection_to_df(ewb_docs)
    pr_df  = _collection_to_df(pr_docs)
    tp_df  = _collection_to_df(tp_docs)

    if inv_df.empty:
        raise ValueError("Invoices collection is empty — run data ingestion first.")

    logger.info(
        "Loaded: Invoices=%d G1=%d G2B=%d G3B=%d EWB=%d PR=%d TP=%d",
        len(inv_df), len(g1_df), len(g2b_df), len(g3b_df),
        len(ewb_df), len(pr_df), len(tp_df),
    )

    # ── 2. Numeric coercions ──────────────────────────────────────────────
    inv_df["Value"]       = inv_df["Value"].apply(_safe_float)
    if not g1_df.empty:
        g1_df["Tax"]      = g1_df["Tax"].apply(_safe_float)
        g1_df["Value"]    = g1_df["Value"].apply(_safe_float)
    if not g2b_df.empty:
        g2b_df["Tax"]     = g2b_df["Tax"].apply(_safe_float)
        g2b_df["Value"]   = g2b_df["Value"].apply(_safe_float)
    if not g3b_df.empty:
        g3b_df["Tax_Paid"] = g3b_df["Tax_Paid"].apply(_safe_float)
    if not ewb_df.empty:
        ewb_df["Value"]   = ewb_df["Value"].apply(_safe_float)
    if not pr_df.empty:
        pr_df["Value_Claimed"] = pr_df["Value_Claimed"].apply(_safe_float)
        pr_df["Tax_Claimed"]   = pr_df["Tax_Claimed"].apply(_safe_float)

    # ── 3. Taxpayer lookup maps ───────────────────────────────────────────
    if not tp_df.empty:
        tp_map: dict[str, dict] = {
            row["GSTIN"]: {"Name": row.get("Name", ""), "Risk_Category": row.get("Risk_Category", "LOW")}
            for _, row in tp_df.iterrows()
        }
    else:
        tp_map = {}

    # ── 4. Seller-level GSTR3B aggregate (period-level, not invoice-level) ─
    seller_g3b: dict[str, dict] = {}
    if not g3b_df.empty:
        for seller, grp in g3b_df.groupby("Seller_GSTIN"):
            total = len(grp)
            confirmed = (grp["Payment_Confirmed"].str.upper() == "Y").sum()
            seller_g3b[seller] = {
                "payment_rate": confirmed / total if total > 0 else 0.0,
                "total_records": total,
            }

    # ── 5. Seller-level GSTR2B aggregate ──────────────────────────────────
    seller_g2b_block: dict[str, float] = {}
    if not g2b_df.empty:
        for seller, grp in g2b_df.groupby("Seller_GSTIN"):
            total = len(grp)
            blocked = (grp["ITC_Eligible"].str.upper() == "NO").sum()
            seller_g2b_block[seller] = blocked / total if total > 0 else 0.0

    # ── 6. Seller-level invoice aggregate (from Invoices collection) ───────
    seller_stats: dict[str, dict] = {}
    for seller, grp in inv_df.groupby("Seller_GSTIN"):
        values = grp["Value"].values
        seller_stats[seller] = {
            "count":  len(values),
            "mean":   float(np.mean(values)),
            "std":    float(np.std(values)) if len(values) > 1 else 0.0,
        }

    # Seller GSTR1 filing rates
    seller_g1_stats: dict[str, dict] = {}
    for seller, grp in inv_df.groupby("Seller_GSTIN"):
        total_inv = len(grp)
        inv_ids   = set(grp["Invoice_ID"])
        if not g1_df.empty:
            seller_g1 = g1_df[g1_df["Seller_GSTIN"] == seller]
            filed     = seller_g1[seller_g1["Status"].str.upper() == "FILED"]
            late      = seller_g1[seller_g1["Status"].str.upper() == "LATE"]
            g1_count  = len(seller_g1[seller_g1["Invoice_ID"].isin(inv_ids)])
        else:
            filed = late = pd.DataFrame()
            g1_count = 0
        seller_g1_stats[seller] = {
            "filing_rate": g1_count / total_inv if total_inv > 0 else 0.0,
            "late_rate":   len(late) / g1_count if g1_count > 0 else 0.0,
        }

    # ── 7. Buyer-level aggregate ───────────────────────────────────────────
    buyer_stats: dict[str, dict] = {}
    for buyer, grp in inv_df.groupby("Buyer_GSTIN"):
        values = grp["Value"].values
        buyer_stats[buyer] = {
            "count": len(values),
            "mean":  float(np.mean(values)),
        }

    buyer_g2b_block: dict[str, float] = {}
    if not g2b_df.empty:
        for buyer, grp in g2b_df.groupby("Buyer_GSTIN"):
            total = len(grp)
            blocked = (grp["ITC_Eligible"].str.upper() == "NO").sum()
            buyer_g2b_block[buyer] = blocked / total if total > 0 else 0.0

    # ── 8. Index lookup dicts ──────────────────────────────────────────────
    g1_index:  dict[str, pd.Series] = {}
    if not g1_df.empty:
        for _, row in g1_df.iterrows():
            g1_index[row["Invoice_ID"]] = row

    g2b_index: dict[str, pd.Series] = {}
    if not g2b_df.empty:
        for _, row in g2b_df.iterrows():
            g2b_index[row["Invoice_ID"]] = row

    ewb_index: dict[str, pd.Series] = {}
    if not ewb_df.empty:
        for _, row in ewb_df.iterrows():
            ewb_index[row["Invoice_ID"]] = row

    pr_index:  dict[str, pd.Series] = {}
    if not pr_df.empty:
        for _, row in pr_df.iterrows():
            pr_index[row["Invoice_ID"]] = row

    # ── 9. Build per-invoice feature rows ─────────────────────────────────
    feature_rows:  list[dict] = []
    metadata_rows: list[dict] = []

    for _, inv in inv_df.iterrows():
        inv_id     = inv["Invoice_ID"]
        seller_gstin = inv["Seller_GSTIN"]
        buyer_gstin  = inv["Buyer_GSTIN"]
        inv_value    = _safe_float(inv["Value"])

        # ── GSTR1 signals ──
        g1 = g1_index.get(inv_id)
        if g1 is not None:
            gstr1_filed = 1
            status      = str(g1.get("Status", "MISSING")).upper()
            status_enc  = {"FILED": 0, "LATE": 1, "PENDING": 2}.get(status, 3)
            g1_tax      = _safe_float(g1.get("Tax"))
            g1_value    = _safe_float(g1.get("Value"))
            filing_date = _parse_date(g1.get("Filing_Date"))
        else:
            gstr1_filed = 0
            status      = "MISSING"
            status_enc  = 3
            g1_tax      = 0.0
            g1_value    = inv_value
            filing_date = None

        inv_date = _parse_date(inv.get("Invoice_Date"))
        if filing_date and inv_date:
            days_to_file = (filing_date - inv_date).days
        else:
            days_to_file = 0.0

        # Effective tax rate (capped to handle edge cases)
        eff_tax_rate = (g1_tax / inv_value) if inv_value > 0 and g1_tax > 0 else 0.0
        eff_tax_rate = min(eff_tax_rate, 1.0)  # cap at 100%

        # ── GSTR2B signals ──
        g2b = g2b_index.get(inv_id)
        if g2b is not None:
            itc_raw    = str(g2b.get("ITC_Eligible", "")).upper()
            itc_elig   = 1.0 if itc_raw == "YES" else 0.0
            g2b_value  = _safe_float(g2b.get("Value"))
            g2b_val_disc = abs(g2b_value - inv_value) / inv_value if inv_value > 0 else 0.0
            g2b_val_disc = min(g2b_val_disc, 2.0)  # cap at 200%
        else:
            itc_elig     = 0.0   # treat missing GSTR2B as ineligible
            g2b_value    = 0.0
            g2b_val_disc = 0.0
            itc_raw      = "MISSING"

        # ── Purchase Register signals ──
        pr = pr_index.get(inv_id)
        if pr is not None:
            pr_value   = _safe_float(pr.get("Value_Claimed"))
            pr_tax     = _safe_float(pr.get("Tax_Claimed"))
            pr_val_disc = abs(pr_value - inv_value) / inv_value if inv_value > 0 else 0.0
            pr_val_disc = min(pr_val_disc, 2.0)
            pr_tax_disc = abs(pr_tax - g1_tax) / g1_tax if g1_tax > 0 else 0.0
            pr_tax_disc = min(pr_tax_disc, 2.0)
        else:
            pr_value    = 0.0
            pr_tax      = 0.0
            pr_val_disc = 0.0
            pr_tax_disc = 0.0

        # ── EWayBill signals ──
        ewb = ewb_index.get(inv_id)
        if ewb is not None:
            has_ewb    = 1
            ewb_value  = _safe_float(ewb.get("Value"))
            ewb_val_disc = abs(ewb_value - inv_value) / inv_value if inv_value > 0 else 0.0
            ewb_val_disc = min(ewb_val_disc, 2.0)
        else:
            has_ewb      = 0
            ewb_value    = 0.0
            ewb_val_disc = 0.0

        # ── Seller aggregates ──
        s_stats = seller_stats.get(seller_gstin, {"count": 1, "mean": inv_value, "std": 0.0})
        s_mean  = s_stats["mean"]
        s_std   = s_stats["std"]
        s_count = s_stats["count"]
        inv_zscore = (inv_value - s_mean) / s_std if s_std > 0 else 0.0
        inv_zscore = float(np.clip(inv_zscore, -5, 5))  # clip extreme z-scores

        s_g1   = seller_g1_stats.get(seller_gstin, {"filing_rate": 0.0, "late_rate": 0.0})
        s_g3b  = seller_g3b.get(seller_gstin, {"payment_rate": 0.0})
        s_itc_block = seller_g2b_block.get(seller_gstin, 0.0)
        s_risk = tp_map.get(seller_gstin, {}).get("Risk_Category", "LOW")
        s_risk_enc = _risk_to_int(s_risk)

        # GSTR3B payment confirmed for this invoice's period
        g3b_confirmed = 1.0 if s_g3b["payment_rate"] >= 0.5 else 0.0

        # ── Buyer aggregates ──
        b_stats = buyer_stats.get(buyer_gstin, {"count": 1, "mean": inv_value})
        b_count = b_stats["count"]
        b_mean  = b_stats["mean"]
        b_itc_block = buyer_g2b_block.get(buyer_gstin, 0.0)
        b_risk  = tp_map.get(buyer_gstin, {}).get("Risk_Category", "LOW")
        b_risk_enc = _risk_to_int(b_risk)

        # ── Temporal features ──
        if inv_date:
            inv_day   = inv_date.day
            inv_month = inv_date.month
        else:
            inv_day   = 15  # midpoint default
            inv_month = 1

        # ── Assemble feature dict ──
        feature_rows.append({
            "invoice_id":                inv_id,
            # A
            "invoice_value":             inv_value,
            "log_invoice_value":         float(np.log1p(inv_value)),
            "effective_tax_rate":        eff_tax_rate,
            # B
            "gstr1_filed":               gstr1_filed,
            "gstr1_status_encoded":      status_enc,
            "days_to_file":              float(days_to_file),
            # C
            "itc_eligible":              itc_elig,
            "gstr2b_value_discrepancy":  g2b_val_disc,
            "pr_value_discrepancy":      pr_val_disc,
            "pr_tax_discrepancy":        pr_tax_disc,
            # D
            "has_ewaybill":              has_ewb,
            "ewb_value_discrepancy":     ewb_val_disc,
            # E
            "seller_invoice_count":      float(s_count),
            "seller_avg_value":          s_mean,
            "seller_value_std":          s_std,
            "invoice_value_zscore":      inv_zscore,
            "seller_filing_rate":        s_g1["filing_rate"],
            "seller_late_rate":          s_g1["late_rate"],
            "seller_gstr3b_payment_rate": s_g3b["payment_rate"],
            "seller_itc_block_rate":     s_itc_block,
            "seller_risk_encoded":       float(s_risk_enc),
            # F
            "buyer_invoice_count":       float(b_count),
            "buyer_avg_value":           b_mean,
            "buyer_itc_block_rate":      b_itc_block,
            "buyer_risk_encoded":        float(b_risk_enc),
            # G
            "invoice_day":               float(inv_day),
            "invoice_month":             float(inv_month),
        })

        # ── Metadata (for display, not used in model) ──
        metadata_rows.append({
            "invoice_id":     inv_id,
            "seller_gstin":   seller_gstin,
            "buyer_gstin":    buyer_gstin,
            "seller_name":    tp_map.get(seller_gstin, {}).get("Name", "Unknown"),
            "buyer_name":     tp_map.get(buyer_gstin, {}).get("Name", "Unknown"),
            "invoice_date":   str(inv.get("Invoice_Date", "")),
            "invoice_value":  inv_value,
            "gstr1_status":   status,
            "itc_eligible":   itc_raw,
            "has_ewaybill":   bool(has_ewb),
            "seller_risk":    s_risk,
            "buyer_risk":     b_risk,
            "g1_tax":         g1_tax,
            "g2b_value":      g2b_value,
            "pr_value":       pr_value,
        })

    features_df  = pd.DataFrame(feature_rows).set_index("invoice_id")
    metadata_df  = pd.DataFrame(metadata_rows).set_index("invoice_id")

    # Ensure feature columns are exactly in canonical order
    features_df = features_df[FEATURE_COLS]

    logger.info(
        "Feature matrix built: %d invoices × %d features",
        len(features_df), len(FEATURE_COLS),
    )
    return features_df, metadata_df
