"""
ML API routes — Invoice Anomaly Detection via Isolation Forest.

Endpoints
---------
GET  /ml/anomalies              list all scored invoices, sorted by risk
GET  /ml/anomaly/{invoice_id}   full detail for a single invoice
GET  /ml/risk-summary           portfolio-level ML risk intelligence
POST /ml/retrain                force retrain on current MongoDB data

Authentication
--------------
All routes require the X-API-Key header (same as other protected routes).
The auth dependency is applied at router-include time in main.py.

Auto-training
-------------
If the model has never been trained (no saved file on disk), the first
request to any endpoint automatically trains the model.  This may take
~2 seconds depending on data size.  Subsequent calls are instant.
"""

import time
from fastapi import APIRouter, HTTPException, Query

from backend.database import get_mongo_db
from backend.ml.model import get_detector, get_or_train_detector, train_detector
from backend.ml.schemas import (
    AnomalyRecord,
    AnomalyDetail,
    ComplianceSnapshot,
    FeatureSnapshot,
    MostFlaggedFeature,
    RetrainResult,
    RiskDistribution,
    RiskSummary,
)
from backend.utils.logger import get_logger

router = APIRouter(prefix="/ml", tags=["ML Risk Intelligence"])
logger = get_logger(__name__)

# ── Risk level ordering helper ─────────────────────────────────────────────
_RISK_ORDER = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

_RISK_INT_TO_STR = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}


def _to_anomaly_record(r: dict) -> AnomalyRecord:
    return AnomalyRecord(
        invoice_id      = r["invoice_id"],
        seller_gstin    = r["seller_gstin"],
        buyer_gstin     = r["buyer_gstin"],
        invoice_value   = r["invoice_value"],
        risk_score      = r["risk_score"],
        risk_level      = r["risk_level"],
        anomaly_reasons = r["anomaly_reasons"],
        isolation_score = r["isolation_score"],
    )


def _to_anomaly_detail(r: dict) -> AnomalyDetail:
    feat = r.get("features", {})
    comp = r.get("compliance", {})

    seller_risk_enc = int(feat.get("seller_risk_encoded", 0))
    buyer_risk_enc  = int(feat.get("buyer_risk_encoded", 0))

    feature_snap = FeatureSnapshot(
        invoice_value             = float(feat.get("invoice_value", 0)),
        effective_tax_rate        = float(feat.get("effective_tax_rate", 0)),
        gstr1_filed               = int(feat.get("gstr1_filed", 0)),
        gstr1_status              = comp.get("gstr1_status", "MISSING"),
        days_to_file              = float(feat.get("days_to_file", 0)),
        itc_eligible              = float(feat.get("itc_eligible", 0)),
        gstr2b_value_discrepancy  = float(feat.get("gstr2b_value_discrepancy", 0)),
        pr_value_discrepancy      = float(feat.get("pr_value_discrepancy", 0)),
        pr_tax_discrepancy        = float(feat.get("pr_tax_discrepancy", 0)),
        has_ewaybill              = int(feat.get("has_ewaybill", 0)),
        invoice_value_zscore      = float(feat.get("invoice_value_zscore", 0)),
        seller_filing_rate        = float(feat.get("seller_filing_rate", 0)),
        seller_late_rate          = float(feat.get("seller_late_rate", 0)),
        seller_gstr3b_payment_rate= float(feat.get("seller_gstr3b_payment_rate", 0)),
        seller_itc_block_rate     = float(feat.get("seller_itc_block_rate", 0)),
        seller_risk               = _RISK_INT_TO_STR.get(seller_risk_enc, "LOW"),
        buyer_risk                = _RISK_INT_TO_STR.get(buyer_risk_enc, "LOW"),
        buyer_itc_block_rate      = float(feat.get("buyer_itc_block_rate", 0)),
    )

    compliance_snap = ComplianceSnapshot(
        gstr1_present             = bool(comp.get("gstr1_present", False)),
        gstr1_status              = comp.get("gstr1_status", "MISSING"),
        gstr2b_present            = bool(comp.get("gstr2b_present", False)),
        itc_eligible              = comp.get("itc_eligible", "MISSING"),
        gstr3b_payment_confirmed  = bool(comp.get("gstr3b_payment_confirmed", False)),
        ewaybill_present          = bool(comp.get("ewaybill_present", False)),
    )

    return AnomalyDetail(
        invoice_id      = r["invoice_id"],
        seller_gstin    = r["seller_gstin"],
        buyer_gstin     = r["buyer_gstin"],
        invoice_value   = r["invoice_value"],
        risk_score      = r["risk_score"],
        risk_level      = r["risk_level"],
        anomaly_reasons = r["anomaly_reasons"],
        isolation_score = r["isolation_score"],
        seller_name     = r.get("seller_name", "Unknown"),
        buyer_name      = r.get("buyer_name", "Unknown"),
        invoice_date    = r.get("invoice_date", ""),
        features        = feature_snap,
        compliance      = compliance_snap,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /ml/anomalies
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/anomalies", response_model=list[AnomalyRecord])
async def list_anomalies(
    min_risk_level: str = Query(
        "LOW",
        description="Minimum risk level to include. One of: LOW | MEDIUM | HIGH | CRITICAL",
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    seller_gstin: str | None = Query(None, description="Filter by seller GSTIN"),
    buyer_gstin:  str | None = Query(None, description="Filter by buyer GSTIN"),
):
    """
    Return all invoices scored by the ML model, sorted by risk score (highest first).

    Auto-trains the model if it has not yet been fitted.
    Use ``min_risk_level=MEDIUM`` to focus on genuinely suspicious invoices.
    """
    if min_risk_level not in _RISK_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid min_risk_level '{min_risk_level}'. Must be LOW, MEDIUM, HIGH, or CRITICAL.",
        )

    det = await get_or_train_detector()
    results = det.get_all_results()

    # Apply filters
    min_ord = _RISK_ORDER[min_risk_level]
    results = [
        r for r in results
        if _RISK_ORDER.get(r["risk_level"], 0) >= min_ord
        and (seller_gstin is None or r["seller_gstin"] == seller_gstin)
        and (buyer_gstin  is None or r["buyer_gstin"]  == buyer_gstin)
    ]

    # Sort by risk score descending
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    return [_to_anomaly_record(r) for r in results[:limit]]


# ═══════════════════════════════════════════════════════════════════════════
# GET /ml/anomaly/{invoice_id}
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/anomaly/{invoice_id}", response_model=AnomalyDetail)
async def get_anomaly_detail(invoice_id: str):
    """
    Full ML anomaly detail for a single invoice:
    risk score, risk level, all engineered features, compliance snapshot,
    and ranked explainable reasons.

    Returns 404 if the invoice was not found in the training data.
    """
    det = await get_or_train_detector()
    result = det.get_result(invoice_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Invoice '{invoice_id}' not found in ML results. "
                "Ensure data has been ingested and the model has been trained."
            ),
        )

    return _to_anomaly_detail(result)


# ═══════════════════════════════════════════════════════════════════════════
# GET /ml/risk-summary
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/risk-summary", response_model=RiskSummary)
async def risk_summary():
    """
    Portfolio-level ML risk intelligence dashboard feed.

    Returns:
      - overall risk distribution (CRITICAL/HIGH/MEDIUM/LOW counts)
      - top 10 most anomalous invoices
      - most frequently flagged features across the dataset
      - total invoice value in HIGH/CRITICAL risk category
      - aggregate blocked ITC exposure
    """
    det = await get_or_train_detector()
    info = det.get_model_info()
    results = det.get_all_results()

    if not results:
        return RiskSummary(
            total_invoices              = 0,
            model_trained               = False,
            contamination_rate          = det.contamination,
            risk_distribution           = RiskDistribution(),
            top_anomalies               = [],
            most_flagged_features       = [],
            avg_risk_score              = 0.0,
            total_invoice_value_at_risk = 0.0,
            total_blocked_itc           = 0.0,
        )

    counts = det.get_risk_counts()

    # Top 10 by risk score
    sorted_results = sorted(results, key=lambda x: x["risk_score"], reverse=True)
    top_10 = [_to_anomaly_record(r) for r in sorted_results[:10]]

    # Average risk score
    avg_score = round(sum(r["risk_score"] for r in results) / len(results), 4)

    # Total invoice value in HIGH + CRITICAL
    at_risk_value = sum(
        r["invoice_value"]
        for r in results
        if r["risk_level"] in ("HIGH", "CRITICAL")
    )

    # Most frequently flagged features → MostFlaggedFeature list
    flag_counts = det._feature_flag_counts
    # Friendly descriptions for each flag label
    _FLAG_DESCRIPTIONS = {
        "GSTR1 Not Filed":       "Seller did not file GSTR1 outward supply return",
        "ITC Blocked":           "Input Tax Credit marked ineligible in GSTR2B",
        "Missing EWayBill":      "No E-Way Bill — missing transport compliance document",
        "High Value Z-Score":    "Invoice value is a statistical outlier vs seller average",
        "GSTR1 Late/Pending":    "GSTR1 filed late or still pending",
        "Low Seller Filing Rate":"Seller has poor GSTR1 filing history",
        "Poor Payment Confirm":  "Seller's GSTR3B tax payment poorly confirmed",
        "High ITC Block Rate":   "Most of seller's invoices have ITC blocked",
        "Value-GSTR2B Mismatch": "Value discrepancy between Invoice and GSTR2B auto-draft",
        "PR Over-Claim":         "Purchase Register claims more than Invoice value",
        "Seller HIGH Risk":      "Seller is categorised as HIGH risk by GSTN",
    }
    most_flagged = [
        MostFlaggedFeature(
            feature     = label,
            flag_count  = cnt,
            description = _FLAG_DESCRIPTIONS.get(label, label),
        )
        for label, cnt in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
        if cnt > 0
    ][:8]

    return RiskSummary(
        total_invoices              = len(results),
        model_trained               = info["trained"],
        model_trained_at            = info["trained_at"],
        contamination_rate          = info["contamination"],
        risk_distribution           = RiskDistribution(**counts),
        top_anomalies               = top_10,
        most_flagged_features       = most_flagged,
        avg_risk_score              = avg_score,
        total_invoice_value_at_risk = round(at_risk_value, 2),
        total_blocked_itc           = det._total_blocked_itc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# POST /ml/retrain
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/retrain", response_model=RetrainResult)
async def retrain_model():
    """
    Force a full retrain of the Isolation Forest on the current MongoDB data.

    Use this endpoint after ingesting new CSV files to refresh ML scores.
    Retraining takes ~1–3 seconds and automatically persists the new model to disk.
    """
    t0 = time.perf_counter()
    try:
        db  = get_mongo_db()
        det = await train_detector(db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Retrain failed")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")

    duration = round(time.perf_counter() - t0, 2)
    counts   = det.get_risk_counts()
    info     = det.get_model_info()

    return RetrainResult(
        status               = "success",
        invoices_trained_on  = info["n_invoices"],
        features_used        = info["n_features"],
        contamination_rate   = info["contamination"],
        critical_count       = counts["CRITICAL"],
        high_count           = counts["HIGH"],
        medium_count         = counts["MEDIUM"],
        low_count            = counts["LOW"],
        trained_at           = info["trained_at"] or "",
        duration_seconds     = duration,
    )
