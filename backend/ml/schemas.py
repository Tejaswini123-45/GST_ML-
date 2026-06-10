"""
Pydantic schemas for ML API responses.

All monetary values are in ₹ (Indian Rupees).
Risk scores are normalised to [0.0, 1.0].
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Per-invoice anomaly record ─────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    """Lightweight anomaly record — one per invoice in list responses."""
    invoice_id: str
    seller_gstin: str
    buyer_gstin: str
    invoice_value: float
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalised anomaly score 0→1")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    anomaly_reasons: list[str] = Field(default_factory=list)
    isolation_score: float = Field(..., description="Raw IsolationForest decision_function output")


class FeatureSnapshot(BaseModel):
    """Key engineered features attached to an AnomalyDetail."""
    invoice_value: float
    effective_tax_rate: float
    gstr1_filed: int
    gstr1_status: str
    days_to_file: float
    itc_eligible: float
    gstr2b_value_discrepancy: float
    pr_value_discrepancy: float
    pr_tax_discrepancy: float
    has_ewaybill: int
    invoice_value_zscore: float
    seller_filing_rate: float
    seller_late_rate: float
    seller_gstr3b_payment_rate: float
    seller_itc_block_rate: float
    seller_risk: str
    buyer_risk: str
    buyer_itc_block_rate: float


class ComplianceSnapshot(BaseModel):
    """Cross-collection compliance signals for a single invoice."""
    gstr1_present: bool
    gstr1_status: str
    gstr2b_present: bool
    itc_eligible: str
    gstr3b_payment_confirmed: bool
    ewaybill_present: bool


class AnomalyDetail(AnomalyRecord):
    """Full anomaly detail — returned by GET /ml/anomaly/{invoice_id}."""
    seller_name: str = "Unknown"
    buyer_name: str = "Unknown"
    invoice_date: str = ""
    features: FeatureSnapshot
    compliance: ComplianceSnapshot


# ── Portfolio-level summary ────────────────────────────────────────────────

class RiskDistribution(BaseModel):
    LOW: int = 0
    MEDIUM: int = 0
    HIGH: int = 0
    CRITICAL: int = 0


class MostFlaggedFeature(BaseModel):
    feature: str
    flag_count: int
    description: str


class RiskSummary(BaseModel):
    """Returned by GET /ml/risk-summary."""
    total_invoices: int
    model_trained: bool
    model_trained_at: Optional[str] = None
    contamination_rate: float
    risk_distribution: RiskDistribution
    top_anomalies: list[AnomalyRecord]
    most_flagged_features: list[MostFlaggedFeature]
    avg_risk_score: float
    total_invoice_value_at_risk: float  # sum of values with HIGH/CRITICAL invoices
    total_blocked_itc: float            # from GSTR2B


# ── Retraining ─────────────────────────────────────────────────────────────

class RetrainResult(BaseModel):
    """Returned by POST /ml/retrain."""
    status: str
    invoices_trained_on: int
    features_used: int
    contamination_rate: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    trained_at: str
    duration_seconds: float
