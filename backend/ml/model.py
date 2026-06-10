"""
GSTAnomalyDetector — Isolation Forest wrapper with full persistence.

Pipeline:
    SimpleImputer (median) → StandardScaler → IsolationForest

Scoring convention:
    IsolationForest.decision_function() returns POSITIVE for normal observations
    and NEGATIVE for anomalies.  We invert and min-max normalise to [0, 1]:

        risk_score = (−raw_score − min) / (max − min)

    Higher risk_score → more anomalous.

Risk categories:
    CRITICAL  ≥ 0.75      (~top 10% most anomalous)
    HIGH      ≥ 0.50
    MEDIUM    ≥ 0.30
    LOW       <  0.30

Singleton lifecycle:
    get_detector()          — return singleton (may not yet be fitted)
    get_or_train_detector() — auto-train on first call if not fitted
    train_detector()        — explicit (re)train; saves to disk

Persistence:
    backend/ml_store/detector.joblib   (sklearn objects + full result cache)
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml.explainer import compute_population_stats, generate_explanation
from backend.ml.features import FEATURE_COLS, build_feature_matrix

logger = logging.getLogger("ml.model")

# ── Storage location ───────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_store" / "detector.joblib"

# ── Risk thresholds ────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"CRITICAL": 0.75, "HIGH": 0.50, "MEDIUM": 0.30}

_RISK_INT_TO_STR = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}


# ═══════════════════════════════════════════════════════════════════════════
# Detector class
# ═══════════════════════════════════════════════════════════════════════════

class GSTAnomalyDetector:
    """
    Isolation Forest anomaly detector for GST invoices.

    Usage
    -----
    det = GSTAnomalyDetector()
    det.fit(features_df, metadata_df)   # CPU-bound — call via asyncio.to_thread
    det.save()

    result = det.get_result("INV-25")   # dict with risk_score, reasons, …
    """

    def __init__(
        self,
        contamination: float = 0.15,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

        self._preprocessor: Pipeline | None = None
        self._iso_forest:   IsolationForest | None = None
        self._pop_stats:    dict[str, dict] = {}
        self._trained_at:   str | None = None
        self._n_trained:    int = 0

        # invoice_id → full result dict (cached after fit)
        self._invoice_results: dict[str, dict] = {}
        # aggregate counters (cached after fit)
        self._feature_flag_counts: dict[str, int] = {}
        self._total_blocked_itc:   float = 0.0

    # ── Queries ─────────────────────────────────────────────────────────────

    def is_fitted(self) -> bool:
        return self._iso_forest is not None and self._n_trained > 0

    def get_all_results(self) -> list[dict]:
        return list(self._invoice_results.values())

    def get_result(self, invoice_id: str) -> dict | None:
        return self._invoice_results.get(invoice_id)

    def get_model_info(self) -> dict:
        return {
            "trained":       self.is_fitted(),
            "trained_at":    self._trained_at,
            "n_invoices":    self._n_trained,
            "n_features":    len(FEATURE_COLS),
            "contamination": self.contamination,
            "n_estimators":  self.n_estimators,
        }

    def get_risk_counts(self) -> dict[str, int]:
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in self._invoice_results.values():
            counts[r["risk_level"]] = counts.get(r["risk_level"], 0) + 1
        return counts

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """
        Fit preprocessor + IsolationForest and cache per-invoice results.
        This method is CPU-bound — call it via ``asyncio.to_thread``.
        """
        X = features_df[FEATURE_COLS].values.astype(float)

        # Step 1: preprocessing
        self._preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        X_proc = self._preprocessor.fit_transform(X)

        # Step 2: Isolation Forest
        self._iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._iso_forest.fit(X_proc)

        # Step 3: score every invoice
        raw_scores  = self._iso_forest.decision_function(X_proc)
        risk_scores = self._normalise(raw_scores)

        # Step 4: population statistics for explainability
        self._pop_stats = compute_population_stats(features_df)

        # Step 5: build per-invoice result cache
        self._invoice_results = {}
        flag_counts: dict[str, int] = {}
        total_blocked = 0.0

        for i, (inv_id, feat_row) in enumerate(features_df.iterrows()):
            feat_dict  = feat_row.to_dict()
            risk_score = float(risk_scores[i])
            iso_score  = float(raw_scores[i])
            risk_level = self._categorise(risk_score)
            reasons    = generate_explanation(feat_dict, self._pop_stats)

            if inv_id in metadata_df.index:
                meta = metadata_df.loc[inv_id]
            else:
                meta = pd.Series(dtype=object)

            # Accumulate feature-level flag counts
            self._update_flag_counts(feat_dict, flag_counts)

            # Accumulate blocked ITC
            if feat_dict.get("itc_eligible", 1) < 0.5:
                total_blocked += feat_dict.get("invoice_value", 0)

            self._invoice_results[inv_id] = {
                "invoice_id":      inv_id,
                "seller_gstin":    str(meta.get("seller_gstin", "")),
                "buyer_gstin":     str(meta.get("buyer_gstin", "")),
                "seller_name":     str(meta.get("seller_name", "Unknown")),
                "buyer_name":      str(meta.get("buyer_name", "Unknown")),
                "invoice_date":    str(meta.get("invoice_date", "")),
                "invoice_value":   float(feat_dict.get("invoice_value", 0)),
                "risk_score":      round(risk_score, 4),
                "risk_level":      risk_level,
                "isolation_score": round(iso_score, 4),
                "anomaly_reasons": reasons,
                "features":        feat_dict,
                "compliance": {
                    "gstr1_present":             bool(feat_dict.get("gstr1_filed", 0)),
                    "gstr1_status":              str(meta.get("gstr1_status", "MISSING")),
                    "gstr2b_present":            feat_dict.get("itc_eligible", -1) >= 0,
                    "itc_eligible":              str(meta.get("itc_eligible", "MISSING")),
                    "gstr3b_payment_confirmed":  feat_dict.get("seller_gstr3b_payment_rate", 0) >= 0.5,
                    "ewaybill_present":          bool(feat_dict.get("has_ewaybill", 0)),
                },
            }

        self._feature_flag_counts = flag_counts
        self._total_blocked_itc   = round(total_blocked, 2)
        self._trained_at = datetime.now(timezone.utc).isoformat()
        self._n_trained  = len(features_df)

        counts = self.get_risk_counts()
        logger.info(
            "Trained on %d invoices — CRITICAL:%d HIGH:%d MEDIUM:%d LOW:%d",
            self._n_trained, counts["CRITICAL"], counts["HIGH"],
            counts["MEDIUM"], counts["LOW"],
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalise(raw: np.ndarray) -> np.ndarray:
        """Invert and min-max normalise to [0, 1]. Higher = more anomalous."""
        inv = -raw
        lo, hi = inv.min(), inv.max()
        return np.zeros_like(inv) if hi == lo else (inv - lo) / (hi - lo)

    @staticmethod
    def _categorise(score: float) -> str:
        if score >= RISK_THRESHOLDS["CRITICAL"]: return "CRITICAL"
        if score >= RISK_THRESHOLDS["HIGH"]:     return "HIGH"
        if score >= RISK_THRESHOLDS["MEDIUM"]:   return "MEDIUM"
        return "LOW"

    @staticmethod
    def _update_flag_counts(feat: dict, counts: dict) -> None:
        """Track which compliance signals fired across the dataset."""
        checks = {
            "GSTR1 Not Filed":          feat.get("gstr1_filed", 1) == 0,
            "ITC Blocked":              feat.get("itc_eligible", 1) < 0.5,
            "Missing EWayBill":         feat.get("has_ewaybill", 1) == 0,
            "High Value Z-Score":       abs(feat.get("invoice_value_zscore", 0)) >= 1.5,
            "GSTR1 Late/Pending":       feat.get("gstr1_status_encoded", 0) >= 1,
            "Low Seller Filing Rate":   feat.get("seller_filing_rate", 1) < 0.8,
            "Poor Payment Confirm":     feat.get("seller_gstr3b_payment_rate", 1) < 0.7,
            "High ITC Block Rate":      feat.get("seller_itc_block_rate", 0) >= 0.4,
            "Value-GSTR2B Mismatch":    feat.get("gstr2b_value_discrepancy", 0) >= 0.10,
            "PR Over-Claim":            feat.get("pr_value_discrepancy", 0) >= 0.10,
            "Seller HIGH Risk":         feat.get("seller_risk_encoded", 0) >= 2,
        }
        for label, triggered in checks.items():
            if triggered:
                counts[label] = counts.get(label, 0) + 1

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        """Serialise the fitted model and all cached results to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "preprocessor":       self._preprocessor,
            "iso_forest":         self._iso_forest,
            "pop_stats":          self._pop_stats,
            "trained_at":         self._trained_at,
            "n_trained":          self._n_trained,
            "invoice_results":    self._invoice_results,
            "feature_flag_counts": self._feature_flag_counts,
            "total_blocked_itc":  self._total_blocked_itc,
            "contamination":      self.contamination,
            "n_estimators":       self.n_estimators,
            "random_state":       self.random_state,
        }, path, compress=3)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "GSTAnomalyDetector":
        """Deserialise a previously saved model from disk."""
        payload = joblib.load(path)
        inst = cls(
            contamination=payload.get("contamination", 0.15),
            n_estimators=payload.get("n_estimators", 100),
            random_state=payload.get("random_state", 42),
        )
        inst._preprocessor        = payload["preprocessor"]
        inst._iso_forest          = payload["iso_forest"]
        inst._pop_stats           = payload.get("pop_stats", {})
        inst._trained_at          = payload.get("trained_at")
        inst._n_trained           = payload.get("n_trained", 0)
        inst._invoice_results     = payload.get("invoice_results", {})
        inst._feature_flag_counts = payload.get("feature_flag_counts", {})
        inst._total_blocked_itc   = payload.get("total_blocked_itc", 0.0)
        logger.info("Model loaded ← %s  (%d invoices)", path, inst._n_trained)
        return inst


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton + lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════════

_detector: GSTAnomalyDetector | None = None


def get_detector() -> GSTAnomalyDetector:
    """Return the singleton detector (may be unfitted on first call)."""
    global _detector
    if _detector is None:
        _detector = _try_load_from_disk()
    return _detector


def _try_load_from_disk() -> GSTAnomalyDetector:
    """Attempt to restore a previously saved model; fall back to blank instance."""
    if MODEL_PATH.exists():
        try:
            return GSTAnomalyDetector.load(MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load saved model: %s — starting fresh.", exc)
    return GSTAnomalyDetector()


async def train_detector(db=None) -> GSTAnomalyDetector:
    """
    (Re)train the detector on current MongoDB data and persist to disk.
    Blocking sklearn work runs in a thread-pool executor.
    Returns the fitted detector and updates the module singleton.
    """
    global _detector

    if db is None:
        from backend.database import get_mongo_db
        db = get_mongo_db()

    logger.info("Building feature matrix from MongoDB…")
    features_df, metadata_df = await build_feature_matrix(db)

    det = GSTAnomalyDetector()
    logger.info(
        "Training IsolationForest — %d invoices × %d features, contamination=%.2f",
        len(features_df), len(FEATURE_COLS), det.contamination,
    )
    await asyncio.to_thread(det.fit, features_df, metadata_df)
    await asyncio.to_thread(det.save)

    _detector = det
    return det


async def get_or_train_detector(db=None) -> GSTAnomalyDetector:
    """
    Return the fitted detector, automatically training if not yet fitted.
    Safe to call on every request — training only runs once per process
    (unless POST /ml/retrain is called).
    """
    det = get_detector()
    if not det.is_fitted():
        det = await train_detector(db)
    return det
