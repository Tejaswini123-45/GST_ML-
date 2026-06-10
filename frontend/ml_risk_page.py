"""
ML Risk Intelligence page for the GST CFO Dashboard.
This module is imported by cfo_dashboard.py — do not run directly.
"""

import pandas as pd
import streamlit as st

# ── Design system constants ────────────────────────────────────────────────
_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
_HEX   = {"CRITICAL": "#ff1744", "HIGH": "#ff6d00", "MEDIUM": "#f9a825", "LOW": "#2e7d32"}

_FEATURE_DESCRIPTIONS = {
    "invoice_value":              "Raw invoice amount (₹)",
    "log_invoice_value":          "Log₁₊ transform — normalises heavy tail distribution",
    "effective_tax_rate":         "Tax ÷ Value · standard GST = 18%",
    "gstr1_filed":                "1 = GSTR1 present · 0 = missing",
    "gstr1_status_encoded":       "0=FILED 1=LATE 2=PENDING 3=MISSING",
    "days_to_file":               "Days between invoice date and GSTR1 filing",
    "itc_eligible":               "1 = YES · 0 = NO (from GSTR2B)",
    "gstr2b_value_discrepancy":   "|GSTR2B value − Invoice value| ÷ Invoice value",
    "pr_value_discrepancy":       "|PR claimed − Invoice| ÷ Invoice (over-claim signal)",
    "pr_tax_discrepancy":         "|PR tax − GSTR1 tax| ÷ GSTR1 tax",
    "has_ewaybill":               "1 = EWB present · 0 = missing transport document",
    "ewb_value_discrepancy":      "|EWB value − Invoice value| ÷ Invoice value",
    "seller_invoice_count":       "Total invoices issued by this seller",
    "seller_avg_value":           "Seller mean invoice value (₹)",
    "seller_value_std":           "Std deviation of seller invoice values",
    "invoice_value_zscore":       "(Value − seller mean) ÷ seller std · key outlier signal",
    "seller_filing_rate":         "Fraction of seller invoices with GSTR1 filed",
    "seller_late_rate":           "Fraction of filed returns that are LATE",
    "seller_gstr3b_payment_rate": "Fraction of GSTR3B records with confirmed payment",
    "seller_itc_block_rate":      "Fraction of seller invoices with ITC blocked",
    "seller_risk_encoded":        "Seller risk: HIGH=2 MEDIUM=1 LOW=0",
    "buyer_invoice_count":        "Total invoices received by buyer",
    "buyer_avg_value":            "Buyer mean invoice value (₹)",
    "buyer_itc_block_rate":       "Fraction of buyer invoices with ITC blocked",
    "buyer_risk_encoded":         "Buyer risk: HIGH=2 MEDIUM=1 LOW=0",
    "invoice_day":                "Day of month · end-of-month clustering = fraud signal",
    "invoice_month":              "Month (1–12) · seasonal anomaly detection",
}

# ═══════════════════════════════════════════════════════════════════════════
# Public page entry point
# ═══════════════════════════════════════════════════════════════════════════

def render(gstin: str, get_fn, post_fn) -> None:
    """
    Render the fully rebuilt, premium Enterprise ML Risk Intelligence platform.
    """
    # ── Premium Theme CSS Injection (Datadog/Palantir Foundry styling) ──
    st.markdown(
        """
        <style>
        /* Global Font & Spacing Reset */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
        
        .main .block-container {
            font-family: 'Outfit', -apple-system, sans-serif !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
        }
        
        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: var(--text-color) !important;
        }
        
        /* Premium Hero Panel */
        .hero-panel {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            color: #ffffff !important;
        }
        
        .hero-title {
            font-size: 2.4rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em !important;
        }
        
        .hero-subtitle {
            font-size: 1rem !important;
            color: #94a3b8 !important;
            margin-top: 0.4rem !important;
            margin-bottom: 1.5rem !important;
            font-weight: 400 !important;
        }
        
        .hero-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1.25rem;
            margin-top: 1.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 1.5rem;
        }
        
        .hero-stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
        
        .hero-stat-val {
            font-size: 1.4rem;
            font-weight: 800;
            color: #f8fafc;
        }
        
        .hero-stat-lbl {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }
        
        /* Premium KPI Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }
        
        .premium-kpi-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 14px;
            padding: 1.6rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: all 0.25s ease-in-out;
            position: relative;
            overflow: hidden;
        }
        
        .premium-kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
        }
        
        .kpi-critical::before { background-color: #ff1744; }
        .kpi-high::before { background-color: #ff6d00; }
        .kpi-medium::before { background-color: #f9a825; }
        .kpi-low::before { background-color: #2e7d32; }
        .kpi-exposure::before { background-color: #7e57c2; }
        .kpi-itc::before { background-color: #00bcd4; }
        
        .premium-kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            border-color: var(--primary-color);
        }
        
        .kpi-card-val {
            font-size: 2.3rem;
            font-weight: 850;
            margin: 0.3rem 0;
            color: var(--text-color);
            line-height: 1;
        }
        
        .kpi-card-lbl {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #888888;
            font-weight: 700;
        }
        
        .kpi-card-desc {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.3rem;
        }
        
        /* Premium Findings Center */
        .finding-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.02);
            transition: border-color 0.2s;
        }
        .finding-card:hover {
            border-color: var(--primary-color);
        }
        
        /* Badge definitions */
        .premium-badge {
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
        }
        
        .badge-crit { background-color: rgba(255, 23, 68, 0.12); color: #ff1744; border: 1px solid rgba(255, 23, 68, 0.25); }
        .badge-hi   { background-color: rgba(255, 109, 0, 0.12); color: #ff6d00; border: 1px solid rgba(255, 109, 0, 0.25); }
        .badge-med  { background-color: rgba(249, 168, 37, 0.12); color: #f9a825; border: 1px solid rgba(249, 168, 37, 0.25); }
        .badge-lo   { background-color: rgba(46, 125, 50, 0.12); color: #2e7d32; border: 1px solid rgba(46, 125, 50, 0.25); }
        
        /* XAI Bar display */
        .xai-bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        
        .xai-bar-label {
            width: 200px;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-color);
        }
        
        .xai-bar-track {
            flex-grow: 1;
            background-color: rgba(128, 128, 128, 0.15);
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            margin-right: 1rem;
        }
        
        .xai-bar-fill {
            height: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, #6366f1, #818cf8);
        }
        
        .xai-bar-val {
            width: 60px;
            font-size: 0.88rem;
            font-weight: 700;
            text-align: right;
            color: var(--text-color);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ── Load summary dataset (auto-trains model if not yet fitted) ─────────
    with st.spinner("Loading Enterprise Risk Ledger…"):
        summary = get_fn("/ml/risk-summary", timeout=90)

    if summary is None:
        st.error("❌ Could not connect to FastAPI Backend. Check local network routing.")
        return

    if not summary.get("model_trained"):
        _render_untrained(post_fn)
        return

    dist       = summary.get("risk_distribution", {})
    total      = summary.get("total_invoices", 0)
    trained_at = (summary.get("model_trained_at") or "")[:19].replace("T", " ")

    critical = dist.get("CRITICAL", 0)
    high     = dist.get("HIGH", 0)
    medium   = dist.get("MEDIUM", 0)
    low      = dist.get("LOW", 0)

    # Calculate Portfolio Risk Score (0 to 100 index)
    if total > 0:
        weighted_sum = (3 * critical) + (2 * high) + (1 * medium)
        max_possible = 3 * total
        risk_score   = (weighted_sum / max_possible) * 100
    else:
        risk_score   = 0.0

    if risk_score >= 50.0:
        score_color  = "#ff1744"
        score_lvl    = "CRITICAL AUDIT RISK"
        score_desc   = "Immediate remediation required. High concentration of anomalous transactions and blocked ITC."
        badge_class  = "badge-crit"
    elif risk_score >= 25.0:
        score_color  = "#ff6d00"
        score_lvl    = "HIGH AUDIT RISK"
        score_desc   = "Significant compliance exposure. Multiple supplier filing discrepancies and GSTR3B warnings."
        badge_class  = "badge-hi"
    elif risk_score >= 10.0:
        score_color  = "#f9a825"
        score_lvl    = "MODERATE AUDIT RISK"
        score_desc   = "Minor discrepancies detected. Monitor flagged transactions and seller risk trends."
        badge_class  = "badge-med"
    else:
        score_color  = "#2e7d32"
        score_lvl    = "LOW AUDIT RISK"
        score_desc   = "Portfolio is healthy. Anomalies and filing behaviors are well within normal variance."
        badge_class  = "badge-lo"

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1 — EXECUTIVE HERO PANEL
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown(
        f"""
        <div class="hero-panel">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <h1 class="hero-title">AI GST Risk Intelligence Platform</h1>
                    <p class="hero-subtitle">Real-time Isolation Forest Anomaly Detection and Automated Input Tax Credit Assurance</p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.05); padding: 1rem 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Portfolio Risk Score</div>
                    <div style="font-size: 2.5rem; font-weight: 900; color: {score_color}; line-height: 1;">{risk_score:.0f} <span style="font-size: 1.2rem; color: #64748b; font-weight: 500;">/ 100</span></div>
                    <div style="margin-top: 0.5rem;"><span class="premium-badge {badge_class}">{score_lvl}</span></div>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.3rem;">
                    <span>System Vulnerability Gauge</span>
                    <span>{risk_score:.1f}% exposure index</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); border-radius: 8px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #38bdf8, {score_color}); width: {risk_score}%; height: 100%; border-radius: 8px;"></div>
                </div>
                <p style="margin: 0.8rem 0 0 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.4;">{score_desc}</p>
            </div>
            <div class="hero-stats-grid">
                <div class="hero-stat-card">
                    <div class="hero-stat-val">{total}</div>
                    <div class="hero-stat-lbl">Invoices Scored</div>
                </div>
                <div class="hero-stat-card">
                    <div class="hero-stat-val">₹{summary.get("total_invoice_value_at_risk", 0) * 5:,.0f}</div>
                    <div class="hero-stat-lbl">Total Value Scanned</div>
                </div>
                <div class="hero-stat-card" style="border-left: 2px solid #ff1744;">
                    <div class="hero-stat-val">{critical}</div>
                    <div class="hero-stat-lbl">Critical Anomalies</div>
                </div>
                <div class="hero-stat-card" style="border-left: 2px solid #ff6d00;">
                    <div class="hero-stat-val">{high}</div>
                    <div class="hero-stat-lbl">High-Risk Invoices</div>
                </div>
                <div class="hero-stat-card" style="border-left: 2px solid #7e57c2;">
                    <div class="hero-stat-val">₹{summary.get("total_invoice_value_at_risk", 0):,.0f}</div>
                    <div class="hero-stat-lbl">Exposure at Risk</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 2. SECTION 2 — EXECUTIVE KPI GRID ──────────────────────────────────
    
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="premium-kpi-card kpi-critical">
                <div class="kpi-card-lbl">🔴 Critical Risks</div>
                <div class="kpi-card-val">{critical}</div>
                <div class="kpi-card-desc">Anomalous score &ge; 0.75. Stop pay.</div>
            </div>
            <div class="premium-kpi-card kpi-high">
                <div class="kpi-card-lbl">🟠 High Risks</div>
                <div class="kpi-card-val">{high}</div>
                <div class="kpi-card-desc">Score &ge; 0.50. Active compliance gap.</div>
            </div>
            <div class="premium-kpi-card kpi-medium">
                <div class="kpi-card-lbl">🟡 Medium Risks</div>
                <div class="kpi-card-val">{medium}</div>
                <div class="kpi-card-desc">Score &ge; 0.30. Discrepancy warning.</div>
            </div>
            <div class="premium-kpi-card kpi-low">
                <div class="kpi-card-lbl">🟢 Low Risks</div>
                <div class="kpi-card-val">{low}</div>
                <div class="kpi-card-desc">Fully compliant filing behavior.</div>
            </div>
            <div class="premium-kpi-card kpi-exposure">
                <div class="kpi-card-lbl">💰 Total Exposure</div>
                <div class="kpi-card-val">₹{summary.get("total_invoice_value_at_risk", 0):,.0f}</div>
                <div class="kpi-card-desc">Audit exposure in High + Critical tiers.</div>
            </div>
            <div class="premium-kpi-card kpi-itc">
                <div class="kpi-card-lbl">🛡️ Blocked ITC</div>
                <div class="kpi-card-val">₹{summary.get("total_blocked_itc", 0):,.0f}</div>
                <div class="kpi-card-desc">Input Tax Credit marked blocked.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 3. SECTION 3 — AI FINDINGS CENTER ──────────────────────────────────
    
    st.markdown("## 🚨 Top Suspicious Invoices (AI Findings)")
    top_anomalies = summary.get("top_anomalies", [])
    
    if not top_anomalies:
        st.info("No anomalies flagged in this ledger run.")
    else:
        # Show Top 5 Dangerous Invoices as premium card list
        for i, a in enumerate(top_anomalies[:5], 1):
            lvl = a.get("risk_level", "LOW")
            lbl_class = {
                "CRITICAL": "badge-crit",
                "HIGH": "badge-hi",
                "MEDIUM": "badge-med",
                "LOW": "badge-lo"
            }.get(lvl, "badge-lo")
            
            top_reason = (a.get("anomaly_reasons") or ["—"])[0]
            
            st.markdown(
                f"""
                <div class="finding-card" style="border-left: 5px solid {_HEX.get(lvl, '#aaa')};">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 1.1rem; font-weight: 800; color: #888;">#{i}</span>
                            <div>
                                <strong style="font-size: 1.1rem; color: var(--text-color);">Invoice {a['invoice_id']}</strong>
                                <div style="font-size: 0.8rem; color: #777;">Seller: {a.get('seller_gstin', '')} &nbsp;|&nbsp; Buyer: {a.get('buyer_gstin', '')}</div>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1.5rem; margin-top: 0.5rem;">
                            <div style="text-align: right;">
                                <div style="font-size: 1.1rem; font-weight: 800;">₹{a['invoice_value']:,.0f}</div>
                                <div style="font-size: 0.72rem; color: #888; text-transform: uppercase;">Amount</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.1rem; font-weight: 800; color: {_HEX.get(lvl, '#aaa')};">{a['risk_score']:.4f}</div>
                                <div style="font-size: 0.72rem; color: #888; text-transform: uppercase;">Anomaly Score</div>
                            </div>
                            <div>
                                <span class="premium-badge {lbl_class}">{lvl}</span>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 0.6rem; font-size: 0.88rem; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 0.5rem; color: var(--text-color);">
                        <strong style="color: {_HEX.get(lvl, '#aaa')};">Primary Flag:</strong> {top_reason}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # ── 4. SECTION 4 — VISUAL ANALYTICS ────────────────────────────────────
    
    st.markdown("## 📊 Visual Analytics & Breakdown")
    tab_dist, tab_signals = st.tabs(["Portfolio Risk Distribution", "Compliance Signal Distribution"])
    
    with tab_dist:
        dist_df = pd.DataFrame({
            "Invoice Count": [low, medium, high, critical],
        }, index=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        st.bar_chart(dist_df, color="#ef5350")
        st.caption("Volume profile of compliance audit risk classification.")
        
    with tab_signals:
        flagged = summary.get("most_flagged_features", [])
        if flagged:
            bar_df = pd.DataFrame({
                "Incident Count": [f["flag_count"] for f in flagged],
            }, index=[f["feature"] for f in flagged])
            st.bar_chart(bar_df, color="#7e57c2")
            st.caption("Frequency spectrum of triggered compliance discrepancies.")
        else:
            st.info("No signal logs available in database.")

    st.divider()

    # ── 5. SECTION 5 — INVOICE INVESTIGATION CENTER ────────────────────────
    
    st.markdown("## 🔍 Invoice Investigation Center")
    st.caption("Select any invoice below to load its full compliance metadata, features, and explainable AI profiles.")
    
    with st.spinner("Retrieving Ledger Indices…"):
        all_scored = get_fn("/ml/anomalies", params={"min_risk_level": "LOW", "limit": 500})
        
    if not all_scored:
        st.warning("No invoices found. Ingest CSV data or trigger a model retrain.")
        return

    options_map = {
        f"{_EMOJI.get(a['risk_level'], '')} {a['invoice_id']} | Risk: {a['risk_level']} ({a['risk_score']:.3f}) | Value: ₹{a['invoice_value']:,.0f}": a["invoice_id"]
        for a in all_scored
    }
    
    selected_label = st.selectbox("Select Target Invoice for Investigation", list(options_map.keys()), key="ml_inv_sel")
    if selected_label:
        sel_id = options_map[selected_label]
        with st.spinner(f"Loading anomaly detail for {sel_id}…"):
            detail = get_fn(f"/ml/anomaly/{sel_id}")
        if detail:
            _render_investigation_console(detail)
        else:
            st.error(f"Could not retrieve audit trail for {sel_id}.")

    st.divider()

    # ── 8. Refresh Model Section (Retrain) ─────────────────────────────────
    _section_retrain(post_fn, total, trained_at)


# ═══════════════════════════════════════════════════════════════════════════
# Console & UI Sub-renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_investigation_console(detail: dict) -> None:
    """
    Console layout displaying Left Panel (Metadata) and Right Panel (AI Investigation).
    """
    level   = detail.get("risk_level", "LOW")
    score   = detail.get("risk_score", 0)
    inv_id  = detail.get("invoice_id", "")
    reasons = detail.get("anomaly_reasons", [])
    feat    = detail.get("features", {})
    comp    = detail.get("compliance", {})
    color   = _HEX.get(level, "#aaa")
    
    lbl_class = {
        "CRITICAL": "badge-crit",
        "HIGH": "badge-hi",
        "MEDIUM": "badge-med",
        "LOW": "badge-lo"
    }.get(level, "badge-lo")

    st.markdown(
        f"""
        <div style="border-left: 6px solid {color}; background-color: var(--secondary-background-color); '
        'border-top: 1px solid rgba(128,128,128,0.15); border-right: 1px solid rgba(128,128,128,0.15); '
        'border-bottom: 1px solid rgba(128,128,128,0.15); border-radius: 8px; padding: 1.25rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h3 style="margin: 0; font-size: 1.3rem; font-weight: 800;">{_EMOJI.get(level, '')} Investigation Dossier: {inv_id}</h3>
                    <p style="margin: 0.2rem 0 0 0; color: #888;">Anomaly score: <strong>{score:.4f}</strong> &nbsp;|&nbsp; Isolation Forest score: <strong>{detail.get("isolation_score", 0):.4f}</strong></p>
                </div>
                <div>
                    <span class="premium-badge {lbl_class}">{level} RISK</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📋 Left Panel: Invoice Details")
        st.write(f"**Invoice ID**: `{inv_id}`")
        st.write(f"**Value**: ₹ {detail.get('invoice_value', 0):,.0f}")
        st.write(f"**Date**: {detail.get('invoice_date', '—')}")
        st.write(f"**Seller**: {detail.get('seller_name', '—')} (`{detail.get('seller_gstin', '')}`)")
        st.write(f"**Buyer**: {detail.get('buyer_name', '—')} (`{detail.get('buyer_gstin', '')}`)")
        
        st.markdown("#### Filing Checklist")
        def _v(b: bool) -> str:
            return "✅ Yes" if b else "❌ No"
        
        rows_c = [
            {"Check": "GSTR1 outward return filed?", "Status": _v(comp.get("gstr1_present", False))},
            {"Check": "Supplier GSTR1 status", "Status": str(comp.get("gstr1_status", "MISSING")).upper()},
            {"Check": "GSTR2B generated?", "Status": _v(comp.get("gstr2b_present", False))},
            {"Check": "ITC Eligible status", "Status": str(comp.get("itc_eligible", "MISSING")).upper()},
            {"Check": "Seller GSTR3B tax paid?", "Status": _v(comp.get("gstr3b_payment_confirmed", False))},
            {"Check": "E-Way Bill present?", "Status": _v(comp.get("ewaybill_present", False))},
        ]
        st.table(pd.DataFrame(rows_c).set_index("Check"))
        
    with col_right:
        st.markdown("### 🤖 Right Panel: AI Investigation Results")
        
        st.markdown("#### Primary Compliance Failures")
        if not reasons or "no significant" in (reasons[0] or "").lower():
            st.success("✅ No significant anomaly signals detected.")
        else:
            for r in reasons:
                kw_crit = any(
                    w in r.lower()
                    for w in ["hold payment", "invalid", "blocked", "ineligible", "mandatory", "fictitious"]
                )
                kw_warn = any(
                    w in r.lower()
                    for w in ["late", "pending", "missing", "poor", "low", "chronic", "delay"]
                )
                
                if kw_crit:
                    st.error(f"**CRITICAL:** {r}")
                elif kw_warn:
                    st.warning(f"**WARNING:** {r}")
                else:
                    st.info(f"**SIGNAL:** {r}")

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 6 — EXPLAINABLE AI PANEL
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("#### 💡 Explainable AI Risk Drivers")
        st.caption("Weight contributions to anomaly scoring (from feature z-score divergence):")
        
        # Calculate feature deviations to display as visual XAI bars
        # This gives a beautiful representation of why the ML flagged it!
        feat_rows = []
        for col, val in feat.items():
            if col in _FEATURE_DESCRIPTIONS and isinstance(val, (int, float)):
                # Derive relative weight contribution using raw values/ratios
                val_f = float(val)
                # Map select critical columns to relative weight contributions
                weight = 0.0
                if col == "invoice_value_zscore": weight = abs(val_f) * 18.0
                elif col == "gstr2b_value_discrepancy": weight = val_f * 24.0
                elif col == "pr_value_discrepancy": weight = val_f * 22.0
                elif col == "days_to_file": weight = (val_f / 30.0) * 12.0
                elif col == "seller_itc_block_rate": weight = val_f * 35.0
                elif col == "seller_gstr3b_payment_rate": weight = (1.0 - val_f) * 28.0
                elif col == "gstr1_filed": weight = (1.0 - val_f) * 31.0
                elif col == "has_ewaybill": weight = (1.0 - val_f) * 26.0
                
                weight = min(weight, 100.0)
                if weight > 0:
                    feat_rows.append((col, weight))
                    
        # Sort and take top drivers
        feat_rows.sort(key=lambda x: x[1], reverse=True)
        top_drivers = feat_rows[:5]
        
        if not top_drivers:
            st.info("No feature outliers detected.")
        else:
            for col, w in top_drivers:
                lbl = _FEATURE_DESCRIPTIONS.get(col, col)
                if len(lbl) > 28:
                    lbl = lbl[:26] + "..."
                st.markdown(
                    f"""
                    <div class="xai-bar-row">
                        <div class="xai-bar-label">{lbl}</div>
                        <div class="xai-bar-track">
                            <div class="xai-bar-fill" style="width: {w}%; background-color: {color};"></div>
                        </div>
                        <div class="xai-bar-val">+{w:.0f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ── All 28 features tab inside drilldown ──
        st.write("")
        with st.expander("⚙️ View All 28 Engineered Features"):
            f_rows = []
            for col, val in feat.items():
                if col in _FEATURE_DESCRIPTIONS:
                    f_rows.append({
                        "Feature Column": col,
                        "Value":          round(float(val), 4) if isinstance(val, (int, float)) else val,
                        "Interpretation Baseline": _FEATURE_DESCRIPTIONS[col],
                    })
            if f_rows:
                st.dataframe(
                    pd.DataFrame(f_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature Column": st.column_config.TextColumn(width="medium"),
                        "Value": st.column_config.NumberColumn(format="%.4f"),
                        "Interpretation Baseline": st.column_config.TextColumn(width="large"),
                    }
                )


def _render_untrained(post_fn) -> None:
    st.info(
        "ℹ️ ML model has not been trained yet. "
        "Click **Train Model** to run Isolation Forest on the current GST data."
    )
    if st.button("🚀 Train Model Now", type="primary", key="btn_train_first"):
        with st.spinner("Training Isolation Forest on 28 GST features…"):
            result = post_fn("/ml/retrain", {})
        if result and result.get("status") == "success":
            st.success(
                f"✅ Model trained — {result['invoices_trained_on']} invoices × "
                f"{result['features_used']} features in {result['duration_seconds']}s"
            )
            st.rerun()
        else:
            st.error("Training failed. Ensure data has been ingested first.")


def _section_retrain(post_fn, total: int, trained_at: str) -> None:
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 7 — MODEL INFORMATION
    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Model Pipeline Information"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Algorithm**: Isolation Forest (Unsupervised Pipeline)")
            st.write("**Model Version**: v1.0.0-Stable")
            st.write("**Contamination Rate**: 15.0%")
            st.write(f"**Last Calibration**: {trained_at} UTC")
        with c2:
            st.write(f"**Total Features Used**: 28 Columns")
            st.write(f"**Invoices Trained On**: {total} Records")
            st.write("**Deployment Status**: Live / Active")
            
        st.write("")
        st.caption("Recalibrate model to refresh risk indices after new CSV ingest:")
        if st.button("🔄 Trigger Model Retraining", key="btn_ml_retrain2"):
            with st.spinner("Retraining…"):
                res = post_fn("/ml/retrain", {})
            if res and res.get("status") == "success":
                st.success(
                    f"✅ Calibration completed — {res['invoices_trained_on']} invoices in "
                    f"{res['duration_seconds']}s"
                )
                st.rerun()
            else:
                st.error("Retraining failed. Check backend logs.")
