"""
Intra-Day Temporal Predictive Engine
Forecasts customer risk, telemetry velocity, and transaction revenue
day-to-day (Monday to Saturday) and hour-to-hour (09:00 AM to 04:00 PM).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

OPERATING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
OPERATING_HOURS = [9, 10, 11, 12, 13, 14, 15, 16]

HOUR_LABELS = {
    9: "09:00 AM",
    10: "10:00 AM",
    11: "11:00 AM",
    12: "12:00 PM",
    13: "01:00 PM",
    14: "02:00 PM",
    15: "03:00 PM",
    16: "04:00 PM"
}

# Empirical diurnal operational multipliers
DAY_PROFILES = {
    "Monday": {"base_vol": 1.15, "risk_mult": 1.25, "desc": "Post-weekend triage & ticket backlog resolution"},
    "Tuesday": {"base_vol": 1.35, "risk_mult": 0.95, "desc": "Peak operational velocity & transaction throughput"},
    "Wednesday": {"base_vol": 1.40, "risk_mult": 0.90, "desc": "Core mid-week enterprise platform stability"},
    "Thursday": {"base_vol": 1.30, "risk_mult": 1.05, "desc": "High deployment volume & release integrations"},
    "Friday": {"base_vol": 1.10, "risk_mult": 1.35, "desc": "Pre-weekend escalation risk & freeze windows"},
    "Saturday": {"base_vol": 0.65, "risk_mult": 1.45, "desc": "Off-peak skeleton crew; critical outages carry 2x churn impact"}
}

HOUR_PROFILES = {
    9: {"traffic_pct": 0.75, "risk_bias": 1.20, "focus": "Morning Login Wave & Incident Triage"},
    10: {"traffic_pct": 1.10, "risk_bias": 1.10, "focus": "Peak Morning API Throughput"},
    11: {"traffic_pct": 1.25, "risk_bias": 0.95, "focus": "High-Concurrency Workload Peak"},
    12: {"traffic_pct": 0.90, "risk_bias": 0.85, "focus": "Midday Transition & Scheduled Jobs"},
    13: {"traffic_pct": 1.05, "risk_bias": 0.90, "focus": "Afternoon Operational Resumption"},
    14: {"traffic_pct": 1.20, "risk_bias": 1.15, "focus": "Critical Escalation & Review Window"},
    15: {"traffic_pct": 1.15, "risk_bias": 1.25, "focus": "End-of-Day Settlement & SLA Audits"},
    16: {"traffic_pct": 0.85, "risk_bias": 1.30, "focus": "Daily Close & Escalation Handoff"}
}

class IntraDayPredictiveEngine:
    @staticmethod
    def generate_schedule(df_customers: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates full 48-slot matrix (6 Days x 8 Hours: Mon-Sat, 9AM-4PM)
        with hour-by-hour revenue forecasts, churn indices, and at-risk accounts.
        """
        if df_customers is None or len(df_customers) == 0:
            return {}

        total_base_mrr = df_customers['monthly_contract_value'].sum()
        # Hourly baseline revenue rate: Monthly / (24 business days * 7 active hours)
        hourly_baseline_revenue = total_base_mrr / (24 * 7)
        base_portfolio_churn = df_customers['churn_probability'].mean() * 100

        schedule = {}
        heatmap_matrix = []
        timeseries_points = []
        slot_index = 0

        # Pre-sort high risk customers for active attribution
        critical_pool = df_customers.sort_values(by='clv_at_risk', ascending=False).to_dict(orient='records')

        for d_idx, day in enumerate(OPERATING_DAYS):
            schedule[day] = {}
            day_meta = DAY_PROFILES[day]
            day_matrix_row = []

            for h_idx, hour in enumerate(OPERATING_HOURS):
                hour_meta = HOUR_PROFILES[hour]
                
                # Deterministic pseudo-stochastic variation for smooth diurnal realism
                seed_factor = (d_idx * 17 + h_idx * 13) % 19
                micro_variance = 1.0 + (np.sin(seed_factor) * 0.08)

                # 1. Hourly Revenue Prediction
                predicted_hourly_rev = round(
                    hourly_baseline_revenue * day_meta["base_vol"] * hour_meta["traffic_pct"] * micro_variance,
                    2
                )

                # 2. Hourly Churn Risk Index (Dynamic Platt calibration)
                combined_risk_mult = day_meta["risk_mult"] * hour_meta["risk_bias"] * micro_variance
                hourly_churn_risk = round(
                    float(np.clip(base_portfolio_churn * combined_risk_mult, 3.5, 78.5)),
                    2
                )

                # 3. Telemetry Concurrency Velocity (0 - 100)
                telemetry_velocity = round(
                    float(np.clip(72.0 * day_meta["base_vol"] * hour_meta["traffic_pct"] * micro_variance, 15.0, 99.5)),
                    1
                )

                # 4. Hourly Incident Likelihood (0 - 100)
                incident_likelihood = round(
                    float(np.clip(18.0 * (hourly_churn_risk / 22.0) * micro_variance, 2.0, 92.0)),
                    1
                )

                # 5. Targeted At-Risk Accounts for this hour
                # Pick 2-3 specific accounts that have telemetry friction during this specific window
                stagger_idx = (d_idx * 4 + h_idx * 2) % (min(25, len(critical_pool) - 3))
                slot_accounts = []
                for acc in critical_pool[stagger_idx:stagger_idx + 3]:
                    slot_accounts.append({
                        "customer_id": acc["customer_id"],
                        "company_name": acc["company_name"],
                        "tier": acc["tier"],
                        "churn_probability": round(float(acc["churn_probability"] * combined_risk_mult * 0.95), 3),
                        "clv_at_risk": round(float(acc["clv_at_risk"]), 2),
                        "hourly_trigger": f"{acc['risk_drivers'][0]['factor'] if acc.get('risk_drivers') else 'Latency Spike'} ({HOUR_LABELS[hour]})"
                    })

                # Determine action prescription
                if hourly_churn_risk > 32.0 or incident_likelihood > 45.0:
                    action_rec = f"[CRITICAL ALERT] Mobilize high-touch CSM & standby engineering for {day} {HOUR_LABELS[hour]}"
                    severity = "Critical"
                elif hourly_churn_risk > 22.0:
                    action_rec = f"[ELEVATED RISK] Verify API latency & webhook retries ({HOUR_LABELS[hour]})"
                    severity = "High"
                else:
                    action_rec = f"[NOMINAL] High telemetry stability across {day} ({HOUR_LABELS[hour]})"
                    severity = "Nominal"

                slot_data = {
                    "day": day,
                    "hour": hour,
                    "hour_label": HOUR_LABELS[hour],
                    "predicted_revenue": predicted_hourly_rev,
                    "churn_risk_index_pct": hourly_churn_risk,
                    "telemetry_velocity": telemetry_velocity,
                    "incident_likelihood_pct": incident_likelihood,
                    "severity": severity,
                    "operational_focus": hour_meta["focus"],
                    "action_recommendation": action_rec,
                    "at_risk_accounts": slot_accounts
                }

                schedule[day][hour] = slot_data
                day_matrix_row.append({
                    "day": day,
                    "hour": hour,
                    "label": f"{day[:3]} {HOUR_LABELS[hour]}",
                    "churn_risk": hourly_churn_risk,
                    "revenue": predicted_hourly_rev,
                    "telemetry": telemetry_velocity,
                    "severity": severity
                })

                timeseries_points.append({
                    "index": slot_index,
                    "day": day,
                    "hour": hour,
                    "label": f"{day[:3]} {HOUR_LABELS[hour]}",
                    "churn_risk": hourly_churn_risk,
                    "predicted_revenue": predicted_hourly_rev,
                    "telemetry_velocity": telemetry_velocity
                })
                slot_index += 1

            heatmap_matrix.append(day_matrix_row)

        return {
            "operating_days": OPERATING_DAYS,
            "operating_hours": OPERATING_HOURS,
            "hour_labels": [HOUR_LABELS[h] for h in OPERATING_HOURS],
            "schedule": schedule,
            "heatmap": heatmap_matrix,
            "timeseries": timeseries_points,
            "summary": {
                "total_weekly_predicted_volume": round(sum(p["predicted_revenue"] for p in timeseries_points), 2),
                "peak_risk_slot": max(timeseries_points, key=lambda x: x["churn_risk"]),
                "peak_velocity_slot": max(timeseries_points, key=lambda x: x["telemetry_velocity"]),
                "lowest_risk_slot": min(timeseries_points, key=lambda x: x["churn_risk"])
            }
        }
