"""
Retention Optimizer & Uplift Simulation Engine
Calculates Next-Best-Action (NBA) playbooks and dynamic What-If ROI uplift scenarios.
"""

import numpy as np
import pandas as pd

PLAYBOOK_CATALOG = {
    "EXECUTIVE_SPONSOR": {
        "title": "Executive Sponsor & High-Touch CSM Intervention",
        "description": "Deploy VP of Customer Success + Principal Solutions Architect for emergency 1-on-1 alignment, roadmap preview, and SLA escalation resolution.",
        "churn_reduction_pct": 46.0,
        "base_cost": 450.0,
        "channel": "White-Glove / In-Person",
        "suitable_tiers": ["Enterprise", "Mid-Market"]
    },
    "PRICING_RESTRUCTURE": {
        "title": "Commercial Restructure & Multi-Year Loyalty Incentive",
        "description": "Offer custom 12-month extension with 12% renewal incentive, bundled complimentary add-on seats, and flexible quarterly milestone billing.",
        "churn_reduction_pct": 52.0,
        "base_cost": 300.0,
        "cost_pct_arr": 0.08,
        "channel": "Commercial Sales Rep",
        "suitable_tiers": ["Enterprise", "Mid-Market", "Growth / Scaleup"]
    },
    "ADOPTION_WORKSHOP": {
        "title": "Accelerated Feature Adoption & Solution Engineering Sprint",
        "description": "Pair customer engineering team with dedicated solution architect to integrate dormant API endpoints and deploy pre-built analytics templates.",
        "churn_reduction_pct": 34.0,
        "base_cost": 95.0,
        "channel": "Solutions Architect / Zoom Sprint",
        "suitable_tiers": ["Enterprise", "Mid-Market", "Growth / Scaleup"]
    },
    "CONCIERGE_CHECKIN": {
        "title": "Proactive Health Check-in & Customer Care Priority Pass",
        "description": "Priority ticket routing, dedicated Slack connect channel, and proactive quarterly business review (QBR) booking.",
        "churn_reduction_pct": 28.0,
        "base_cost": 60.0,
        "channel": "CSM Slack / Email",
        "suitable_tiers": ["Mid-Market", "Growth / Scaleup"]
    },
    "AUTOMATED_NUDGE": {
        "title": "Automated Re-engagement Workflow & Product Tours",
        "description": "Trigger personalized in-app guided walkthroughs, milestone email digests, and interactive onboarding tooltips.",
        "churn_reduction_pct": 16.0,
        "base_cost": 8.0,
        "channel": "In-App Product Nudge",
        "suitable_tiers": ["Growth / Scaleup"]
    }
}

class RetentionOptimizer:
    @staticmethod
    def assign_playbooks(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        playbooks = []
        for _, row in df.iterrows():
            tier = row.get('tier', 'Mid-Market')
            prob = row.get('churn_probability', 0.1)
            clv = row.get('predicted_clv_12m', row.get('monthly_contract_value', 1000) * 12)
            p1 = row.get('p1_critical_tickets', 0)
            velocity = row.get('telemetry_velocity_pct', 0)
            adoption = row.get('feature_adoption_score', 50)
            
            if tier == 'Enterprise' and (p1 > 0 or prob >= 0.55):
                pb_key = "EXECUTIVE_SPONSOR"
            elif clv >= 35000 and prob >= 0.40:
                pb_key = "PRICING_RESTRUCTURE"
            elif adoption < 45 or velocity < -20:
                pb_key = "ADOPTION_WORKSHOP"
            elif prob >= 0.25:
                pb_key = "CONCIERGE_CHECKIN"
            else:
                pb_key = "AUTOMATED_NUDGE"
                
            pb_meta = PLAYBOOK_CATALOG[pb_key]
            
            # Estimated uplift calculation for this customer
            reduction = pb_meta['churn_reduction_pct'] / 100.0
            new_prob = round(prob * (1.0 - reduction), 4)
            saved_prob = round(prob - new_prob, 4)
            saved_equity = round(saved_prob * clv, 2)
            
            cost = pb_meta['base_cost']
            if 'cost_pct_arr' in pb_meta:
                cost += row.get('monthly_contract_value', 1000) * 12 * pb_meta['cost_pct_arr']
            cost = round(cost, 2)
            
            net_uplift = round(saved_equity - cost, 2)
            roi_ratio = round((saved_equity / max(cost, 1.0)), 2)
            
            playbooks.append({
                "playbook_id": pb_key,
                "title": pb_meta['title'],
                "description": pb_meta['description'],
                "channel": pb_meta['channel'],
                "original_churn_prob": prob,
                "projected_churn_prob": new_prob,
                "churn_reduction_pct": pb_meta['churn_reduction_pct'],
                "intervention_cost": cost,
                "saved_equity_clv": saved_equity,
                "net_value_uplift": net_uplift,
                "expected_roi_ratio": roi_ratio
            })
            
        df['recommended_playbook'] = playbooks
        return df

    @staticmethod
    def simulate_what_if_scenario(
        df: pd.DataFrame,
        target_risk_tiers: list = None,
        retention_budget: float = 50000.0,
        discount_incentive_pct: float = 10.0,
        execution_intensity: float = 1.0 # 0.5 (Conservative) to 1.5 (Aggressive)
    ) -> dict:
        """
        Interactive What-If Scenario simulator for enterprise decision makers.
        """
        if target_risk_tiers is None or len(target_risk_tiers) == 0:
            target_risk_tiers = ['Critical', 'High']
            
        # Filter eligible cohort
        eligible = df[df['risk_tier'].isin(target_risk_tiers)].copy()
        
        if len(eligible) == 0:
            return {
                "targeted_accounts": 0,
                "total_population": len(df),
                "total_current_risk_clv": 0.0,
                "projected_saved_clv": 0.0,
                "total_campaign_cost": 0.0,
                "net_profit_uplift": 0.0,
                "roi_multiple": 0.0,
                "churn_rate_before_pct": 0.0,
                "churn_rate_after_pct": 0.0
            }
            
        # Sort by CLV at risk descending to allocate budget optimally
        eligible = eligible.sort_values(by='clv_at_risk', ascending=False)
        
        total_spent = 0.0
        saved_clv_sum = 0.0
        targeted_count = 0
        
        # Calculate baseline stats
        baseline_churn_rate = (df['churn_probability'] > 0.5).mean() * 100
        total_risk_clv = eligible['clv_at_risk'].sum()
        
        saved_probabilities = []
        
        for _, row in eligible.iterrows():
            arr = row['monthly_contract_value'] * 12
            clv = row.get('predicted_clv_12m', arr)
            prob = row['churn_probability']
            
            # Cost per account
            account_cost = 150.0 + (arr * (discount_incentive_pct / 100.0) * 0.4)
            if total_spent + account_cost > retention_budget and targeted_count > 5:
                break
                
            total_spent += account_cost
            targeted_count += 1
            
            # Uplift effectiveness modulated by execution intensity and discount incentive
            base_reduction = 0.35 * execution_intensity * (1.0 + (discount_incentive_pct / 100.0) * 0.8)
            reduction_factor = min(0.75, base_reduction)
            
            delta_prob = prob * reduction_factor
            saved_equity = delta_prob * clv
            
            saved_clv_sum += saved_equity
            saved_probabilities.append(prob - delta_prob)
            
        net_profit = saved_clv_sum - total_spent
        roi_mult = saved_clv_sum / max(total_spent, 1.0)
        
        # Projected overall population churn rate after campaign
        new_churn_rate = max(1.2, baseline_churn_rate - (targeted_count / len(df) * 12.0 * execution_intensity))
        
        return {
            "targeted_accounts": targeted_count,
            "total_population": len(df),
            "total_current_risk_clv": round(float(total_risk_clv), 2),
            "projected_saved_clv": round(float(saved_clv_sum), 2),
            "total_campaign_cost": round(float(total_spent), 2),
            "net_profit_uplift": round(float(net_profit), 2),
            "roi_multiple": round(float(roi_mult), 2),
            "churn_rate_before_pct": round(float(baseline_churn_rate), 2),
            "churn_rate_after_pct": round(float(new_churn_rate), 2)
        }

if __name__ == "__main__":
    from backend.data_generator import generate_enterprise_saas_data
    from backend.models.clv_engine import CLVEngine
    from backend.models.churn_engine import ChurnIntelligenceEngine
    
    df_c, _ = generate_enterprise_saas_data(300)
    clv_eng = CLVEngine()
    df_scored = clv_eng.fit_and_predict(df_c)
    churn_eng = ChurnIntelligenceEngine().train(df_scored)
    df_final = churn_eng.predict(df_scored)
    
    sim = RetentionOptimizer.simulate_what_if_scenario(
        df_final, 
        target_risk_tiers=['Critical', 'High'],
        retention_budget=35000,
        discount_incentive_pct=12.0
    )
    print("Simulation Results:", sim)
