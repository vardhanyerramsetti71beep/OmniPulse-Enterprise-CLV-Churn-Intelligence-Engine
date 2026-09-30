"""
Synthetic Enterprise Customer & Transaction Data Generator
Generates realistic multi-dimensional cohorts for B2B SaaS and E-Commerce domains.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_enterprise_saas_data(n_customers: int = 1200, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    random.seed(seed)
    
    reference_date = datetime(2026, 9, 30)
    
    industries = ['Fintech & Banking', 'Healthcare & Life Sciences', 'Cybersecurity', 'Logistics & Supply', 'E-Commerce Tech', 'AI & Cloud Infrastructure', 'MarTech']
    tiers = ['Enterprise', 'Mid-Market', 'Growth / Scaleup']
    tier_weights = [0.20, 0.45, 0.35]
    
    customers = []
    transactions = []
    
    first_names = ["Sarah", "Marcus", "Elena", "Devin", "Aisha", "Liam", "Sophia", "Carlos", "Priya", "Alexander", "Chloe", "Tariq", "Jessica", "Vikram", "Hannah", "David", "Naomi", "James", "Mei", "Arthur"]
    last_names = ["Vance", "Sterling", "Chen", "Dubois", "Al-Mansoor", "Kowalski", "Patel", "Rodriguez", "O'Connor", "Sato", "Mercer", "Blackwood", "Thorne", "Gupta", "Lindqvist", "Kaufman", "Nakamura", "Novak", "Adeyemi", "Sinclair"]
    company_prefixes = ["Apex", "Hyperion", "Nexus", "Quantum", "Vortex", "Starlight", "Synthetix", "Acro", "Pulse", "Stratis", "Omni", "Prism", "Aura", "Catalyst", "Vertex", "Solaria", "Zenith", "Cortex", "Terra", "Nova"]
    company_suffixes = ["Technologies", "Data Corp", "Systems", "Health", "Logistics", "Cloud Labs", "Analytics", "Networks", "Financial", "Interactive", "AI Solutions", "Security Group"]

    for i in range(1, n_customers + 1):
        cust_id = f"CUST-{i:04d}"
        comp_name = f"{random.choice(company_prefixes)} {random.choice(company_suffixes)}"
        contact_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{contact_name.lower().replace(' ', '.')}@{comp_name.lower().replace(' ', '').replace('&', '')[:10]}.io"
        industry = random.choice(industries)
        tier = np.random.choice(tiers, p=tier_weights)
        
        # Tenure in months (1 to 36 months)
        tenure_months = int(np.clip(np.random.exponential(scale=14) + 1, 1, 36))
        signup_date = reference_date - timedelta(days=int(tenure_months * 30.4))
        
        # Contract base sizing by tier
        if tier == 'Enterprise':
            base_monthly = np.random.uniform(4000, 18000)
            seat_count = int(np.random.uniform(150, 2000))
        elif tier == 'Mid-Market':
            base_monthly = np.random.uniform(1200, 4500)
            seat_count = int(np.random.uniform(40, 250))
        else:
            base_monthly = np.random.uniform(350, 1400)
            seat_count = int(np.random.uniform(8, 60))
            
        # Product Engagement & Telemetry
        # Feature adoption rate: 0 - 100%
        feature_adoption = np.random.beta(a=5, b=3) * 100
        # Telemetry velocity (change in DAU/MAU over last 60 days): -60% to +40%
        telemetry_velocity = np.random.normal(loc=-0.02, scale=0.25)
        # Login frequency in days since last active
        days_since_last_login = int(np.clip(np.random.exponential(scale=4 if telemetry_velocity > -0.1 else 22), 0, 90))
        
        # Support metrics
        open_tickets = int(np.random.poisson(lam=1.2 if telemetry_velocity > -0.2 else 3.8))
        p1_critical_tickets = int(np.random.binomial(n=1, p=0.08 if telemetry_velocity > -0.2 else 0.35))
        csat_score = np.clip(np.random.normal(loc=4.4 if p1_critical_tickets == 0 else 2.7, scale=0.6), 1.0, 5.0)
        nps_score = int(np.clip(csat_score * 2.2 - 2 + np.random.normal(0, 1.2), -10, 10))
        
        # Recency, Frequency, Monetary (RFM for transactions)
        # Transactions are renewals or monthly billing runs
        n_transactions = tenure_months
        recency_days = days_since_last_login  # Proxy for RFM
        avg_monthly_value = base_monthly * (1.0 + np.random.normal(0, 0.05))
        total_historic_spend = avg_monthly_value * tenure_months
        
        # Latent Churn Propensity (Ground truth simulation model)
        # Factors: High P1 tickets, declining telemetry, high recency of login, low CSAT
        churn_logits = (
            - 1.8 
            + (days_since_last_login / 18.0)
            - (telemetry_velocity * 3.5)
            + (p1_critical_tickets * 2.2)
            + (open_tickets * 0.35)
            - ((csat_score - 3.0) * 1.2)
            - ((feature_adoption / 100.0) * 1.5)
            + (0.4 if tier == 'Growth / Scaleup' else -0.3)
        )
        churn_prob_latent = 1.0 / (1.0 + np.exp(-churn_logits))
        churn_prob_latent = float(np.clip(churn_prob_latent, 0.02, 0.98))
        
        # Status
        is_churned = bool(np.random.binomial(1, p=churn_prob_latent) if tenure_months > 3 and days_since_last_login > 35 else 0)
        
        # Risk Category
        if churn_prob_latent >= 0.70:
            risk_tier = "Critical"
        elif churn_prob_latent >= 0.45:
            risk_tier = "High"
        elif churn_prob_latent >= 0.22:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"
            
        customers.append({
            "customer_id": cust_id,
            "company_name": comp_name,
            "contact_name": contact_name,
            "email": email,
            "industry": industry,
            "tier": tier,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tenure_months": tenure_months,
            "seat_count": seat_count,
            "monthly_contract_value": round(avg_monthly_value, 2),
            "total_historic_spend": round(total_historic_spend, 2),
            "days_since_last_login": days_since_last_login,
            "telemetry_velocity_pct": round(telemetry_velocity * 100, 1),
            "feature_adoption_score": round(feature_adoption, 1),
            "open_tickets": open_tickets,
            "p1_critical_tickets": p1_critical_tickets,
            "csat_score": round(csat_score, 1),
            "nps_score": nps_score,
            "churn_prob_latent": round(churn_prob_latent, 3),
            "is_churned": is_churned,
            "risk_tier": risk_tier,
            # RFM primitives for lifetimes BG/NBD
            "frequency": max(0, tenure_months - 1),
            "recency_months": max(0, tenure_months - (days_since_last_login / 30.4)),
            "T_months": tenure_months,
            "monetary_value": round(avg_monthly_value, 2)
        })
        
        # Generate monthly transactions
        for m in range(tenure_months):
            tx_date = signup_date + timedelta(days=int(m * 30.4))
            if tx_date > reference_date:
                break
            tx_amount = avg_monthly_value * (1.0 + np.random.normal(0, 0.04))
            transactions.append({
                "transaction_id": f"TX-{cust_id}-{m+1:02d}",
                "customer_id": cust_id,
                "date": tx_date.strftime("%Y-%m-%d"),
                "amount": round(tx_amount, 2),
                "cohort_month": signup_date.strftime("%Y-%m"),
                "period_month": tx_date.strftime("%Y-%m")
            })

    df_cust = pd.DataFrame(customers)
    df_tx = pd.DataFrame(transactions)
    
    return df_cust, df_tx

if __name__ == "__main__":
    df_c, df_t = generate_enterprise_saas_data()
    print(f"Generated {len(df_c)} customers and {len(df_t)} transactions.")
    print("Sample Customer:")
    print(df_c.head(2))
