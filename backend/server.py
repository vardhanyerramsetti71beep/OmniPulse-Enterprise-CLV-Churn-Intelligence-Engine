"""
FastAPI Server for OmniPulse CLV & Churn Intelligence Engine
Provides real-time predictive analytics, Customer 360 exploration,
What-If scenario simulation, and Next-Best-Action retention automation.
"""

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import numpy as np

from backend.data_generator import generate_enterprise_saas_data
from backend.models.clv_engine import CLVEngine
from backend.models.churn_engine import ChurnIntelligenceEngine
from backend.models.retention_simulator import RetentionOptimizer, PLAYBOOK_CATALOG
from backend.models.cohort_engine import CohortEngine
from backend.models.risk_advisor import RiskAdvisorBot
from backend.models.intraday_engine import IntraDayPredictiveEngine

app = FastAPI(
    title="OmniPulse CLV & Churn Intelligence Engine",
    description="Enterprise API for Customer Lifetime Value Optimization & Predictive Churn Mitigation",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory application state
STATE = {
    "df_customers": None,
    "df_transactions": None,
    "clv_engine": None,
    "churn_engine": None,
    "cohort_data": None,
    "intraday_schedule": None,
    "dataset_type": "B2B SaaS (Enterprise)"
}

def initialize_engine(n_customers=1200, seed=42):
    print("[OmniPulse] Initializing OmniPulse Intelligence Engine...")
    df_c, df_t = generate_enterprise_saas_data(n_customers=n_customers, seed=seed)
    
    clv_eng = CLVEngine(discount_rate=0.08, gross_margin=0.78)
    df_scored = clv_eng.fit_and_predict(df_c)
    
    churn_eng = ChurnIntelligenceEngine()
    churn_eng.train(df_scored)
    df_final = churn_eng.predict(df_scored)
    
    df_final = RetentionOptimizer.assign_playbooks(df_final)
    cohort_matrix = CohortEngine.compute_cohort_matrix(df_t, df_final)
    intraday_sched = IntraDayPredictiveEngine.generate_schedule(df_final)
    
    STATE["df_customers"] = df_final
    STATE["df_transactions"] = df_t
    STATE["clv_engine"] = clv_eng
    STATE["churn_engine"] = churn_eng
    STATE["cohort_data"] = cohort_matrix
    STATE["intraday_schedule"] = intraday_sched
    print("[OmniPulse] Engine initialization complete. Ready to serve predictions.")

# Initialize on module load
initialize_engine()

# --- Request / Response Models ---
class WhatIfRequest(BaseModel):
    target_risk_tiers: List[str] = ["Critical", "High"]
    retention_budget: float = 65000.0
    discount_incentive_pct: float = 12.0
    execution_intensity: float = 1.0

class ExecutePlaybookRequest(BaseModel):
    customer_id: str
    playbook_id: str

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None

advisor_bot = RiskAdvisorBot(STATE)

# --- Endpoints ---

@app.get("/api/overview")
def get_overview_kpis():
    df = STATE["df_customers"]
    churn_eng = STATE["churn_engine"]
    
    total_customers = len(df)
    active_customers = int((df['churn_probability'] < 0.70).sum())
    total_clv = float(df['predicted_clv_12m'].sum())
    total_revenue_at_risk = float(df['clv_at_risk'].sum())
    avg_churn_prob = float(df['churn_probability'].mean() * 100)
    avg_clv = float(df['predicted_clv_12m'].mean())
    
    # Net Retention Rate (NRR) approximation
    expansion_revenue = df[df['telemetry_velocity_pct'] > 10]['monthly_contract_value'].sum() * 0.15
    churn_loss = df[df['churn_probability'] >= 0.70]['monthly_contract_value'].sum()
    base_mrr = df['monthly_contract_value'].sum()
    nrr = round(float(((base_mrr + expansion_revenue - churn_loss) / max(base_mrr, 1.0)) * 100), 1)
    
    # Segments breakdown
    segment_counts = df['clv_segment'].value_counts().to_dict()
    segment_clv = df.groupby('clv_segment')['predicted_clv_12m'].sum().round(2).to_dict()
    
    # Risk Tier breakdown
    risk_counts = df['risk_tier'].value_counts().to_dict()
    risk_revenue = df.groupby('risk_tier')['clv_at_risk'].sum().round(2).to_dict()
    
    # Industry distribution
    industry_dist = df['industry'].value_counts().to_dict()
    
    # Value Concentration (Top 20% Pareto analysis)
    sorted_clv = df['predicted_clv_12m'].sort_values(ascending=False).values
    top_20_pct_count = int(total_customers * 0.20)
    top_20_clv_share = round(float((sorted_clv[:top_20_pct_count].sum() / max(1.0, total_clv)) * 100), 1)

    return {
        "kpis": {
            "total_accounts": total_customers,
            "active_accounts": active_customers,
            "total_predicted_clv": round(total_clv, 2),
            "avg_predicted_clv": round(avg_clv, 2),
            "total_revenue_at_risk": round(total_revenue_at_risk, 2),
            "avg_churn_probability": round(avg_churn_prob, 1),
            "net_retention_rate_nrr": nrr,
            "top_20_pareto_clv_share_pct": top_20_clv_share
        },
        "model_metrics": churn_eng.metrics,
        "feature_importances": churn_eng.feature_importances_global,
        "segments": {
            "counts": segment_counts,
            "clv_value": segment_clv
        },
        "risk_tiers": {
            "counts": risk_counts,
            "revenue_at_risk": risk_revenue
        },
        "industries": industry_dist,
        "dataset_type": STATE["dataset_type"]
    }

@app.get("/api/customers")
def get_customers(
    risk_tier: Optional[str] = "All",
    tier: Optional[str] = "All",
    industry: Optional[str] = "All",
    search: Optional[str] = None,
    sort_by: Optional[str] = "clv_at_risk",
    order: Optional[str] = "desc",
    page: int = Query(1, ge=1),
    limit: int = Query(15, ge=5, le=100)
):
    df = STATE["df_customers"].copy()
    
    if risk_tier and risk_tier != "All":
        df = df[df['risk_tier'] == risk_tier]
    if tier and tier != "All":
        df = df[df['tier'] == tier]
    if industry and industry != "All":
        df = df[df['industry'] == industry]
    if search:
        s = search.lower().strip()
        df = df[
            df['customer_id'].str.lower().str.contains(s) |
            df['company_name'].str.lower().str.contains(s) |
            df['contact_name'].str.lower().str.contains(s) |
            df['email'].str.lower().str.contains(s)
        ]
        
    ascending = (order.lower() == "asc")
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    else:
        df = df.sort_values(by='clv_at_risk', ascending=False)
        
    total_records = len(df)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_data = df.iloc[start_idx:end_idx].to_dict(orient="records")
    
    return {
        "page": page,
        "limit": limit,
        "total_records": total_records,
        "total_pages": int(np.ceil(total_records / limit)),
        "data": page_data
    }

@app.get("/api/customer/{customer_id}")
def get_customer_detail(customer_id: str):
    df = STATE["df_customers"]
    df_tx = STATE["df_transactions"]
    
    match = df[df['customer_id'] == customer_id]
    if len(match) == 0:
        raise HTTPException(status_code=404, detail="Customer ID not found")
        
    cust = match.iloc[0].to_dict()
    
    # Fetch customer's transactions
    txs = df_tx[df_tx['customer_id'] == customer_id].sort_values(by='date', ascending=False).head(24).to_dict(orient="records")
    cust['transactions_history'] = txs
    
    # Calculate dimensional health scores (0-100)
    health_radar = {
        "Telemetry Activity": int(np.clip(100 - (cust['days_since_last_login'] * 2.5), 5, 100)),
        "Adoption Depth": int(cust['feature_adoption_score']),
        "Support Health": int(np.clip(100 - (cust['p1_critical_tickets'] * 40 + cust['open_tickets'] * 12), 10, 100)),
        "Executive Sentiment": int(cust['csat_score'] * 20),
        "Commercial Value": int(np.clip((cust['monthly_contract_value'] / 15000) * 100, 15, 100))
    }
    cust['health_radar'] = health_radar
    
    return cust

@app.post("/api/simulate-uplift")
def simulate_uplift(req: WhatIfRequest):
    df = STATE["df_customers"]
    results = RetentionOptimizer.simulate_what_if_scenario(
        df=df,
        target_risk_tiers=req.target_risk_tiers,
        retention_budget=req.retention_budget,
        discount_incentive_pct=req.discount_incentive_pct,
        execution_intensity=req.execution_intensity
    )
    return results

@app.post("/api/execute-playbook")
def execute_playbook(req: ExecutePlaybookRequest):
    df = STATE["df_customers"]
    idx = df.index[df['customer_id'] == req.customer_id].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    i = idx[0]
    pb_key = req.playbook_id
    if pb_key not in PLAYBOOK_CATALOG:
        pb_key = "EXECUTIVE_SPONSOR"
        
    pb_meta = PLAYBOOK_CATALOG[pb_key]
    reduction = pb_meta['churn_reduction_pct'] / 100.0
    
    old_prob = float(df.at[i, 'churn_probability'])
    new_prob = round(old_prob * (1.0 - reduction), 4)
    
    # Update row state
    df.at[i, 'churn_probability'] = new_prob
    df.at[i, 'risk_tier'] = "Low" if new_prob < 0.20 else ("Medium" if new_prob < 0.45 else "High")
    df.at[i, 'p1_critical_tickets'] = 0  # Resolved critical ticket
    df.at[i, 'days_since_last_login'] = 1  # Re-engaged
    df.at[i, 'clv_at_risk'] = round(df.at[i, 'predicted_clv_12m'] * new_prob, 2)
    
    # Add intervention tag
    rec_pb = df.at[i, 'recommended_playbook'].copy()
    rec_pb['status'] = 'Active Intervention Deployed'
    df.at[i, 'recommended_playbook'] = rec_pb
    
    return {
        "status": "success",
        "message": f"Playbook '{pb_meta['title']}' successfully activated for {df.at[i, 'company_name']}.",
        "customer_id": req.customer_id,
        "old_churn_probability": old_prob,
        "new_churn_probability": new_prob,
        "saved_equity_clv": round((old_prob - new_prob) * df.at[i, 'predicted_clv_12m'], 2),
        "new_risk_tier": df.at[i, 'risk_tier']
    }

@app.get("/api/cohorts")
def get_cohorts():
    return STATE["cohort_data"]

@app.get("/api/intraday/schedule")
def get_intraday_schedule():
    return STATE["intraday_schedule"]

@app.get("/api/intraday/slot")
def get_intraday_slot(day: str = Query("Monday"), hour: int = Query(9)):
    sched = STATE.get("intraday_schedule")
    if not sched or "schedule" not in sched:
        raise HTTPException(status_code=503, detail="Intraday engine not yet initialized")
    day_slots = sched["schedule"].get(day)
    if not day_slots or hour not in day_slots:
        raise HTTPException(status_code=400, detail=f"Invalid day '{day}' or hour '{hour}' (Operating days: Mon-Sat, 9AM-4PM)")
    return day_slots[hour]

@app.post("/api/regenerate-data")
def regenerate_data(dataset_type: str = Body("B2B SaaS (Enterprise)", embed=True)):
    STATE["dataset_type"] = dataset_type
    initialize_engine(n_customers=1200, seed=int(np.random.randint(10, 9999)))
    return {"status": "success", "message": f"Regenerated 1,200 accounts for {dataset_type}"}

@app.post("/api/chat")
def chat_with_risk_advisor(req: ChatRequest):
    return advisor_bot.answer_query(req.message, req.customer_id)

# Serve Frontend static assets
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, "public")
if not os.path.exists(frontend_dir) or not os.path.exists(os.path.join(frontend_dir, "index.html")):
    frontend_dir = os.path.join(base_dir, "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "OmniPulse Engine API active. Ready to serve predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
