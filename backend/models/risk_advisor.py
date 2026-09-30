"""
PulseAI Risk Advisor - Conversational Risk Management Engine
Provides intelligent natural language diagnostics for customer risk,
CLV modeling, portfolio health audits, and retention playbooks.
"""

import re
import pandas as pd
import numpy as np

class RiskAdvisorBot:
    def __init__(self, state_ref):
        self.state = state_ref

    def answer_query(self, message: str, customer_id: str = None) -> dict:
        msg = message.lower().strip()
        df = self.state.get("df_customers")
        
        if df is None:
            return {
                "reply": "The OmniPulse intelligence engine is still initializing. Please allow a moment for customer models to load.",
                "type": "text"
            }

        # 1. Customer-specific query detection (e.g. CUST-0104 or company name)
        cust_match = re.search(r"cust-(\d{4})", msg)
        target_cust = None
        
        if cust_match:
            cid = f"CUST-{cust_match.group(1)}"
            matched = df[df['customer_id'].str.lower() == cid.lower()]
            if len(matched) > 0:
                target_cust = matched.iloc[0]
        elif customer_id:
            matched = df[df['customer_id'].str.lower() == customer_id.lower()]
            if len(matched) > 0:
                target_cust = matched.iloc[0]
        else:
            # Check for company name keywords
            for _, row in df.head(100).iterrows():
                comp_first = row['company_name'].lower().split()[0]
                if comp_first in msg and len(comp_first) > 3:
                    target_cust = row
                    break

        if target_cust is not None:
            return self._diagnose_customer(target_cust)

        # 2. Top At-Risk Accounts Query
        if any(w in msg for w in ["top at risk", "highest risk", "who is at risk", "critical accounts", "danger"]):
            critical = df.sort_values(by='clv_at_risk', ascending=False).head(4)
            reply = "### [Alert] Top High-Value Accounts at Urgent Risk\n\n"
            reply += "Here are the top accounts by **Revenue at Risk** requiring immediate retention triage:\n\n"
            cards = []
            for _, c in critical.iterrows():
                reply += f"- **{c['company_name']}** (`{c['customer_id']}`)\n"
                reply += f"  - **Tier:** {c['tier']} | **Industry:** {c['industry']}\n"
                reply += f"  - **Churn Risk:** `{(c['churn_probability']*100):.1f}%` ({c['risk_tier']})\n"
                reply += f"  - **Revenue at Risk:** **${c['clv_at_risk']:,.2f}** (Predicted CLV: ${c['predicted_clv_12m']:,.2f})\n"
                reply += f"  - **Primary Trigger:** {c['risk_drivers'][0]['factor'] if c['risk_drivers'] else 'Inactivity'}\n\n"
                cards.append({
                    "customer_id": c['customer_id'],
                    "company_name": c['company_name'],
                    "churn_probability": c['churn_probability'],
                    "clv_at_risk": c['clv_at_risk'],
                    "risk_tier": c['risk_tier']
                })
            reply += "👉 *Click on any account in the table or ask me: 'Audit CUST-XXXX' to generate an action plan.*"
            return {"reply": reply, "type": "accounts_list", "accounts": cards}

        # 3. Portfolio Overview & Health Query
        if any(w in msg for w in ["portfolio", "overview", "summary", "total risk", "how is our health", "kpi"]):
            total_equity = df['predicted_clv_12m'].sum()
            risk_equity = df['clv_at_risk'].sum()
            avg_churn = df['churn_probability'].mean() * 100
            crit_count = (df['risk_tier'] == 'Critical').sum()
            high_count = (df['risk_tier'] == 'High').sum()
            
            reply = f"### 📊 Portfolio Risk & Health Summary\n\n"
            reply += f"- **Total Predicted Customer Equity (12m):** `${total_equity:,.2f}`\n"
            reply += f"- **Revenue at Risk:** `${risk_equity:,.2f}` (`{(risk_equity/total_equity*100):.1f}%` of total equity)\n"
            reply += f"- **Average Churn Likelihood:** `{avg_churn:.1f}%` across `{len(df):,}` accounts\n"
            reply += f"- **Critical Risk Accounts:** `{crit_count}` accounts (Churn Risk ≥ 70%)\n"
            reply += f"- **High Risk Accounts:** `{high_count}` accounts (Churn Risk 45% - 70%)\n\n"
            reply += "**Risk Management Recommendation:** Focus retention budget on the top `Critical` and `High` tier accounts using the **What-If Scenario Lab** to achieve a target ~5.4x ROI."
            return {"reply": reply, "type": "summary"}

        # 4. Methodology / Formula Explanations
        if any(w in msg for w in ["how is clv calculated", "bg/nbd", "gamma gamma", "formula", "mathematics"]):
            reply = r"""### 📐 How Customer Lifetime Value (CLV) is Calculated

OmniPulse utilizes the **BG/NBD (Beta-Geometric / Negative Binomial Distribution)** and **Gamma-Gamma** models (Fader & Hardie, 2005):

1. **Transaction Velocity ($\lambda$):** While active, customer purchases follow a Poisson process with rate $\lambda$. Across the population, $\lambda \sim \text{Gamma}(r, \alpha)$.
2. **Unobserved Churn Dropout ($p$):** After each transaction, dropout occurs with latent probability $p \sim \text{Beta}(a, b)$.
3. **Probability of Being Active:**
   $$P(\text{Alive}) = \frac{1}{1 + \frac{a}{b + x - 1} \left(\frac{\alpha + T}{\alpha + t_x}\right)^{r+x}}$$
4. **Monetary Spend with Bayesian Shrinkage:** Average order value is estimated via Gamma-Gamma Bayesian shrinkage, pulling low-frequency accounts toward the population prior while honoring mature accounts' historical spend.
5. **Discounted CLV:** Discounted over 12 and 24 months with gross margins and cost of capital.
"""
            return {"reply": reply, "type": "theory"}

        # 5. Causal Uplift vs Churn
        if any(w in msg for w in ["uplift", "causal", "persuadables", "sleeping dog", "difference between churn"]):
            reply = """### ⚖️ Causal Uplift vs. Standard Churn Prediction

Standard churn models predict **who is likely to cancel**. However, **Causal Uplift Modeling** predicts **who will change their behavior because of an intervention**:

* **🎯 1. Persuadables:** Will churn if ignored, but will stay if contacted. **100% of retention budget should target this segment!**
* **🛑 2. Sleeping Dogs:** Inactive accounts on autopay. Contacting them or sending a marketing email reminds them of the unused charge, causing immediate churn!
* **💰 3. Sure Things:** High engagement loyalists who renew anyway. Offering them discounts is unnecessary margin dilution.
* **⚠️ 4. Lost Causes:** Customers with fundamental mismatch, bankruptcy, or sunsetted products. Expensive outreach yields zero ROI.
"""
            return {"reply": reply, "type": "uplift"}

        # 6. EMERGENCY REVENUE CRISIS & TROUBLE ADVISORY
        if any(w in msg for w in ["cancel", "threatening", "trouble", "crisis", "facing risk", "losing revenue", "revenue drop", "arr drop"]):
            reply = """### [EMERGENCY REVENUE RESCUE PROTOCOL]

**Situation Assessment:** The account is in imminent churn danger. Losing this account creates an immediate top-line ARR contraction and damages Net Retention Rate (NRR).

#### 🚨 3-Step Emergency Turnaround Action Plan:

1. **Immediate De-escalation Call (Within 2 Hours):**
   - *Objective:* Shift the conversation from "We want to cancel" to "Here is the operational blocker."
   - *Verbatim Script for Executive / CSM:*
     > *"Hi [Name], I completely understand your frustration regarding [issue]. Our VP of Customer Success and I reviewed your account this morning. Before we finalize any cancellation paperwork, give us 48 hours to present a customized preservation plan that addresses your exact concerns without forcing you to write off your historical data and integrations."*

2. **The Revenue Preservation Bridge (Downscale Instead of Churn):**
   - **Option A (The Maintenance Pause):** Freeze billing for 60 days while retaining account history and data access for a nominal 15% maintenance fee.
   - **Option B (The Core Tier Downscale):** Downscale to core seats, preserving 60% of ARR rather than losing 100%.
   - **Option C (Multi-Year Trade-Off):** Grant a 15% concession ONLY in exchange for a 24-month contract commitment. *Never discount without term expansion.*

3. **Executive Alignment & ROI Re-Demonstration:**
   - Deploy an emergency Solution Architect to resolve critical blockers and provide an **Executive Business Review (EBR) Value Memo** demonstrating proven cost savings.

*Ask me: "Audit [Account ID]" to evaluate account-specific contract numbers and calculate optimal concessions.*
"""
            return {"reply": reply, "type": "crisis_emergency"}

        # 7. Budget Cuts & Price Resistance
        if any(w in msg for w in ["budget cut", "too expensive", "price", "afford", "downsize", "cost reduction"]):
            reply = """### [REVENUE ADVISORY: Budget Cut & Price Resistance Defense]

**Diagnosis:** Customer is experiencing corporate cost reduction or cash flow pressure. Flat price rejection without value re-framing leads to 100% ARR forfeiture.

#### 🛡️ Commercial Defense Strategy:

1. **Conduct an ROI Value Audit:**
   - Calculate the customer's *Cost of Replacement*: migrating to a competitor or manual workflows costs an estimated $18,000 - $35,000 in engineering setup and retraining.
   - Re-frame software cost as an operational efficiency driver rather than a discretionary expense.

2. **The "Give-to-Get" Concession Framework:**
   - **Rule:** Never grant an unconditional discount; it destroys pricing power and signals that your product was overpriced.
   - **Counter-Offer A:** Offer a 12% renewal incentive in exchange for moving from monthly to annual upfront billing (accelerating immediate cash flow).
   - **Counter-Offer B:** Restructure seat tiers: reduce inactive licenses by 20% while locking in a 2-year enterprise agreement.

3. **Verbatim Negotiation Script:**
   > *"We know your team is under strict budget constraints this quarter. Rather than walking away from the workflow infrastructure you've built, let's restructure your licensing: we will trim inactive seats and defer milestone billing, keeping your monthly spend within your CFO's target while maintaining your critical operational capabilities."*
"""
            return {"reply": reply, "type": "budget_defense"}

        # 8. Competitor Threat & Poaching Defense
        if any(w in msg for w in ["competitor", "cheaper", "poach", "alternative", "switch"]):
            reply = """### [REVENUE ADVISORY: Competitor Poaching & Price War Defense]

**Diagnosis:** A competitor is targeting your account with aggressive introductory discounting or predatory pricing.

#### 🛡️ 4-Phase Counter-Strategy:

1. **Expose the "Hidden Total Cost of Ownership (TCO)":**
   - Highlight the hidden migration friction: API reconfiguration, webhook re-routing, historical telemetry loss, and 6-8 weeks of developer onboarding downtime.
   - Point out introductory discount traps: competitors offer 40% off in Year 1, followed by a mandatory 60% price increase in Year 2.

2. **Feature Superiority Audit:**
   - Benchmark the specific workflow dependencies the customer relies on that the competitor lacks (e.g. enterprise SLA, SOC2 Type II compliance, custom webhooks, dedicated CSM).

3. **Verbatim Commercial Counter-Script:**
   > *"We respect [Competitor], but their platform lacks our enterprise redundancy and API integration depth. A migration will require an estimated 120 engineering hours from your team. We value our partnership: let's match their contract value on a multi-year agreement so you get their pricing without risking a disruptive migration."*
"""
            return {"reply": reply, "type": "competitor_defense"}

        # 9. Intra-Day Temporal Risk Forecast (Mon-Sat, 9AM-4PM)
        if any(w in msg for w in ["day to day", "hour to hour", "hourly", "intraday", "monday to saturday", "9:00", "9 am", "4:00", "4 pm", "peak risk hour", "schedule"]):
            intraday = self.state.get("intraday_schedule", {})
            summary = intraday.get("summary", {})
            peak = summary.get("peak_risk_slot", {"day": "Friday", "label": "Fri 04:00 PM", "churn_risk": 32.4})
            total_vol = summary.get("total_weekly_predicted_volume", 148500)

            reply = f"""### ⏱️ Intra-Day Operational Risk & Revenue Forecaster (Mon–Sat | 09:00 AM – 04:00 PM)

**Operating Window:** Monday to Saturday across 8 business-hour windows (`09:00 AM` to `04:00 PM`, 48 distinct analytical slots).

#### 📊 Portfolio Temporal Metrics:
- **Weekly Predicted Business-Hour Revenue:** **${total_vol:,.2f}**
- **Peak Risk Window:** **{peak.get('day')} at {peak.get('label', '04:00 PM')}** (`{peak.get('churn_risk', 31.8)}%` Risk Index)
- **High Concurrency Window:** **Tuesday & Wednesday 11:00 AM – 01:00 PM** (Telemetry peak at 96+ index)
- **Weekend Risk Vulnerability:** **Saturday 09:00 AM – 04:00 PM** (Skeleton DevOps staffing makes outages 2.1x more likely to trigger immediate executive churn)

#### 🛡️ Day-to-Day Operational Triage Protocol:
1. **Monday (09:00 AM – 11:00 AM):** Weekend backlog triage. CSMs must resolve unresolved P1 escalations before customers reach frustration threshold.
2. **Tuesday – Thursday (11:00 AM – 02:00 PM):** Peak transaction throughput. Monitor API concurrency and webhook delivery latencies.
3. **Friday (02:00 PM – 04:00 PM):** Freeze deployments and conduct proactive health checks on top accounts to eliminate weekend renewal cancellations.
4. **Saturday (09:00 AM – 04:00 PM):** Automated sentinel monitoring; auto-route P1 tickets directly to on-call VP of Engineering.

👉 *Switch to the **Intra-Day Temporal Radar** workspace in the top navigation bar to explore the live 6x8 heatmap and scrub hour-by-hour.*
"""
            return {"reply": reply, "type": "intraday_forecast"}

        # 10. Playbook Explanations
        if any(w in msg for w in ["playbook", "mitigation", "strategy", "actions", "retention strategies"]):
            reply = """### 🛡️ Retention Mitigation Playbooks Catalog

OmniPulse features 5 automated Next-Best-Action (NBA) intervention playbooks:

1. **Executive Sponsor & High-Touch CSM:**
   - *Target:* Enterprise VIPs with unresolved P1 outages or CSAT drops.
   - *Expected Churn Reduction:* **-46%** | *Cost:* $450
2. **Commercial Restructure & 12% Loyalty Incentive:**
   - *Target:* High-ARR accounts approaching annual renewal cliffs.
   - *Expected Churn Reduction:* **-52%** | *Cost:* 8% ARR concession
3. **Accelerated Feature Adoption Sprint:**
   - *Target:* Accounts with low feature adoption (<35%) or telemetry drops.
   - *Expected Churn Reduction:* **-34%** | *Cost:* $95
4. **Proactive Health Check-in (Concierge):**
   - *Target:* Mid-market accounts showing early drift.
   - *Expected Churn Reduction:* **-28%** | *Cost:* $60
5. **Automated Product Re-engagement:**
   - *Target:* Growth/SMB accounts.
   - *Expected Churn Reduction:* **-16%** | *Cost:* $8
"""
            return {"reply": reply, "type": "playbooks"}

        # Default Helpful Copilot Response
        reply = f"""### 👋 PulseAI Risk Advisor at your service!

I can help you understand portfolio risk, investigate specific accounts, and design mitigation strategies. 

**Try asking me:**
- 🔍 *"Audit CUST-0104"* or *"Why is Synthetix at risk?"*
- ⚠️ *"Who are the top accounts at risk?"*
- 📊 *"Show me portfolio health summary"*
- 📐 *"How is CLV calculated in the BG/NBD model?"*
- 🛡️ *"Explain the 4 causal uplift quadrants (Persuadables vs Sleeping Dogs)"*
- 💡 *"What retention playbooks should we run this quarter?"*
"""
        return {"reply": reply, "type": "help"}

    def _diagnose_customer(self, cust) -> dict:
        cid = cust['customer_id']
        comp = cust['company_name']
        tier = cust['tier']
        clv = cust['predicted_clv_12m']
        prob = cust['churn_probability'] * 100
        risk_tier = cust['risk_tier']
        mrr = cust['monthly_contract_value']
        at_risk = cust['clv_at_risk']
        vel = cust['telemetry_velocity_pct']
        p1 = cust['p1_critical_tickets']
        days = cust['days_since_last_login']
        csat = cust['csat_score']
        pb = cust.get('recommended_playbook', {})

        drivers_md = ""
        for d in cust.get('risk_drivers', []):
            icon = "[!]" if d['type'] == 'negative' else ("[+]" if d['type'] == 'positive' else "[*]")
            drivers_md += f"- {icon} **{d['factor']}** ({d['impact']}): {d['detail']}\n"

        reply = f"""### [Audit] Risk Audit Dossier: {comp} (`{cid}`)

- **Customer Tier:** `{tier}` | **Industry:** {cust['industry']}
- **Predicted 12m CLV:** **${clv:,.2f}**
- **Monthly Contract Value (MRR):** `${mrr:,.2f}`
- **Calculated Churn Probability:** **`{prob:.1f}%`** (**{risk_tier} Risk**)
- **Revenue at Risk:** **`${at_risk:,.2f}`**

#### Key Diagnostic Risk Drivers (SHAP Attribution)
{drivers_md}

#### Prescribed Next-Best-Action (NBA)
- **Playbook:** **{pb.get('title', 'Executive Sponsor Intervention')}**
- **Intervention Channel:** `{pb.get('channel', 'White-Glove')}`
- **Expected Churn Reduction:** **`-{pb.get('churn_reduction_pct', 46)}%`**
- **Estimated Campaign Cost:** `${pb.get('intervention_cost', 450):,.2f}`
- **Projected Net Protected Equity:** **`${pb.get('saved_equity_clv', 18400):,.2f}`**

*Click below or type "Deploy playbook for {cid}" to launch mitigation.*
"""
        return {
            "reply": reply,
            "type": "customer_audit",
            "customer": {
                "customer_id": cid,
                "company_name": comp,
                "churn_probability": cust['churn_probability'],
                "clv_at_risk": at_risk,
                "risk_tier": risk_tier,
                "playbook": pb
            }
        }
