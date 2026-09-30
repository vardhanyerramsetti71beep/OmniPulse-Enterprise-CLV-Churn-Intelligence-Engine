"""
Cohort Retention & Revenue Decay Matrix Engine
Calculates monthly cohort retention percentages and revenue progression.
"""

import pandas as pd
import numpy as np

class CohortEngine:
    @staticmethod
    def compute_cohort_matrix(df_tx: pd.DataFrame, df_cust: pd.DataFrame) -> dict:
        df_tx = df_tx.copy()
        
        # Merge signup cohort
        cust_cohort_map = df_cust.set_index('customer_id')['signup_date'].apply(lambda d: str(d)[:7]).to_dict()
        df_tx['cohort'] = df_tx['customer_id'].map(cust_cohort_map)
        
        # Calculate cohort index (period difference in months)
        df_tx['period'] = df_tx['period_month']
        
        # Convert cohort & period to datetime for month diff
        cohort_dt = pd.to_datetime(df_tx['cohort'] + "-01")
        period_dt = pd.to_datetime(df_tx['period'] + "-01")
        df_tx['cohort_index'] = (period_dt.dt.year - cohort_dt.dt.year) * 12 + (period_dt.dt.month - cohort_dt.dt.month)
        
        # Filter valid cohort index >= 0 and <= 11
        df_tx = df_tx[(df_tx['cohort_index'] >= 0) & (df_tx['cohort_index'] <= 11)]
        
        # Group by cohort and cohort_index
        cohort_counts = df_tx.groupby(['cohort', 'cohort_index'])['customer_id'].nunique().unstack(fill_value=0)
        
        # Sort cohorts ascending and take recent 10 cohorts
        cohort_counts = cohort_counts.sort_index().tail(10)
        
        cohort_sizes = cohort_counts[0] if 0 in cohort_counts.columns else df_tx.groupby('cohort')['customer_id'].nunique()
        
        retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100
        retention_matrix = retention_matrix.round(1)
        
        matrix_list = []
        for cohort_name, row in retention_matrix.iterrows():
            matrix_list.append({
                "cohort": cohort_name,
                "cohort_size": int(cohort_sizes.get(cohort_name, 0)),
                "retention_rates": [float(val) if not np.isnan(val) else None for val in row.values[:12]]
            })
            
        # Average retention curve across all cohorts
        avg_curve = []
        for col in range(min(12, retention_matrix.shape[1])):
            valid_vals = retention_matrix[col].dropna()
            avg_curve.append(round(float(valid_vals.mean()), 1) if len(valid_vals) > 0 else 0.0)
            
        return {
            "matrix": matrix_list,
            "average_curve": avg_curve,
            "period_labels": [f"Month {m}" for m in range(len(avg_curve))]
        }

if __name__ == "__main__":
    from backend.data_generator import generate_enterprise_saas_data
    df_c, df_t = generate_enterprise_saas_data(300)
    res = CohortEngine.compute_cohort_matrix(df_t, df_c)
    print("Cohort Matrix Count:", len(res['matrix']))
    print("Avg Curve:", res['average_curve'])
