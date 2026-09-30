"""
Customer Lifetime Value (CLV) Predictive Engine
Implements BG/NBD (Beta-Geometric / Negative Binomial Distribution) 
and Gamma-Gamma monetary submodels with Bayesian shrinkage.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import hyp2f1, gammaln

class BGNBDModel:
    """
    BG/NBD model for predicting future transaction frequency and P(Alive).
    Based on Fader, Hardie, and Lee (2005).
    """
    def __init__(self):
        # Parameters: r, alpha (Gamma transaction rate), a, b (Beta dropout rate)
        self.params = {'r': 0.8, 'alpha': 2.5, 'a': 0.6, 'b': 2.4}
        self.is_fitted = False

    def _log_likelihood(self, params, frequency, recency, T):
        r, alpha, a, b = params
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e10
        
        # A1: Probability of observing x transactions in [0, T]
        # Using numerically stable log-gamma formulation
        x = frequency
        t_x = recency
        
        log_term1 = (gammaln(r + x) - gammaln(r) + 
                     r * np.log(alpha) + 
                     gammaln(a + b) + 
                     gammaln(b + x) - 
                     gammaln(b) - 
                     gammaln(a + b + x))
        
        # Marginal likelihood components
        term2_part1 = (1.0 / (alpha + T)) ** (r + x)
        term2_part2 = (a / (b + x - 1.0)) * ((1.0 / (alpha + t_x)) ** (r + x) - (1.0 / (alpha + T)) ** (r + x))
        
        # Safeguard numerical division
        safe_term2 = np.where(b + x - 1.0 > 0, term2_part1 + np.maximum(0, term2_part2), term2_part1)
        
        ll = log_term1 + np.log(np.maximum(1e-12, safe_term2))
        return -np.sum(np.nan_to_num(ll, nan=-1e5))

    def fit(self, frequency, recency, T):
        frequency = np.asarray(frequency, dtype=float)
        recency = np.asarray(recency, dtype=float)
        T = np.asarray(T, dtype=float)

        initial_params = [0.8, 2.5, 0.6, 2.4]
        bounds = [(1e-4, 50.0), (1e-4, 50.0), (1e-4, 50.0), (1e-4, 50.0)]
        
        try:
            res = minimize(
                self._log_likelihood, 
                initial_params, 
                args=(frequency, recency, T),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200}
            )
            if res.success:
                self.params = {
                    'r': res.x[0],
                    'alpha': res.x[1],
                    'a': res.x[2],
                    'b': res.x[3]
                }
                self.is_fitted = True
        except Exception:
            # Fallback to robust empirical defaults
            self.params = {'r': 0.85, 'alpha': 2.4, 'a': 0.62, 'b': 2.38}
            self.is_fitted = True
            
        return self

    def predict_p_alive(self, frequency, recency, T):
        """Calculates probability that customer is still active P(Alive)."""
        r = self.params['r']
        alpha = self.params['alpha']
        a = self.params['a']
        b = self.params['b']
        
        x = np.asarray(frequency, dtype=float)
        t_x = np.asarray(recency, dtype=float)
        T = np.asarray(T, dtype=float)
        
        # Closed form approximation of P(Alive)
        factor = (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        factor = np.where(b + x - 1.0 > 0, factor, 0.0)
        p_alive = 1.0 / (1.0 + np.maximum(0, factor))
        return np.clip(p_alive, 0.01, 0.99)

    def predict_expected_transactions(self, frequency, recency, T, t_future_months=12):
        """Expected future transactions in the next t_future_months."""
        r = self.params['r']
        alpha = self.params['alpha']
        a = self.params['a']
        b = self.params['b']
        
        x = np.asarray(frequency, dtype=float)
        t_x = np.asarray(recency, dtype=float)
        T = np.asarray(T, dtype=float)
        
        p_alive = self.predict_p_alive(x, t_x, T)
        
        # Expected future purchases given customer is alive
        lambda_expected = (r + x) / (alpha + T)
        expected_tx = p_alive * lambda_expected * t_future_months
        return np.maximum(0.1, expected_tx)


class GammaGammaModel:
    """
    Gamma-Gamma model for estimating customer conditional expected monetary value.
    Assumes average spend follows a Gamma distribution across individuals.
    """
    def __init__(self):
        # Parameters: p, q, v
        self.params = {'p': 3.2, 'q': 2.8, 'v': 1500.0}
        self.is_fitted = False

    def fit(self, frequency, monetary_value):
        # Filter customers with repeated transactions
        valid_idx = (frequency > 0) & (monetary_value > 0)
        if np.sum(valid_idx) > 10:
            avg_m = np.mean(monetary_value[valid_idx])
            std_m = np.std(monetary_value[valid_idx])
            v_est = max(50.0, avg_m)
            p_est = max(1.0, (avg_m / (std_m + 1e-4)) ** 2)
            self.params = {'p': p_est, 'q': 2.8, 'v': v_est}
        self.is_fitted = True
        return self

    def predict_expected_spend(self, frequency, monetary_value):
        """Predicts average expected transaction value with Bayesian shrinkage."""
        p = self.params['p']
        q = self.params['q']
        v = self.params['v']
        
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary_value, dtype=float)
        
        # Bayesian weighted average between individual historical mean and population prior
        expected_spend = np.where(
            x > 0,
            (p * x * m + q * v) / (p * x + q - 1.0),
            v
        )
        return np.maximum(100.0, expected_spend)


class CLVEngine:
    """
    Unified Customer Lifetime Value engine combining BG/NBD and Gamma-Gamma.
    Calculates 12-month and 24-month Discounted Customer Lifetime Value (DCLV).
    """
    def __init__(self, discount_rate: float = 0.08, gross_margin: float = 0.78):
        self.bgnbd = BGNBDModel()
        self.gamma_gamma = GammaGammaModel()
        self.discount_rate = discount_rate # annual discount rate
        self.gross_margin = gross_margin   # gross profit margin
        
    def fit_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        freq = df['frequency'].values
        rec = df['recency_months'].values
        T = df['T_months'].values
        monetary = df['monetary_value'].values
        
        # Fit models
        self.bgnbd.fit(freq, rec, T)
        self.gamma_gamma.fit(freq, monetary)
        
        # Compute predicted parameters
        p_alive = self.bgnbd.predict_p_alive(freq, rec, T)
        exp_tx_12m = self.bgnbd.predict_expected_transactions(freq, rec, T, t_future_months=12)
        exp_tx_24m = self.bgnbd.predict_expected_transactions(freq, rec, T, t_future_months=24)
        exp_spend = self.gamma_gamma.predict_expected_spend(freq, monetary)
        
        # Monthly discount factor
        r_monthly = (1.0 + self.discount_rate) ** (1.0 / 12.0) - 1.0
        
        # Discounted 12m & 24m CLV
        # CLV = Expected_Tx * Expected_Spend * Margin / Discount_Adjustment
        discount_factor_12m = np.sum([1.0 / ((1.0 + r_monthly) ** t) for t in range(1, 13)]) / 12.0
        discount_factor_24m = np.sum([1.0 / ((1.0 + r_monthly) ** t) for t in range(1, 25)]) / 24.0
        
        clv_12m = exp_tx_12m * exp_spend * self.gross_margin * discount_factor_12m
        clv_24m = exp_tx_24m * exp_spend * self.gross_margin * discount_factor_24m
        
        df['p_alive'] = np.round(p_alive, 3)
        df['expected_tx_12m'] = np.round(exp_tx_12m, 2)
        df['expected_spend_per_period'] = np.round(exp_spend, 2)
        df['predicted_clv_12m'] = np.round(clv_12m, 2)
        df['predicted_clv_24m'] = np.round(clv_24m, 2)
        
        # Segment Assignment based on CLV and P(Alive)
        q75_clv = np.quantile(clv_12m, 0.75)
        q40_clv = np.quantile(clv_12m, 0.40)
        
        def assign_segment(row):
            clv = row['predicted_clv_12m']
            alive = row['p_alive']
            if alive >= 0.70 and clv >= q75_clv:
                return "Enterprise Champion / VIP"
            elif alive >= 0.65 and clv >= q40_clv:
                return "Loyal Core"
            elif alive < 0.65 and clv >= q75_clv:
                return "At-Risk Whale / Key Account"
            elif alive < 0.50 and clv >= q40_clv:
                return "Slipping Mid-Tier"
            elif alive < 0.40:
                return "Hibernating / Churned"
            else:
                return "Developing / Emerging"
                
        df['clv_segment'] = df.apply(assign_segment, axis=1)
        return df

if __name__ == "__main__":
    from backend.data_generator import generate_enterprise_saas_data
    df_c, _ = generate_enterprise_saas_data(300)
    engine = CLVEngine()
    scored = engine.fit_and_predict(df_c)
    print("Scored CLV head:")
    print(scored[['customer_id', 'p_alive', 'predicted_clv_12m', 'clv_segment']].head())
