import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from model_SARIMA import *

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.stats import chi2
from scipy.linalg import toeplitz
from scipy.stats import gamma


def residual_plot(residuals):
    resid_clean = residuals.dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(resid_clean, marker='o', linestyle='-', color='purple')
    plt.title('Residuen: Differenz zwischen Beobachtungen und Fitted values')
    plt.xlabel('Datum')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.show()


def acf_resid_plot(residuals, lags, print_results):
    resid_clean = residuals.dropna()
    T = len(resid_clean)
    bound = 1.96 / np.sqrt(T)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(resid_clean, lags=lags, alpha=0.05, zero=False, ax=ax)

    # feste +-1.96/√T‑Linien
    ax.axhline(bound,  linestyle='--', linewidth=1)
    ax.axhline(-bound, linestyle='--', linewidth=1)

    ax.set_title('Autokorrelationsplot der Residuen')
    ax.set_xlabel('Lags')
    ax.set_ylabel('Autokorrelation')
    ax.grid(True)
    if print_results:
        plt.show()



def count_spikes(residuals, lags):
    resid_clean = residuals.dropna()
    T = len(resid_clean)
    bound = 1.96 / np.sqrt(T)

    acf_vals = acf(resid_clean, nlags=lags, fft=False)
    acf_lags = acf_vals[1:]

    n_pos = np.sum(acf_lags >  bound)
    n_neg = np.sum(acf_lags < -bound)
    n_total = n_pos + n_neg

    return n_total



def box_pierce_test(residuals, store_num, model, sample = "IS", lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_70_new.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    if model == "SARIMA":
        freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    if model == "Naive":
        freedom = 0
    if model == "Additive":
        freedom = 0

    if sample == "IS":
        bp_results = acorr_ljungbox(resid_clean, lags, boxpierce=True, model_df=freedom, period = 52, return_df=True)
    if sample == "OOS":
        bp_results = acorr_ljungbox(resid_clean, lags, boxpierce=True, model_df=freedom, return_df=True)

    bp_stats = bp_results['bp_stat']
    bp_pvalues = bp_results['bp_pvalue'] 

    bp_stat = bp_stats.iloc[-1]
    bp_pvalue = bp_pvalues.iloc[-1]

    if print_results:
        print(f"Box Pierce Test: {bp_pvalue:.4f}")

    return bp_stat, bp_pvalue


def ljung_box_test(residuals, store_num, model, sample = "IS", lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_70_new.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    if model == "SARIMA":
        freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    if model == "Naive":
        freedom = 0
    if model == "Additive":
        freedom = 0

    if sample == "IS":
        lb_results = acorr_ljungbox(resid_clean, lags, boxpierce=False, model_df=freedom, period = 52, return_df=True)
    if sample == "OOS":
        lb_results = acorr_ljungbox(resid_clean, lags, boxpierce=False, model_df=freedom, return_df=True)

    lb_stats = lb_results['lb_stat']
    lb_pvalues = lb_results['lb_pvalue']

    lb_stat = lb_stats.iloc[-1]
    lb_pvalue = lb_pvalues.iloc[-1]
    
    if print_results:
        print(f"Ljung Box Test: {lb_pvalue:.4f}")

    return lb_stat, lb_pvalue


def monti_test(residuals, store_num, model, m, print_results = True):
    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    sarima_params  = optimal_orders_70_new.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    # ACF bis Lag m berechnen
    pacf_vals = pacf(resid, nlags=m, method='ywadjusted')
    r = pacf_vals[1:]

    # Monti Teststatistik
    Q_M = 0.0
    for k in range(1, m+1):
        # r[k-1] ist partielle Autokorrelation bei Lag k
        if (n - k) > 0:
            Q_M += (r[k-1]**2) / (n - k)
    Q_M *= n * (n + 2)

    # Freiheitsgrade
    if model == "SARIMA":
        df = m - (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
        if df < 1:
            df = m
    if model == "Naive":
        df = m
    if model == "Additive":
        df = m

    # p-Wert aus der Chi-Quadrat-Verteilung
    m_pvalue = 1 - chi2.cdf(Q_M, df)
    
    if print_results:
        print(f"Monti Test: {m_pvalue:.4f}")

    return Q_M, m_pvalue


def fisher_test(residuals, store_num, model, version, m, print_results=True):

    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    sarima_params  = optimal_orders_70_new.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    
    # (P)ACF bis Lag m berechnen
    if version == "acf":
        acf_vals = acf(resid, nlags=m, fft=False, adjusted=False, missing='none')
    elif version == "pacf":
        acf_vals = pacf(resid, nlags=m, method='ywadjusted')

    r = acf_vals[1:]
    
    # Fisher Teststatistik
    Q_R = 0.0
    for k in range(1, m+1):
        # Vermeide Division durch 0
        if (m - k) == 0:
            continue
        weight = (m - k + 1) / m
        Q_R += weight * r[k-1]**2 / (n - k)
    Q_R *= n * (n + 2)
    
    # Freiheitsgrade
    if model == "SARIMA":
        df = m - (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
        if df < 1:
            df = m
    if model == "Naive":
        df = m
    if model == "Additive":
        df = m
    
    # p-Wert aus der Chi-Quadrat-Verteilung
    p_value = 1 - chi2.cdf(Q_R, df)
    
    if print_results:
        print(f"Fisher Test: {p_value:.4f}")
    
    return Q_R, p_value


def breusch_godfrey_test(model, lags, print_results=True):

    lm_stat, lm_pvalue, _, _ = acorr_breusch_godfrey(model, nlags=lags)

    if print_results:
        print(f"Breusch Godfrey: {lm_pvalue:.4f}")

    return lm_stat, lm_pvalue


def breusch_godfrey_manuell(errors, lags, print_results=True):

    err = np.asarray(errors)
    err = err[~np.isnan(err)]

    # Einfache Regression: err_t = const + u_t
    X = np.ones((len(err), 1))
    ols_res = sm.OLS(err, X).fit()

    # Breusch-Godfrey Test
    lm_stat, p_value, _, _ = acorr_breusch_godfrey(ols_res, nlags=lags)

    if print_results:
        print(f"Breusch–Godfrey (OOS): {p_value:.4f}")

    return lm_stat, p_value



def pena_rodriguez_test_original(residuals, store_num, model, m=29, print_results=True):

    sarima_params  = optimal_orders_70_new.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    residuals = np.asarray(residuals)

    # Residuen bereinigen
    resid = np.asarray(residuals, float)
    resid = resid[~np.isnan(resid)]
    n = resid.size

    # Sample-ACF bis Lag m
    acf_vals = acf(resid, nlags=m-1, fft=False)
    r = acf_vals[1:]                           

    # Autokorrelationsmatrix und D
    Rm = toeplitz(np.r_[1.0, r])                
    det_Rm = np.linalg.det(Rm)
    D_stat = n * (1.0 - det_Rm**(1.0/m))         

    
    if model == "SARIMA":
        p_plus_q = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]

        alpha_num   = 3 * m * ((m + 1) - 2 * p_plus_q) * ((m + 1) - 2 * p_plus_q)
        alpha_denom = 2 * (2 * (m + 1) * (2 * m + 1) - 12 * m * p_plus_q)

        beta_num   = 3 * m * ((m + 1) - 2 * p_plus_q)
        beta_denom = 2 * (m + 1) * (2 * m + 1) - 12 * m * p_plus_q
    if model == "Additive":
        alpha_num = 3*m*(m + 1)*(m + 1)
        alpha_denom = 2*(2*(m+1)*(2*m+1))

        beta_num = 3*m*(m + 1)
        beta_denom = 2*(m+1)*(2*m+1)


    alpha = alpha_num / alpha_denom
    beta_rate = beta_num / beta_denom

    scale = 1.0 / beta_rate
    p_value = 1 - gamma.cdf(D_stat, a=alpha, scale=scale)

    if print_results:
        print(f"Pena Rodriguez Test: {p_value:.4f}")
    return D_stat, p_value



def run_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    rt_zstat, rt_pvalue = runstest_1samp(resid_clean)

    if print_results:
        print(f"Run Test: {rt_pvalue:.4f}")
        
    return rt_zstat, rt_pvalue


def durbin_watson_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    dw_stat = durbin_watson(resid_clean)
    
    if print_results:
        print(f"Durbin Watson: {dw_stat:.4f}")

    return dw_stat




def compute_D(resid, m):
    n = resid.size
    acf_vals = acf(resid, nlags=m, fft=False, adjusted=False, missing='none')[1:]
    Rm = toeplitz(np.r_[1.0, acf_vals])
    sign, logdet = np.linalg.slogdet(Rm)
    return n * (1.0 - np.exp(logdet / m))

def pena_rodriguez_test_mc(y,residuals, store_num, model, m=29, mc_runs=1000, random_state=None, test_size=70, print_results=True):

    rng = np.random.default_rng(random_state)
    count = 0

    sarima_params = optimal_orders_70.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    if model == "SARIMA":
        # In-Sample-Test
        resid = np.asarray(residuals, float)[~np.isnan(residuals)]
        D_obs = compute_D(resid, m)
        n = resid.size
        sarima_mod = SARIMAX(y,
                             order=order,
                             seasonal_order=seasonal_order)
        sarima_res = sarima_mod.fit(disp=False)
        sigma2 = np.nanvar(sarima_res.resid, ddof=1)

        for _ in range(mc_runs):
            eps = rng.standard_normal(n) * np.sqrt(sigma2)
            sim = sarima_res.simulate(nsimulations=n, measurement_shocks=eps)
            sim_res = SARIMAX(sim,
                              order=order,
                              seasonal_order=seasonal_order).fit(disp=False)
            if compute_D(sim_res.resid[~np.isnan(sim_res.resid)], m) >= D_obs:
                count += 1

    elif model == "SARIMA_OOS":
        # Out-of-Sample-Test
        if test_size is None or test_size <= 0:
            raise ValueError("Für SARIMA_OOS muss test_size > 0 angegeben sein.")
        T = len(y)
        n_train = T - test_size
        y_train = np.asarray(y[:n_train], float)
        y_test = np.asarray(y[n_train:], float)

        # Fit auf Trainingsdaten
        sarima_mod = SARIMAX(y_train,
                             order=order,
                             seasonal_order=seasonal_order)
        sarima_res = sarima_mod.fit(disp=False)
        sigma2 = np.nanvar(sarima_res.resid, ddof=1)

        # Prognose Fehler
        forecast = sarima_res.get_forecast(steps=test_size).predicted_mean
        resid_oos = y_test - forecast
        D_obs = compute_D(resid_oos, m)
        n = test_size

        for _ in range(mc_runs):
            eps_all = rng.standard_normal(T) * np.sqrt(sigma2)
            sim_all = sarima_res.simulate(nsimulations=T, measurement_shocks=eps_all)

            sim_train = sim_all[:n_train]
            sim_test = sim_all[n_train:]

            sim_mod = SARIMAX(sim_train,
                              order=order,
                              seasonal_order=seasonal_order).fit(disp=False)
            sim_pred = sim_mod.get_forecast(steps=test_size).predicted_mean
            resid_sim_o = sim_test - sim_pred

            if compute_D(resid_sim_o, m) >= D_obs:
                count += 1

    elif model == "Naive":
        # Naiver Random-Walk-Test
        resid_naive = np.asarray(residuals, float)[~np.isnan(residuals)]
        D_obs = compute_D(resid_naive, m)
        n = resid_naive.size
        sigma_naive = resid_naive.std(ddof=1)

        for _ in range(mc_runs):
            eps = rng.standard_normal(n) * sigma_naive
            sim = np.empty(n + 1)
            sim[0] = y[0]
            for t in range(1, n + 1):
                sim[t] = sim[t-1] + eps[t-1]
            resid_sim = sim[1:] - sim[:-1]
            if compute_D(resid_sim, m) >= D_obs:
                count += 1

    else:
        raise ValueError(f"Unbekanntes Modell: {model}")

    p_mc = (count + 1) / (mc_runs + 1)

    if print_results:
        print(f"Pena–Rodriguez MC (Model={model}, m={m}, N={mc_runs}): p-value={p_mc:.4f}")

    return D_obs, p_mc






