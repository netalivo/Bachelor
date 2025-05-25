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


# if all autocorrelations are within the threshold limits, 
# indicates that the residuals are behaving like white noise
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

    sarima_params  = optimal_orders_70.get(str(store_num))
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

    sarima_params  = optimal_orders_70.get(str(store_num))
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

    sarima_params  = optimal_orders_70.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    # ACF bis Lag m berechnen
    pacf_vals = pacf(resid, nlags=m, method='ywadjusted')
    # r[0] = PACF bei Lag 1, r[1] = Lag 2 usw
    r = pacf_vals[1:]  # Länge = m

    # Monti Teststatistik berechnen
    Q_M = 0.0
    for k in range(1, m+1):
        # r[k-1] ist die partielle Autokorrelation bei Lag k
        # Wenn n-k <= 0, ist die Datenlänge zu klein für k
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

    sarima_params  = optimal_orders_70.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    
    # (P)ACF bis Lag m berechnen
    if version == "acf":
        acf_vals = acf(resid, nlags=m, fft=False, adjusted=False, missing='none')
    elif version == "pacf":
        acf_vals = pacf(resid, nlags=m, method='ywadjusted')

    r = acf_vals[1:]  # ignoriert den Lag-0-Wert
    
    # Fisher-Teststatistik berechnen:
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


def breusch_godfrey_test(model, lags=29, print_results=True):

    lm_stat, lm_pvalue, _, _ = acorr_breusch_godfrey(model, nlags=lags)

    if print_results:
        print(f"Breusch Godfrey: {lm_pvalue:.4f}")

    return lm_stat, lm_pvalue


def breusch_godfrey_test_naive(sales, lags=5, print_results=True):
    sales = np.asarray(sales)
    if len(sales) <= 52 + lags:
        raise ValueError("Zeitreihe ist zu kurz für die gewählte Saisonalität und Lags.")
    
    # Vorhersage nach saisonal naivem Modell
    y_hat = np.roll(sales, 52)
    y_hat[:52] = np.nan  # Vorhersage nicht möglich für erste 52 Werte

    # Residuen
    residuals = sales - y_hat
    residuals = residuals[52:]  # NaNs am Anfang entfernen

    # Dataframe mit Residuen und ihren Lags
    df = pd.DataFrame({'resid': residuals})
    for i in range(1, lags + 1):
        df[f'resid_lag{i}'] = df['resid'].shift(i)

    df = df.dropna()
    
    y_aux = df['resid']
    X_aux = df.drop(columns='resid')
    X_aux = sm.add_constant(X_aux)

    # Hilfsregression
    aux_model = sm.OLS(y_aux, X_aux).fit()
    R2 = aux_model.rsquared
    n = len(y_aux)
    bg_stat = n * R2
    bg_pvalue = 1 - chi2.cdf(bg_stat, df=lags)

    if print_results:
        print(f"Breusch Godfrey: {bg_pvalue:.4f}")

    return bg_stat, bg_pvalue


def breusch_godfrey_oos(residuals, lags=5):
    resid = np.asarray(residuals)[~np.isnan(residuals)]
    df = pd.DataFrame({'resid': resid})
    for i in range(1, lags+1):
        df[f'lag{i}'] = df['resid'].shift(i)
    df = df.dropna()
    y_aux = df['resid']
    X_aux = sm.add_constant(df.drop(columns='resid'))
    aux = sm.OLS(y_aux, X_aux).fit()
    n = len(y_aux)
    bg_stat = n * aux.rsquared
    bg_pvalue = 1 - chi2.cdf(bg_stat, df=lags)
    return bg_stat, bg_pvalue



def pena_rodriguez_test_original(residuals, store_num, model, m=29, print_results=True):

    sarima_params  = optimal_orders_70.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    residuals = np.asarray(residuals)

    # Residuen bereinigen
    resid = np.asarray(residuals, float)
    resid = resid[~np.isnan(resid)]
    n = resid.size

    # Sample-ACF bis Lag m
    acf_vals = acf(resid, nlags=m, fft=False)
    r = acf_vals[1:]                           

    # Autokorrelationsmatrix und D_m
    Rm = toeplitz(np.r_[1.0, r])                
    det_Rm = np.linalg.det(Rm)
    D_stat = n * (1.0 - det_Rm**(1.0/m))         

    if model == "SARIMA":
    # Gamma-Parameter
        delta = (m + 1) - 2*(order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    # gemeinsamer Nenner B = 2[(m+1)(2m+1) − 6m(p+q+P+Q)]
        B = (m+1)*(2*m+1) - 6*m*(order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    if model == "Additive":
        delta = (m + 1)
        B = (m+1)*(2*m+1)

    alpha = (3*m * delta**2) / (2 * B)          # Shape (α)
    beta_rate = (3*m * delta) / B               # Rate  (β)

    scale = 1.0 / beta_rate
    p_value = 1 - gamma.cdf(D_stat, a=alpha, scale=scale)

    if print_results:
        print(f"Pena Rodriguez Test: {p_value:.4f}")
    return D_stat, p_value


def pena_rodriguez_test_mc(residuals, m=29, mc_runs=1000, random_state=None, print_results=True):

    #mit p‑Wert via Monte‑Carlo‑Simulation


    # Residuen bereinigen
    resid = np.asarray(residuals, float)
    resid = resid[~np.isnan(resid)]
    n = resid.size

    # Hilfsfunktion zur Berechnung von D_m
    def compute_D(x):
        acf_vals = acf(x, nlags=m, fft=False, adjusted=False, missing='none')[1:]      
        Rm = toeplitz(np.r_[1.0, acf_vals])            
        return n * (1.0 - np.linalg.det(Rm)**(1.0/m))  

    # Beobachtete Teststatistik
    D_obs = compute_D(resid)

    # Monte‑Carlo‑Simulation
    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(mc_runs):
        sim = rng.standard_normal(n)
        D_sim = compute_D(sim)
        if D_sim >= D_obs:
            count += 1

    # p‑Wert mit +1‑Korrektur
    p_mc = (count + 1) / (mc_runs + 1)

    if print_results:
        print(f"Pena Rodriguez (m={m}, N={mc_runs}): "
              f"{p_mc:.4f}")
              
    return D_obs, p_mc



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




