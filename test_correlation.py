import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model_SARIMA import optimal_orders_5

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.stats import chi2


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
def acf_resid_plot(residuals, lags=29):
    resid_clean = residuals.dropna()

    plt.figure(figsize=(12, 6))
    plot_acf(resid_clean, lags = lags)
    plt.title('Autokorrelationsplot der Residuen')
    plt.xlabel('Lags')
    plt.ylabel('Autokorrelation')
    plt.grid(True)
    plt.show()


def box_pierce_test(residuals, store_num, lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    bp_results = acorr_ljungbox(resid_clean, lags, boxpierce=True, model_df=freedom, return_df=True)

    bp_stats = bp_results['bp_stat']
    bp_pvalues = bp_results['bp_pvalue'] 

    bp_stat = bp_stats.iloc[-1]
    bp_pvalue = bp_pvalues.iloc[-1]

    if print_results:
        print("Box Pierce Test")
        print(f"p-Wert an lag {lags}: {bp_pvalue:.4f}")
        print(f"p-Wert Median: {bp_pvalues.median()}")
        print("")

    return bp_stat, bp_pvalue


def ljung_box_test(residuals, store_num, lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    lb_results = acorr_ljungbox(resid_clean, lags, boxpierce=False, model_df=freedom, return_df=True)

    lb_stats = lb_results['lb_stat']
    lb_pvalues = lb_results['lb_pvalue']

    lb_stat = lb_stats.iloc[-1]
    lb_pvalue = lb_pvalues.iloc[-1]
    
    if print_results:
        print("Ljung Box Test")
        print(f"p-Wert an lag {lags}: {lb_pvalue:.4f}")
        print(f"p-Wert Median: {lb_pvalues.median()}")
        print("")

    return lb_stat, lb_pvalue


def breusch_godfrey_test(model, lags=29, print_results=True):

    _, _, bg_stat, bg_pvalue = acorr_breusch_godfrey(model, nlags=lags)

    if print_results:
        print(f"Breusch Godfrey: {bg_pvalue:.4f}")

    return bg_stat, bg_pvalue


#TODO: change degree of freedom for monti and fisher?
def monti_test(residuals, store_num, m, print_results = True):
    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    # 2) ACF bis Lag m berechnen (Lag 0 = 1)
    pacf_vals = pacf(resid, nlags=m, method='ols')
    # r[0] = PACF bei Lag 1, r[1] = Lag 2, usw.
    r = pacf_vals[1:]  # Länge = m

    # 3) Monti-Teststatistik berechnen:
    #    Q_M = n*(n+2) * sum_{k=1..m} [ r_k^2 / (n - k) ]
    Q_M = 0.0
    for k in range(1, m+1):
        # r[k-1] ist die partielle Autokorrelation bei Lag k
        # Wenn n-k <= 0, ist die Datenlänge zu klein für so einen großen k
        if (n - k) > 0:
            Q_M += (r[k-1]**2) / (n - k)
    Q_M *= n * (n + 2)

    # 4) Freiheitsgrade
    df = m - (order[0] + order[2])
    if df < 1:
        df = m  # fallback

    # 5) p-Wert aus der Chi-Quadrat-Verteilung
    m_pvalue = 1 - chi2.cdf(Q_M, df)
    
    if print_results:
        print(f"Monti Test: {m_pvalue:.4f}")

    return Q_M, m_pvalue


def fisher_test(residuals, store_num, m, print_results=True):

    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    
    # Berechne die ACF bis Lag m (Lag 0 = 1, daher r[0] entspricht Lag 1)
    acf_vals = acf(resid, nlags=m, fft=False)
    r = acf_vals[1:]  # ignoriert den Lag-0-Wert
    
    Q_R = 0.0
    for k in range(1, m+1):
        # Vermeide Division durch 0, falls m - k == 0 (k == m)
        if (m - k) == 0:
            continue
        weight = (m - k - 1) / (m - k)
        Q_R += r[k-1]**2 * weight
    Q_R *= n * (n + 2)
    
    # Freiheitsgrade: df = m - (p+q); falls df < 1, setze df = m
    df = m - (order[0] + order[2])
    if df < 1:
        df = m
    
    p_value = 1 - chi2.cdf(Q_R, df)
    
    if print_results:
        print(f"Fisher Test: {p_value:.4f}")
    
    return Q_R, p_value


def run_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    rt_zstat, rt_pvalue = runstest_1samp(resid_clean, correction=True)

    if print_results:
        print(f"Run Test: {rt_pvalue:.4f}")
        
    return rt_zstat, rt_pvalue


def durbin_watson_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    dw_stat = durbin_watson(resid_clean)
    
    if print_results:
        print(f"Durbin Watson: {dw_stat:.4f}")

    return dw_stat




