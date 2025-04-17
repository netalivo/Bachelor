import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

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
def acf_resid_plot(residuals, lags=29):
    resid_clean = residuals.dropna()

    plt.figure(figsize=(12, 6))
    plot_acf(resid_clean, lags = lags)
    plt.title('Autokorrelationsplot der Residuen')
    plt.xlabel('Lags')
    plt.ylabel('Autokorrelation')
    plt.grid(True)
    plt.show()


def box_pierce_test(residuals, store_num, model, lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    if model == "SARIMA":
        freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    if model == "Naive":
        freedom = 0

    bp_results = acorr_ljungbox(resid_clean, lags, boxpierce=True, model_df=freedom, period = 52, return_df=True)

    bp_stats = bp_results['bp_stat']
    bp_pvalues = bp_results['bp_pvalue'] 

    bp_stat = bp_stats.iloc[-1]
    bp_pvalue = bp_pvalues.iloc[-1]

    if print_results:
        print(f"Box Pierce Test: {bp_pvalue:.4f}")

    return bp_stat, bp_pvalue


def ljung_box_test(residuals, store_num, model, lags=29, print_results=True):
    resid_clean = residuals.dropna()

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    if model == "SARIMA":
        freedom = (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    if model == "Naive":
        freedom = 0

    lb_results = acorr_ljungbox(resid_clean, lags, boxpierce=False, model_df=freedom, period = 52, return_df=True)

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

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])

    # 2) ACF bis Lag m berechnen (Lag 0 = 1)
    pacf_vals = pacf(resid, nlags=m, method='ywmle')
    # r[0] = PACF bei Lag 1, r[1] = Lag 2, usw.
    r = pacf_vals[1:]  # Länge = m

    # 3) Monti-Teststatistik berechnen:
    Q_M = 0.0
    for k in range(1, m+1):
        # r[k-1] ist die partielle Autokorrelation bei Lag k
        # Wenn n-k <= 0, ist die Datenlänge zu klein für so einen großen k
        if (n - k) > 0:
            Q_M += (r[k-1]**2) / (n - k)
    Q_M *= n * (n + 2)

    # 4) Freiheitsgrade
    if model == "SARIMA":
        df = m - (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
        if df < 1:
            df = m  # fallback
    if model == "Naive":
        df = m

    # 5) p-Wert aus der Chi-Quadrat-Verteilung
    m_pvalue = 1 - chi2.cdf(Q_M, df)
    
    if print_results:
        print(f"Monti Test: {m_pvalue:.4f}")

    return Q_M, m_pvalue

#TODO: PACF?
def fisher_test(residuals, store_num, model, version, m, print_results=True):

    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    
    # Berechne die ACF bis Lag m (Lag 0 = 1, daher r[0] entspricht Lag 1)
    if version == "acf":
        acf_vals = acf(resid, nlags=m, fft=False)
    elif version == "pacf":
        acf_vals = pacf(resid, nlags=m)

    r = acf_vals[1:]  # ignoriert den Lag-0-Wert
    
    Q_R = 0.0
    for k in range(1, m+1):
        # Vermeide Division durch 0, falls m - k == 0 (k == m)
        if (m - k) == 0:
            continue
        weight = (m - k - 1) / (m - k)
        Q_R += weight * r[k-1]**2 / (n - k)
    Q_R *= n * (n + 2)
    
    # Freiheitsgrade: df = m - (p+q); falls df < 1, setze df = m
    if model == "SARIMA":
        df = m - (order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
        if df < 1:
            df = m  # fallback
    if model == "Naive":
        df = m
    
    p_value = 1 - chi2.cdf(Q_R, df)
    
    if print_results:
        print(f"Fisher Test: {p_value:.4f}")
    
    return Q_R, p_value


def breusch_godfrey_test(model, lags=29, print_results=True):

    _, _, bg_stat, bg_pvalue = acorr_breusch_godfrey(model, nlags=lags)

    if print_results:
        print(f"Breusch Godfrey: {bg_pvalue:.4f}")

    return bg_stat, bg_pvalue


def breusch_godfrey_test_naive(sales, lags=5, print_results=True):
    sales = np.asarray(sales)
    if len(sales) <= 52 + lags:
        raise ValueError("Zeitreihe ist zu kurz für die gewählte Saisonalität und Lags.")
    
    # Vorhersage nach saisonal naivem Modell
    y_hat = np.roll(sales, 52)
    y_hat[:52] = np.nan  # Vorhersage nicht möglich für erste s Werte

    # Residuen
    residuals = sales - y_hat
    residuals = residuals[52:]  # NaNs am Anfang entfernen

    # DataFrame mit Residuen und ihren Lags
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

def mahdi_mcleod_test():
    return None

def arranz_test():
    return None

def pena_rodriguez_test(residuals, store_num, lags=29, print_results=True):

    sarima_params  = optimal_orders_5.get(str(store_num))
    order = tuple(sarima_params["order"])
    seasonal_order = tuple(sarima_params["seasonal_order"])
    residuals = np.asarray(residuals)


    n = residuals.size
    m = lags
    total_params = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]

    # 1) Autokorrelationsvektor bis Lag m
    acfs = [1.0] + [
        np.corrcoef(residuals[:-k], residuals[k:])[0,1]
        for k in range(1, m+1)
    ]
    # 2) Autokorrelationsmatrix
    Rm = toeplitz(acfs)
    # 3) Teststatistik
    detRm = np.linalg.det(Rm)
    D_stat = n * (1 - detRm**(1.0 / m))


    # Gamma-Approximation mit Berücksichtigung von p+q+P+Q
    mu  = (m + 1) / 2.0 - total_params
    var = (m + 1) * (2*m + 1) / (3.0 * m) - 2 * total_params
    alpha = mu * mu / var
    beta  = var / mu
    p_value = 1 - gamma.cdf(D_stat, a=alpha, scale=beta)
    
    if print_results:
        print(f"Pena Rodriguez Test: {p_value:.4f}")
    return D_stat, p_value







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




