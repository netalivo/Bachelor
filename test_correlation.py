import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tsa.stattools import pacf
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
def acf_resid_plot(residuals, lags=40):
    resid_clean = residuals.dropna()

    plt.figure(figsize=(12, 6))
    plot_acf(resid_clean, lags=30)
    plt.title('Autokorrelationsplot der Residuen')
    plt.xlabel('Lags')
    plt.ylabel('Autokorrelation')
    plt.grid(True)
    plt.show()


# TODO: Degrees of freedom anpassen


def box_pierce_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    bp_results = acorr_ljungbox(resid_clean, lags=[10], boxpierce=True, return_df=True)

    bp_stat = bp_results.loc[10, 'bp_stat']
    bp_pvalue = bp_results.loc[10, 'bp_pvalue'] 

    if print_results:
        print(f"Box Pierce: {bp_pvalue:.4f}")

    return bp_stat, bp_pvalue


def ljung_box_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    lb_results = acorr_ljungbox(resid_clean, lags=[10], boxpierce=False, return_df=True)

    lb_stat = lb_results.loc[10, 'lb_stat']
    lb_pvalue = lb_results.loc[10, 'lb_pvalue'] 
    
    if print_results:
        print(f"Ljung Box: {lb_pvalue:.4f}")

    return lb_stat, lb_pvalue


def durbin_watson_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    dw_stat = durbin_watson(resid_clean)
    
    if print_results:
        print(f"Durbin Watson: {dw_stat:.4f}")

    return dw_stat


def breusch_godfrey_test(residuals, print_results=True, lags=40):
    resid_clean = residuals.dropna()

    # Erstelle ein DataFrame mit den lagged residuals
    lagged_data = pd.concat([resid_clean.shift(i) for i in range(1, lags+1)], axis=1)
    lagged_data.columns = [f'lag_{i}' for i in range(1, lags+1)]

    lagged_data = lagged_data.dropna()

    # Passe die Residuen so an, dass sie zu den lagged Daten passen
    resid_aligned = resid_clean.loc[lagged_data.index]

    # Füge eine Konstante hinzu
    X = add_constant(lagged_data)

    # Schätze das OLS-Modell: Residuen ~ lagged Residuen
    ols_model = OLS(resid_aligned, X).fit()

    # Berechne die Breusch-Godfrey Teststatistik: n * R²
    bg_stat = ols_model.nobs * ols_model.rsquared

    # Berechne den p-Wert aus der Chi-Quadrat-Verteilung mit nlags Freiheitsgraden
    bg_pvalue = 1 - chi2.cdf(bg_stat, lags)

    if print_results:
        print(f"Breusch Godfrey: {bg_pvalue:.4f}")

    return bg_stat, bg_pvalue


def run_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    rt_zstat, rt_pvalue = runstest_1samp(resid_clean, correction=True)

    if print_results:
        print(f"Run Test: {rt_pvalue:.4f}")
        
    return rt_zstat, rt_pvalue


def monti_test(residuals, m = 20, p = 0, q = 0):
    if isinstance(residuals, pd.Series):
        resid = residuals.dropna().values
    else:
        resid = np.array(residuals)[~np.isnan(residuals)]
    n = len(resid)

    # 2) Partielle Autokorrelationen bis Lag m berechnen (Lag 0 = 1 ignorieren)
    pacf_vals = pacf(resid, nlags=m, method='ols')
    # r[0] = PACF bei Lag 1, r[1] = Lag 2, usw.
    r = pacf_vals[1:]  # Länge = m

    # 3) Monti-Teststatistik berechnen:
    #    Q_M = n*(n+2) * sum_{k=1..m} [ r_k^2 / (n - k) ]
    Q_M = 0.0
    for k in range(1, m+1):
        # Achtung: r[k-1] ist die partielle Autokorrelation bei Lag k
        # Wenn n-k <= 0, ist die Datenlänge zu klein für so einen großen k
        if (n - k) > 0:
            Q_M += (r[k-1]**2) / (n - k)
    Q_M *= n * (n + 2)

    # 4) Freiheitsgrade
    df = m - (p + q)
    if df < 1:
        df = m  # fallback

    # 5) p-Wert aus der Chi-Quadrat-Verteilung
    p_value = 1 - chi2.cdf(Q_M, df)

    return Q_M, p_value




