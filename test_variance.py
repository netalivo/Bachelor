import numpy as np

from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.diagnostic import het_breuschpagan


def goldfeldquandt_test(residuals, fitted_values):
    resid_clean = residuals.dropna()
    fitted_clean = fitted_values.dropna()

    # Gemeinsamen Index berechnen, um nur die übereinstimmenden Beobachtungen zu verwenden:
    common_index = resid_clean.index.intersection(fitted_clean.index)
    resid_clean = resid_clean.loc[common_index]
    fitted_clean = fitted_clean.loc[common_index]

    # fitted values als exogene Variable
    # Da exog als 2D-Array erwartet wird, formen wir sie entsprechend um.
    exog = fitted_clean.values.reshape(-1, 1)

    gq_stat, gq_pvalue, ratio = het_goldfeldquandt(resid_clean, exog)

    return gq_stat, gq_pvalue


def breuschpagan_test(residuals, fitted_values):
    resid_clean = residuals.dropna()
    fitted_clean = fitted_values.dropna()

    # Gemeinsamen Index berechnen:
    common_index = resid_clean.index.intersection(fitted_clean.index)
    resid_clean = resid_clean.loc[common_index]
    fitted_clean = fitted_clean.loc[common_index]

    # fitted values als exogene Variable
    # Füge eine Konstante hinzu, da der Test einen konstanten Term erwartet.
    exog = fitted_clean.values.reshape(-1, 1)
    exog = np.hstack([np.ones((exog.shape[0], 1)), exog])

    bp_test = het_breuschpagan(resid_clean, exog, False)

    lm_stat = bp_test[0]
    lm_pvalue = bp_test[1]
    f_stat = bp_test[2]
    f_pvalue = bp_test[3]

    return lm_stat, lm_pvalue, f_stat, f_pvalue


def koenkerbasset_test(residuals, fitted_values):
    resid_clean = residuals.dropna()
    fitted_clean = fitted_values.dropna()

    # Gemeinsamen Index berechnen
    common_index = resid_clean.index.intersection(fitted_clean.index)
    resid_clean = resid_clean.loc[common_index]
    fitted_clean = fitted_clean.loc[common_index]

    # fitted values als exogene Variable
    exog = fitted_clean.values.reshape(-1, 1)
    exog = np.hstack([np.ones((exog.shape[0], 1)), exog])

    kb_test = het_breuschpagan(resid_clean, exog, True)

    lm_stat = kb_test[0]
    lm_pvalue = kb_test[1]
    f_stat = kb_test[2]
    f_pvalue = kb_test[3]

    return lm_stat, lm_pvalue, f_stat, f_pvalue