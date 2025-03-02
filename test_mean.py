from scipy.stats import ttest_1samp, wilcoxon, binomtest, kstest


def t_test(residuals):
    resid_clean = residuals.dropna()

    t_stat, p_value = ttest_1samp(resid_clean, popmean=0)

    return t_stat, p_value

def wilcoxon_test(residuals):
    resid_clean = residuals.dropna()

    w_stat, p_value = wilcoxon(resid_clean)

    return w_stat, p_value

def binomial_test(residuals):
    resid_clean = residuals.dropna()

    n_total = len(resid_clean)
    n_pos = sum(resid_clean > 0)
    p_value = binomtest(n_pos, n_total, p=0.5).pvalue

    return n_total, n_pos, p_value

def kolmogorov_test(residuals):
    resid_clean = residuals.dropna()

    mean_resid = resid_clean.mean()
    std_resid = resid_clean.std()
    k_stat, p_value = kstest(resid_clean, 'norm', args=(mean_resid, std_resid))

    return k_stat, p_value