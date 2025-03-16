from scipy.stats import ttest_1samp, wilcoxon, binomtest, kstest


def t_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    t_stat, t_pvalue = ttest_1samp(resid_clean, popmean=0)

    if print_results:
        print(f"T-Test: {t_pvalue:.4f}")

    return t_stat, t_pvalue


def wilcoxon_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    w_stat, w_pvalue = wilcoxon(resid_clean)

    if print_results:
        print(f"Wilcoxon Test: {w_pvalue:.4f}")

    return w_stat, w_pvalue


def binomial_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    n_total = len(resid_clean)
    n_pos = sum(resid_clean > 0)
    b_pvalue = binomtest(n_pos, n_total, p=0.5).pvalue

    if print_results:
        print(f"Binomial Test: {b_pvalue:.4f}")

    return n_total, n_pos, b_pvalue


def kolmogorov_test(residuals, print_results=True):
    resid_clean = residuals.dropna()

    mean_resid = resid_clean.mean()
    std_resid = resid_clean.std()
    k_stat, k_pvalue = kstest(resid_clean, 'norm', args=(mean_resid, std_resid))

    if print_results:
        print(f"Kolmogorov Test: {k_pvalue:.4f}")
        
    return k_stat, k_pvalue