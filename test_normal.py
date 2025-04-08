import matplotlib.pyplot as plt

from scipy.stats import kstest


def hist_plot(residuals):
    resid_clean = residuals.dropna()

    plt.figure(figsize=(12, 6))
    plt.hist(resid_clean, bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogramm der Residuen')
    plt.xlabel('Residual')
    plt.ylabel('Häufigkeit')
    plt.grid(True)
    plt.show()

def kolmogorov_test_normal(residuals):
    resid_clean = residuals.dropna()

    mean_resid = resid_clean.mean()
    std_resid = resid_clean.std()
    stat, p_value = kstest(resid_clean, 'norm', args=(mean_resid, std_resid))

    return stat, p_value