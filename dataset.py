import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr



def clean(df):
    # Check for missing values 
    df.info()

    # Check if the 'date' column is a valid date
    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        print("All values in 'date' column are valid dates.")
    except ValueError as e:
        print("Error:", e)
        print("There are non-date values present in the 'date' column.")

    # Check for duplicate values
    duplicate_values=df.duplicated().sum()
    print(f'The data contains {duplicate_values} duplicate values')


def features(df):
    features = ['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']

    plt.figure(figsize=(18, 20))

    for i, col in enumerate(features):
     # Subplots erstellen
        plt.subplot(3, 3, i+1)
        sns.histplot(data=df, x=col, kde=True)

    plt.tight_layout()
    plt.show()


def scatter_plot(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    
    g = sns.PairGrid(df[columns], diag_sharey=False)

    g.map_lower(sns.scatterplot, s=20, alpha=0.7)

    g.map_diag(sns.histplot, kde=True)

    def corrfunc(x, y, **kws):
        # Pearson-Korrelation
        r, p = pearsonr(x.dropna(), y.dropna())
        ax = plt.gca()
        # Text mit r-Wert
        text = f"r = {r:.2f}"
        # Signifikanzniveaus mit Sternen
        if p < 0.001:
            text += "***"
        elif p < 0.01:
            text += "**"
        elif p < 0.05:
            text += "*"
        ax.annotate(text, xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center', fontsize=12)

    g.map_upper(corrfunc)

    # Layout anpassen und Plot anzeigen
    plt.tight_layout()
    plt.show()


def boxplot(df):
    plt.figure(figsize=(3, 6))
    sns.boxplot(df['weekly_sales'])
    plt.title('Boxplot of Weekly Sales')
    plt.xlabel('Weekly Sales')
    plt.grid(True)