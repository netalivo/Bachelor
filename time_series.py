import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from scipy.stats import pearsonr



def seasonal_plot(sales):
    df = sales.to_frame(name='sales')
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week

    pivot = df.pivot(index='week', columns='year', values='sales')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(marker='o', ax=ax)
    
    ax.set_title("Seasonal Plot der Weekly Sales")
    ax.set_xlabel("Kalenderwoche")
    ax.set_ylabel("Weekly Sales")
    ax.legend(title="Jahr")
    
    weeks = range(int(pivot.index.min()), int(pivot.index.max()) + 1, 5)
    ax.set_xticks(weeks)
    
    plt.show()


def seasonal_subseries_plot(sales, date_col='Date', value_col='Weekly_Sales'):
    if isinstance(sales, pd.Series):
        df = sales.to_frame(name=value_col).reset_index()
        df.rename(columns={'index': date_col}, inplace=True)
    else:
        df = sales.copy()
    
    # Stelle sicher, dass das Datum ein Datetime-Typ ist
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extrahiere Jahr und Kalenderwoche
    df['year'] = df[date_col].dt.year
    # ISO-Kalenderwoche
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Sortieren
    df.sort_values([date_col], inplace=True)
    
    # FacetGrid: pro Kalenderwoche ein Panel, z.B. 8 pro Zeile
    g = sns.FacetGrid(df, col='week_of_year', col_wrap=8, sharey=True, sharex=False, height=2.5)

    def _plot_subseries(data, color=None, **kwargs):
        # Nach Jahr sortieren, dann Linienplot
        data = data.sort_values('year')
        plt.plot(data['year'], data[value_col], marker='o', color='black')
        # Horizontaler Mittelwert pro Facet (also pro Woche)
        mean_val = data[value_col].mean()
        plt.axhline(mean_val, color='blue', linestyle='--', alpha=0.7)
    
    g.map_dataframe(_plot_subseries)
    
    g.set_titles(col_template='Wk {col_name}')
    g.set_axis_labels('Year', value_col)
    
    # Gesamttitel
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Weekly Subseries Plot', fontsize=14)
    
    plt.show()

 
def scatter_plot(df, columns=None):
    # Falls keine Spalten angegeben sind, nutze alle
    if columns is None:
        columns = df.columns.tolist()
    
    # PairGrid für die gewünschten Spalten
    g = sns.PairGrid(df[columns], diag_sharey=False)

    # Untere Hälfte: Scatterplot
    g.map_lower(sns.scatterplot, s=20, alpha=0.7)

    # Diagonale: Histogramme (mit optionaler Dichtekurve)
    g.map_diag(sns.histplot, kde=True)

    # Obere Hälfte: Korrelationswerte annotieren
    def corrfunc(x, y, **kws):
        # Pearson-Korrelation
        r, p = pearsonr(x.dropna(), y.dropna())
        ax = plt.gca()
        # Text mit r-Wert
        text = f"r = {r:.2f}"
        # Signifikanzniveaus mit Sternen (optional)
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




def decompose(sales):
    decomposition = seasonal_decompose(sales, model='additive', period=52)
    fig = decomposition.plot()
    plt.show()