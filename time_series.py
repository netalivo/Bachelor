import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot as pd_lag_plot
from model_SARIMA import find_SARIMA, build_SARIMA
from model_naiv import build_naive_model
from model_SARIMA import optimal_orders
from sklearn.metrics import mean_squared_error

# SARIMA-Modelle für alle Stores erstellen
def sarima_for_all_stores(filename):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()

    sarima_models = {}
    residuals_dict = {}
    
    for store in range(1, 46):
        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales'].asfreq('W-FRI')
    
        params = optimal_orders.get(str(store))
        if params:
            order = tuple(params["order"])
            seasonal_order = tuple(params["seasonal_order"])
            try:
                model_fit = build_SARIMA(sales, order, seasonal_order)
                sarima_models[store] = model_fit
                fitted_values = model_fit.fittedvalues
                residuals = sales - fitted_values
                residuals_dict[store] = residuals

                print(f"Store {store}: Modell erstellt mit Order {order} und Seasonal Order {seasonal_order}")
            except Exception as e:
                print(f"Fehler bei Store {store}: {e}")
        else:
            print(f"Keine Parameter für Store {store} gefunden.")
            
    return sarima_models

# SARIMA-Parameter für alle Stores erstellen
def sarima_params(filename):
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    for store in range(1, 46):

        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales'].asfreq('W-FRI')
        
        try:
            sarima_params = find_SARIMA(sales)
            order = sarima_params.order
            seasonal_order = sarima_params.seasonal_order
            print(f'{store}: Optimale Parameter für SARIMA: {order}, {seasonal_order}')
            model_fit = build_SARIMA(sales, order, seasonal_order)
            
        except Exception as e:
            print(f"Fehler bei Store {store}: {e}")
            
    return model_fit


def naive_residuals_for_all_stores(filename):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    residuals_dict = {}
    
    for store in range(1, 46):

        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        
        sales = store_df['weekly_sales'].asfreq('W-FRI')
        
        try:
            residuals, fitted_values = build_naive_model(sales)

            residuals_dict[store] = residuals
        except Exception as e:
            print(f"Fehler bei Store {store}: {e}")
            residuals_dict[store] = None
            
    return residuals_dict



def cross_validation(sales, order, seasonal_order):
    # Festlegen der ersten Trainingsgröße (z.B. 70 % der Daten)
    train_size = int(len(sales) * 0.7)
    cv_results = []  # Liste, um Ergebnisse aus jedem CV-Durchlauf zu speichern

    # Expanding Window Cross-Validation: 
    # Starte mit dem Trainingsset der Größe 'train_size' und erweitere es in jedem Schritt um einen Datenpunkt.
    for i in range(train_size, len(sales)):
        #Trainingsdaten: von Beginn der Zeitreihe bis zum aktuellen Index i
        train_data = sales.iloc[:i]
        # Testdaten: der direkt folgende Datenpunkt (One-Step-Ahead)
        test_data = sales.iloc[i:i+1]
    
        try:
            model_cv = build_SARIMA(train_data, order=order, seasonal_order=seasonal_order)
        
            #One-Step-Ahead-Prognose
            forecast = model_cv.forecast(steps=1)
            #Error berechnen
            error = test_data.iloc[0] - forecast.iloc[0]
        
            # Speichere das Datum, den tatsächlichen Wert, die Prognose und den Fehler
            cv_results.append({
                'date': sales.index[i],
                'actual': test_data.iloc[0],
                'forecast': forecast.iloc[0],
                'error': error
            })
        except Exception as e:
            print(f"Fehler bei Index {i}: {e}")
        continue

    # Ergebnisse in ein DataFrame umwandeln
    cv_df = pd.DataFrame(cv_results)

    # Berechnung des RMSE (Root Mean Squared Error) als Performance-Metrik
    rmse = np.sqrt(mean_squared_error(cv_df['actual'], cv_df['forecast']))
    print("Cross-Validation RMSE:", rmse)

    return cv_df, train_size


def cross_validation_naive(sales):
    # Definiere die Größe des Trainingsdatensatzes (z.B. 70% der Daten)
    train_size = int(len(sales) * 0.7)
    cv_results = []  # Hier speichern wir für jeden Split Datum, tatsächlichen Wert, Prognose und Fehler
    
    # Expanding Window Cross-Validation: Beginne mit train_size und erweitere in jedem Schritt um einen Datenpunkt
    for i in range(train_size, len(sales)):
        # Trainingsdaten: alle Daten bis zum aktuellen Index i
        train_data = sales.iloc[:i]
        # Testdaten: der direkt folgende Datenpunkt (One-Step-Ahead)
        test_data = sales.iloc[i:i+1]
        
        try:
            # Für das naive Modell: Prognose = letzter Wert aus den Trainingsdaten
            forecast = train_data.iloc[-1]
            
            # Fehler berechnen: tatsächlicher Wert - prognostizierter Wert
            error = test_data.iloc[0] - forecast
            
            # Ergebnisse speichern
            cv_results.append({
                'date': sales.index[i],
                'actual': test_data.iloc[0],
                'forecast': forecast,
                'error': error
            })
        except Exception as e:
            print(f"Fehler bei Index {i}: {e}")
            continue

    # Ergebnisse in ein DataFrame umwandeln
    cv_df = pd.DataFrame(cv_results)
    
    # RMSE (Root Mean Squared Error) berechnen
    rmse = np.sqrt(mean_squared_error(cv_df['actual'], cv_df['forecast']))
    print("Naive Model Cross-Validation RMSE:", rmse)
    
    # Plot: Gesamte Zeitreihe inkl. Forecasts aus der CV (nur im Testbereich)
    plt.figure(figsize=(12, 6))
    plt.plot(sales.index, sales, label='Beobachtete Werte', marker='.', linestyle='-')
    plt.plot(cv_df['date'], cv_df['forecast'], label='Prognosen (Naive CV)', marker='x', linestyle='-')
    plt.axvline(x=sales.index[train_size], color='black', linestyle='--', label='Train/Test Split')
    plt.title('One-Step-Ahead Forecasts (Naive Cross-Validation)')
    plt.xlabel('Datum')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return cv_df

def cross_validation_all_stores(filename):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    
    results_list = []
    
    for store in range(1, 46):
        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales']

        print(f"Verarbeite Store {store} (Datenlänge: {len(sales)})...")

        params = optimal_orders.get(str(store))
        order = tuple(params["order"])
        seasonal_order = tuple(params["seasonal_order"])

        cv_df = cross_validation(sales, order, seasonal_order)
        
        # Füge die Store-Nummer hinzu
        cv_df['store'] = store
        
        results_list.append(cv_df)
    
    # Kombiniere alle Ergebnisse in einem DataFrame
    all_cv_results = pd.concat(results_list)
    return all_cv_results

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


def seasonal_subseries_plot(sales, date_col='date', value_col='weekly_sales'):
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


def acf_plot(sales, lags=40):
    sales_clean = sales.dropna()

    plt.figure(figsize=(12, 6))
    plot_acf(sales, lags=30)
    plt.title('Autokorrelationsplot der Sales')
    plt.xlabel('Lags')
    plt.ylabel('Autokorrelation')
    plt.grid(True)
    plt.show()


def decompose(sales):
    decomposition = seasonal_decompose(sales, model='additive', period=52)
    fig = decomposition.plot()
    plt.show()


def lag_plot(sales, max_lag=12):
    # Erstelle ein Gitter von Subplots (z.B. 2 Zeilen, 3 Spalten für max_lag=6)
    rows = (max_lag + 2) // 3  # einfache Heuristik für die Anzahl der Zeilen
    cols = 3
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows))
    axes = axes.flatten()  # Einfacher Zugriff über ein eindimensionales Array
    
    # Erstelle für jeden Lag einen Plot
    for i in range(1, max_lag + 1):
        ax = axes[i - 1]
        plt.sca(ax)  # Setze die aktuelle Achse auf ax
        pd_lag_plot(sales, i)  # Übergib den Lag als Positionsargument
        ax.set_title(f"Lag = {i}")
    
    # Falls nicht alle Subplots benötigt werden, verstecke die übrigen
    for j in range(max_lag, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()