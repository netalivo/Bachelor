import pmdarima as pm
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


# SARIMA-Parameter suchen
def find_SARIMA(sales):
    auto_model = pm.auto_arima(sales, 
                            seasonal=True, 
                            m=52,                   # Saisonalität: 52 Wochen pro Jahr
                            trace=False,             # Ausgabe des Suchprozesses
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)          # Schrittweise Suche (schneller)
    return auto_model

# SARIMA-Modell erstellen
def build_SARIMA(sales, order, seasonal_order):
    model = SARIMAX(sales, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

# SARIMA-Modelle für alle Stores erstellen
def SARIMA_for_all_stores(filename):

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

# SARIMA-Parameter für alle Stores suchen
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

# SARIMA-Parameter für alle Stores
optimal_orders = {
    "1": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "2": {"order": [4, 0, 3], "seasonal_order": [1, 0, 1, 52]},
    "3": {"order": [1, 1, 1], "seasonal_order": [2, 0, 0, 52]},
    "4": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "5": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "6": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "7": {"order": [0, 1, 1], "seasonal_order": [0, 1, 0, 52]},
    "8": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [4, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "17": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [4, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "20": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [3, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "24": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "25": {"order": [4, 0, 2], "seasonal_order": [1, 0, 1, 52]},
    "26": {"order": [4, 0, 3], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "28": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "29": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "31": {"order": [5, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [2, 1, 5], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "35": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [2, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "38": {"order": [1, 1, 0], "seasonal_order": [0, 1, 0, 52]},
    "39": {"order": [5, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "41": {"order": [4, 1, 3], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [1, 1, 0], "seasonal_order": [2, 1, 0, 52]},
    "43": {"order": [2, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "44": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "45": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}
}
