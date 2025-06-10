import pmdarima as pm
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


# SARIMA-Parameter suchen
def find_SARIMA(sales):
    auto_model = pm.auto_arima(sales, 
                            seasonal=True, 
                            m=52,                   # Saisonalität: 52 Wochen pro Jahr
                            trace=True,             # Ausgabe des Suchprozesses
                            start_p=0, start_q=0,   # Parameter für AR und MA
                            max_p=5, max_q=5,
                            start_P=0, start_Q=0,
                            max_P=5, max_Q=5,      
                            error_action='ignore',
                            information_criterion='aic',  
                            suppress_warnings=True, 
                            stepwise=False)         # Schrittweise Suche?
    return auto_model



# SARIMA-Modell erstellen
def build_SARIMA(sales, order, seasonal_order):
    n = len(sales)
    split = int(n * 0.7)
    y_train = sales.iloc[:split]
    y_test = sales.iloc[split:]
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, trend='c')
    model_fit = model.fit(disp=False)
    return model_fit, y_train, y_test



# SARIMA-Modelle für alle Stores erstellen
def SARIMA_for_all_stores(filename, whichorder):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()

    sarima_models = {}
    y_test_dict = {}
    y_train_dict = {}
    
    for store in range(1, 46):
        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales'].asfreq('W-FRI')

        if whichorder == 5: params = optimal_orders_5.get(str(store))
        if whichorder == 70: params = optimal_orders_70.get(str(store))
        if whichorder == 75: params = optimal_orders_70_new.get(str(store))
        
        if params:
            order = tuple(params["order"])
            seasonal_order = tuple(params["seasonal_order"])
            try:
                model_fit, y_train, y_test = build_SARIMA(sales, order, seasonal_order)
                sarima_models[store] = model_fit
                y_train_dict[store] = y_train
                y_test_dict[store] = y_test

                print(f"Store {store}: Modell erstellt mit Order {order} und Seasonal Order {seasonal_order}")
            except Exception as e:
                print(f"Fehler bei Store {store}: {e}")
        else:
            print(f"Keine Parameter für Store {store} gefunden.")
            
    return sarima_models, y_train_dict, y_test_dict



# SARIMA-Parameter für alle Stores suchen
def sarima_params(filename):
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    
    output_file = "optimal_sarima_orders.txt"
    with open(output_file, "w") as f:
        for store in range(22, 46):
            store_df = df[df['store'] == store].copy()
            store_df.sort_values('date', inplace=True)
            store_df.set_index('date', inplace=True)
            sales = store_df['weekly_sales'].asfreq('W-FRI')
            
            n = len(sales)
            split = int(n * 0.7)
            y_train = sales.iloc[:split]
            print(f"Verarbeite Store {store} (Datenlänge: {len(sales)})...")
            
            try:
                auto_model = find_SARIMA(y_train)
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
                # Schreibe Ergebnisse zeilenweise in die Datei
                result_line = f"Store {store}: Optimale Parameter für SARIMA: {order}, {seasonal_order}\n"
                f.write(result_line)
                print(result_line, end="")

            except Exception as e:
                error_line = f"Fehler bei Store {store}: {e}\n"
                f.write(error_line)
                print(error_line, end="")
                
    return output_file


# SARIMA-Parameter für alle Stores
optimal_orders_5 = {
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

optimal_orders_10 = {
    "1": {"order": [6, 1, 0], "seasonal_order": [1, 0, 1, 52]},
    "2": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "3": {"order": [1, 1, 1], "seasonal_order": [2, 0, 0, 52]},
    "4": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "5": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "6": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "7": {"order": [0, 1, 1], "seasonal_order": [0, 1, 0, 52]},
    "8": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "9": {"order": [6, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "11": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "12": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "13": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "14": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "16": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "17": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [2, 0, 0], "seasonal_order": [0, 0, 2, 52]},
    "19": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "20": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "21": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "22": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "23": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "24": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "25": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "26": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "27": {"order": [1, 0, 1], "seasonal_order": [0, 0, 3, 52]},
    "28": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "29": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "31": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "32": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "33": {"order": [2, 1, 5], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [1, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "35": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [2, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "38": {"order": [1, 1, 0], "seasonal_order": [0, 1, 0, 52]},
    "39": {"order": [5, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]},
    "41": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [1, 1, 0], "seasonal_order": [2, 1, 0, 52]},
    "43": {"order": [2, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "44": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "45": {"order": [2, 0, 0], "seasonal_order": [0, 0, 3, 52]}
}

optimal_orders_notstepwise = {
    "1": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "2": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "3": {"order": [1, 1, 1], "seasonal_order": [2, 0, 0, 52]},
    "4": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "5": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "6": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "7": {"order": [2, 1, 3], "seasonal_order": [0, 1, 0, 52]},
    "8": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [3, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "17": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "20": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "24": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "25": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "26": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "28": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "29": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "31": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "35": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "38": {"order": [1, 1, 0], "seasonal_order": [0, 1, 0, 52]},
    "39": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "41": {"order": [3, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [2, 1, 3], "seasonal_order": [0, 1, 0, 52]},
    "43": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "44": {"order": [0, 1, 1], "seasonal_order": [3, 0, 0, 52]},
    "45": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}
}

optimal_orders_aicc = {
    "1": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "2": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "3": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "4": {"order": [2, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "5": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "6": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "7": {"order": [2, 1, 3], "seasonal_order": [0, 1, 0, 52]},
    "8": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [3, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "17": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "20": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "24": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "25": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "26": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "28": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "29": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "31": {"order": [0, 0, 4], "seasonal_order": [0, 0, 1, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "35": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "38": {"order": [1, 1, 0], "seasonal_order": [0, 1, 0, 52]},
    "39": {"order": [4, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "41": {"order": [3, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [2, 1, 3], "seasonal_order": [0, 1, 0, 52]},
    "43": {"order": [0, 1, 4], "seasonal_order": [1, 0, 0, 52]},
    "44": {"order": [0, 1, 1], "seasonal_order": [3, 0, 0, 52]},
    "45": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}
}

optimal_orders_60 = {
    "1":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "2":  {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "3":  {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "4":  {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 52]},
    "5":  {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "6":  {"order": [5, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "7":  {"order": [0, 0, 1], "seasonal_order": [0, 1, 0, 52]},
    "8":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [2, 0, 2], "seasonal_order": [0, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [0, 0, 0], "seasonal_order": [0, 1, 0, 52]},
    "17": {"order": [0, 0, 0], "seasonal_order": [0, 1, 0, 52]},
    "18": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "20": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "24": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "25": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "26": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "28": {"order": [2, 0, 0], "seasonal_order": [0, 1, 0, 52]},
    "29": {"order": [2, 0, 2], "seasonal_order": [1, 0, 1, 52]},
    "30": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "31": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "35": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [0, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "38": {"order": [0, 1, 0], "seasonal_order": [1, 1, 0, 52]},
    "39": {"order": [2, 0, 2], "seasonal_order": [1, 0, 1, 52]},
    "40": {"order": [1, 0, 0], "seasonal_order": [0, 0, 1, 52]},
    "41": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [0, 0, 0], "seasonal_order": [0, 1, 0, 52]},
    "43": {"order": [0, 1, 0], "seasonal_order": [1, 1, 0, 52]},
    "44": {"order": [1, 1, 0], "seasonal_order": [0, 0, 1, 52]},
    "45": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]}
}

optimal_orders_70 = {
    "1":  {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "2":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "3":  {"order": [5, 0, 0], "seasonal_order": [0, 0, 2, 52]},
    "4":  {"order": [0, 0, 2], "seasonal_order": [0, 1, 0, 52]},
    "5":  {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "6":  {"order": [1, 0, 5], "seasonal_order": [1, 0, 0, 52]},
    "7":  {"order": [0, 0, 1], "seasonal_order": [0, 1, 0, 52]},
    "8":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9":  {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "17": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "20": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "24": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "25": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "26": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "28": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "29": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [0, 0, 1, 52]},
    "31": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "34": {"order": [1, 0, 0], "seasonal_order": [0, 0, 2, 52]},
    "35": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]},
    "37": {"order": [1, 0, 1], "seasonal_order": [0, 1, 0, 52]},
    "38": {"order": [0, 1, 0], "seasonal_order": [1, 1, 0, 52]},
    "39": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "41": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [0, 0, 0], "seasonal_order": [0, 1, 0, 52]},
    "43": {"order": [0, 0, 1], "seasonal_order": [0, 1, 0, 52]},
    "44": {"order": [1, 1, 0], "seasonal_order": [0, 0, 1, 52]},
    "45": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}
}

optimal_orders_70_new = {
    "1":  {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "2":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "3":  {"order": [5, 0, 0], "seasonal_order": [0, 0, 2, 52]},
    "4":  {"order": [0, 1, 2], "seasonal_order": [0, 1, 0, 52]},
    "5":  {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "6":  {"order": [1, 0, 5], "seasonal_order": [1, 0, 0, 52]},
    "7":  {"order": [0, 0, 1], "seasonal_order": [0, 1, 0, 52]},
    "8":  {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "9":  {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "10": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "11": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "12": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "13": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "14": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "15": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "16": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}, 
    "17": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "18": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "19": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "20": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "21": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "22": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "23": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "24": {"order": [1, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "25": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "26": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "27": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "28": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "29": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "30": {"order": [1, 1, 0], "seasonal_order": [0, 0, 1, 52]}, 
    "31": {"order": [1, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "32": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]},
    "33": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]}, 
    "34": {"order": [1, 0, 0], "seasonal_order": [0, 0, 2, 52]}, 
    "35": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "36": {"order": [0, 1, 0], "seasonal_order": [1, 0, 0, 52]}, 
    "37": {"order": [1, 1, 1], "seasonal_order": [0, 1, 0, 52]},
    "38": {"order": [0, 1, 0], "seasonal_order": [1, 1, 0, 52]}, 
    "39": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "40": {"order": [2, 0, 1], "seasonal_order": [1, 0, 0, 52]},
    "41": {"order": [0, 1, 1], "seasonal_order": [1, 0, 0, 52]},
    "42": {"order": [0, 1, 1], "seasonal_order": [0, 1, 0, 52]},
    "43": {"order": [0, 1, 1], "seasonal_order": [0, 1, 0, 52]},
    "44": {"order": [1, 1, 0], "seasonal_order": [0, 0, 1, 52]}, 
    "45": {"order": [2, 0, 2], "seasonal_order": [1, 0, 0, 52]}
}




