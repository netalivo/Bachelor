import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model_SARIMA import build_SARIMA
from model_SARIMA import optimal_orders_5, optimal_orders_10
from sklearn.metrics import mean_squared_error



def cross_validation_SARIMA(sales, order, seasonal_order, print_results=True):
    # Festlegen der ersten Trainingsgröße
    train_size = int(len(sales) * 0.6)
    cv_results = [] 

    # Expanding Window Cross-Validation
    # Starte mit dem Trainingsset der Größe 'train_size' und erweitere es in jedem Schritt um einen Datenpunkt
    for i in range(train_size, len(sales)):
        # Trainingsdaten: von Beginn der Zeitreihe bis zum aktuellen Index i
        train_data = sales.iloc[:i]
        # Testdaten: der direkt folgende Datenpunkt (One-Step-Ahead)
        test_data = sales.iloc[i:i+1]
    
        try:
            model_cv = build_SARIMA(train_data, order=order, seasonal_order=seasonal_order)
        
            # One-Step-Ahead Prognose
            forecast = model_cv.forecast(steps=1)
            # Error berechnen
            error = test_data.iloc[0] - forecast.iloc[0]
        
            # Datum, den tatsächlichen Wert, die Prognose und den Fehler speichern
            cv_results.append({
                'date': sales.index[i],
                'actual': test_data.iloc[0],
                'forecast': forecast.iloc[0],
                'error': error
            })
        except Exception as e:
            print(f"Fehler bei Index {i}: {e}")
        continue

    # Ergebnisse in ein dataframe umwandeln
    cv_df = pd.DataFrame(cv_results)

    # Berechnung des RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(cv_df['actual'], cv_df['forecast']))
    print("Cross-Validation RMSE:", rmse)

    if print_results:
        #Plot: Beobachtete Werte vs. Prognosen
        plt.figure(figsize=(12, 6))
        plt.plot(sales.index, sales, label='Beobachtete Werte', marker='.', linestyle='-')
        plt.plot(cv_df['date'], cv_df['forecast'], label='Prognosen (CV)', marker='x', linestyle='-')
        plt.axvline(x=sales.index[train_size], color='black', linestyle='--', label='Train/Test Split')
        plt.title('One-Step-Ahead Forecasts (Cross-Validation)')
        plt.xlabel('Datum')
        plt.ylabel('Weekly Sales')
        plt.legend()
        plt.grid(True)
        plt.show()

    return cv_df, train_size


def cross_validation_naive(sales, seasonal_period=52, print_results=True):

    # Festlegen der ersten Trainingsgröße
    train_size = int(len(sales) * 0.6)
    cv_results = []
    
    # Expanding Window Cross-Validation
    # Starte mit dem Trainingsset der Größe 'train_size' und erweitere es in jedem Schritt um einen Datenpunkt
    for i in range(train_size, len(sales)):
        # Prüfen ob genügend Daten vorhanden sind, um die saisonale Prognose zu bilden
        if i - seasonal_period < 0:
            continue
        
        # Prognose = Wert, der seasonal_period Schritte zurückliegt
        forecast = sales.iloc[i - seasonal_period]
        test_value = sales.iloc[i]
        error = test_value - forecast
        
        # Datum, den tatsächlichen Wert, die Prognose und den Fehler speichern
        cv_results.append({
            'date': sales.index[i],
            'actual': test_value,
            'forecast': forecast,
            'error': error
        })
    
    # Ergebnisse in ein DataFrame umwandeln
    cv_df = pd.DataFrame(cv_results)
    
    # Berechne RMSE
    valid_df = cv_df.dropna(subset=['forecast'])
    rmse = np.sqrt(mean_squared_error(valid_df['actual'], valid_df['forecast']))
    print(f"Seasonal Naive Model Cross-Validation RMSE (period={seasonal_period}):", rmse)
    
    if print_results:
        # Plot: Zeige die gesamte Zeitreihe und die CV-Prognosen im Testbereich
        plt.figure(figsize=(12, 6))
        plt.plot(sales.index, sales, label='Beobachtete Werte', marker='.', linestyle='-')
        plt.plot(valid_df['date'], valid_df['forecast'], label='Prognosen (Seasonal Naive CV)', marker='x', linestyle='-')
        plt.axvline(x=sales.index[train_size], color='black', linestyle='--', label='Train/Test Split')
        plt.title(f'One-Step-Ahead Forecasts (Seasonal Naive CV, period={seasonal_period})')
        plt.xlabel('Datum')
        plt.ylabel('Weekly Sales')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return cv_df


def cv_SARIMA_all_stores(filename, whichorder):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    
    results_list = []
    
    for store in range(1, 46):
        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales']

        print(f"Verarbeite Store {store} (Datenlänge: {len(sales)})...")

        if whichorder == 5: params = optimal_orders_5.get(str(store))
        if whichorder == 10: params = optimal_orders_10.get(str(store))

        order = tuple(params["order"])
        seasonal_order = tuple(params["seasonal_order"])

        cv_df, _ = cross_validation_SARIMA(sales, order, seasonal_order, print_results=False)
        
        # Store-Nummer hinzufügen
        cv_df['store'] = store
        
        results_list.append(cv_df)
    
    # Ergebnisse in ein Dataframe umwandeln
    all_cv_results = pd.concat(results_list)
    return all_cv_results


def cv_naive_all_stores(filename):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    
    results_list = []
    
    for store in range(1, 46):
        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        sales = store_df['weekly_sales']

        print(f"Verarbeite Store {store} (Datenlänge: {len(sales)})...")

        cv_df = cross_validation_naive(sales, print_results=False)
        
        # Store-Nummer hinzufügen
        cv_df['store'] = store
        
        results_list.append(cv_df)
    
    # Ergebnisse in ein dataframe umwandeln
    all_cv_results = pd.concat(results_list)
    return all_cv_results