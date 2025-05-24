import pandas as pd



def build_naive_additive(sales, sample, train_ratio=0.7):

    saison = 52
    sales = sales.copy()
    n = len(sales)
    train_size = int(train_ratio * n)

    y_train = sales.iloc[:train_size]
    y_test = sales.iloc[train_size:]
    t = len(y_train) - 1
    steps_ahead = len(y_test)

    fitted = []
    index = []

    # In-Sample Prognose:
    if sample == 'IS':
        for t in range(1, len(y_train)):
            if t > saison: # für t größer saison
                saison_delta = y_train.iloc[t - saison] - y_train.iloc[t - saison - 1]
                forecast = y_train.iloc[t - 1] + saison_delta # einfache naive Prognose + saisonale Anpassung
            else: # für t kleiner saison
                forecast = y_train.iloc[t - 1]  # einfache naive Prognose
            fitted.append(forecast)
            index.append(y_train.index[t])
        
        fitted_series = pd.Series(fitted, index=index)
        true_values = y_train.loc[fitted_series.index]
        residuals = true_values - fitted_series


    # Out-of-Sample Prognose:
    forecasts = []

    if sample == 'OOS':
        for h in range(1, steps_ahead + 1):
            if (t - saison >= 0):
                saison_delta = sales.iloc[t + h - saison] - sales.iloc[t - saison]
                forecast = y_train.iloc[t] + saison_delta
            else:
                forecast = y_train.iloc[t]  # Fallback
            forecasts.append(forecast)

        forecast_index = y_test.index[:steps_ahead]
        fitted_series = pd.Series(forecasts, index=forecast_index)
        residuals = y_test.loc[fitted_series.index] - fitted_series

    return residuals, fitted_series


def additive_for_all_stores(filename, sample):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    residuals_dict = {}
    fitted_values_dict = {}
    
    for store in range(1, 46):

        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        
        sales = store_df['weekly_sales'].asfreq('W-FRI')
        
        try:
            residuals, fitted_values = build_naive_additive(sales, sample)

            residuals_dict[store] = residuals
            fitted_values_dict[store] = fitted_values
        except Exception as e:
            print(f"Fehler bei Store {store}: {e}")
            residuals_dict[store] = None
            
    return residuals_dict, fitted_values_dict









def build_naive_model(sales, seasonal_period = 52):
    fitted_values = sales.shift(seasonal_period)
    residuals = sales - fitted_values
    return residuals.dropna(), fitted_values.dropna()


def naive_for_all_stores(filename):

    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df.columns = df.columns.str.lower()
    residuals_dict = {}
    fitted_values_dict = {}
    
    for store in range(1, 46):

        store_df = df[df['store'] == store].copy()
        store_df.sort_values('date', inplace=True)
        store_df.set_index('date', inplace=True)
        
        sales = store_df['weekly_sales'].asfreq('W-FRI')
        
        try:
            residuals, fitted_values = build_naive_model(sales)

            residuals_dict[store] = residuals
            fitted_values_dict[store] = fitted_values
        except Exception as e:
            print(f"Fehler bei Store {store}: {e}")
            residuals_dict[store] = None
            
    return residuals_dict, fitted_values_dict