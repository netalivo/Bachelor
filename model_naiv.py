import pandas as pd



def build_naive_model(sales):
    fitted_values = sales.shift(1)
    residuals = sales - fitted_values
    return residuals, fitted_values


def build_seasonal_naive_model(sales, seasonal_period = 52):
    fitted_values = sales.shift(seasonal_period)
    residuals = sales - fitted_values
    return fitted_values, residuals


def naive_for_all_stores(filename):

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