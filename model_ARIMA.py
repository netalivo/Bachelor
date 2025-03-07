import itertools

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm


def grid_search_and_build_model(sales):
    # Grid Search zur Optimierung der ARIMA-Parameter mittels AIC
    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_order = None
    best_model = None

    # print("Suche nach optimalen Parametern basierend auf AIC...")
    for order in pdq:
        try:
            model = ARIMA(sales, order=order)
            results = model.fit()
            # print(f"ARIMA{order} - AIC:{results.aic:.2f}  BIC:{results.bic:.2f}")
            if results.aic < best_aic:
                best_aic = results.aicc
                best_order = order
                best_model = results
        except Exception as e:
            # Fehler während der Modellanpassung überspringen
            continue

    if best_order is not None:
        print(f"\nBestes Modell: ARIMA{best_order} mit AIC: {best_aic:.2f}")
    else:
        print("Kein geeignetes Modell gefunden.")
        exit()

    # Ausgabe der Modellzusammenfassung
    # print(best_model.summary()) 
    return best_model


def find_SARIMA(sales):
    auto_model = pm.auto_arima(sales, 
                            seasonal=True, 
                            m=52,                   # Saisonalität: 52 Wochen pro Jahr
                            trace=False,             # Ausgabe des Suchprozesses
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)          # Schrittweise Suche (schneller)
    return auto_model

def build_SARIMA(sales, order, seasonal_order):
    model = SARIMAX(sales, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit