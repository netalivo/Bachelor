import itertools

from statsmodels.tsa.arima.model import ARIMA

# TODO: Die einzelnen Methoden produzieren Ausreißer, die nicht in die Daten passen.
#       Es ist noch unklar, ob die Methoden korrekt implementiert sind.
def grid_search(sales):
    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_order = None
    # best_model = None

    for order in pdq:
        try:
            model = ARIMA(sales, order=order)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                # best_model = results
        except Exception as e:
            # Fehler während der Modellanpassung überspringen
            continue

    if best_order is not None:
        print(f"\nBestes Modell: ARIMA{best_order} mit AIC: {best_aic:.2f}")
    else:
        print("Kein geeignetes Modell gefunden.")
        exit()

    return order

def build_ARIMA_model(sales, order):
    model = ARIMA(sales, order=order)
    model_fit = model.fit()
    fitted_values = model_fit.fittedvalues
    residuals = sales - fitted_values
    return residuals, fitted_values


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
                best_aic = results.aic
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