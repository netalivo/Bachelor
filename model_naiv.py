def build_naive_model(sales):
    fitted_values = sales.shift(1)
    residuals = sales - fitted_values
    return residuals, fitted_values


def build_seasonal_naive_model(sales, seasonal_period = 52):
    fitted_values = sales.shift(seasonal_period)
    residuals = sales - fitted_values
    return fitted_values, residuals