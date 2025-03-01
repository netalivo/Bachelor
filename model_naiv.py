def build_naive_model(sales):

    fitted_values = sales.shift(1)
    residuals = sales - fitted_values
    return residuals, fitted_values