import numpy as np


def compute_sharpe_adjusted_values(values, risk, risk_free_rate=0.0):
    """
    :param values: List or array of expected returns
    :param risk: List or array of standard deviations (risk)
    :param risk_free_rate: Optional baseline (e.g., 0.0)
    :return: List of adjusted values
    """
    values = np.array(values)
    risk = np.array(risk)

    # Avoid division by zero
    risk = np.where(risk == 0, 1e-6, risk)

    sharpe = (values - risk_free_rate) / risk
    return sharpe.tolist()
