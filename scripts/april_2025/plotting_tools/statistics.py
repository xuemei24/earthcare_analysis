import numpy as np

def normalized_mean_bias(predicted, observed):
    predicted = np.array(predicted)
    observed = np.array(observed)

    if predicted.shape != observed.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Mask out NaNs from both arrays
    mask = ~np.isnan(predicted) & ~np.isnan(observed)
    pred_clean = predicted[mask]
    obs_clean = observed[mask]

    if obs_clean.size == 0:
        raise ValueError("No valid (non-NaN) observed values.")

    numerator = np.sum(pred_clean - obs_clean)
    denominator = np.sum(obs_clean)

    if denominator == 0:
        raise ZeroDivisionError("Sum of observed values is zero, cannot compute NMB.")

    nmb = (numerator / denominator) * 100
    return nmb

