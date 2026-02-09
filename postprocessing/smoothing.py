# postprocessing/smoothing.py

def smooth_moving_average(values, window=5):
    """
    Smooth signal using a moving average.
    Reduces noise.
    """
    smoothed = []

    for i in range(len(values)):
        window_vals = [
            v for v in values[max(0, i - window): i + 1]
            if v is not None
        ]

        if window_vals:
            smoothed.append(sum(window_vals) / len(window_vals))
        else:
            smoothed.append(None)

    return smoothed
