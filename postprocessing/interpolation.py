# postprocessing/interpolation.py

def interpolate_missing(values):
    """
    Fill short gaps (None values) using linear interpolation.
    """
    interpolated = values[:]

    for i in range(1, len(values) - 1):
        if interpolated[i] is None:
            prev_v = interpolated[i - 1]
            next_v = interpolated[i + 1]

            if prev_v is not None and next_v is not None:
                interpolated[i] = (prev_v + next_v) / 2

    return interpolated
