# postprocessing/filtering.py

def filter_outliers(values, max_jump=0.15):
    """
    Remove sudden unrealistic jumps in normalized values.
    Keeps temporal consistency.
    """
    if not values:
        return values

    filtered = [values[0]]

    for v in values[1:]:
        prev = filtered[-1]

        if v is None or prev is None:
            filtered.append(v)
        elif abs(v - prev) > max_jump:
            filtered.append(prev)
        else:
            filtered.append(v)

    return filtered
