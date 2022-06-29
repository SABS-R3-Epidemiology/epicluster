"""Mathematical and statistical functions.
"""

import math
import scipy.special


def log_poch(z, m):
    """Return the logarithm of the rising factorial.

    The rising factorial is also known as the Pochhammer function.

    It is defined by z^(m) = z (z+1) (z+2) ... (z+m-1).

    If m=0, the result is zero.

    Parameters
    ----------
    z : float
    m : float

    Returns
    -------
    float
        Logarithm of rising factorial
    """
    if z == 0:
        return 0
    return math.lgamma(z+m) - math.lgamma(z)


def poch_negatives(z, m):
    if z == 0:
        return 0
    p = z
    i = 1
    while i < m:
        p *= (z + i)
        i += 1
    return p
    return math.log(p)

# def log_poch_arrays(z, m):
#     """Same as log_poch, but for array inputs.
#     """
#     return scipy.special.loggamma(z+m) - scipy.special.loggamma(z)
