import nashpy as nash
import numpy as np
from gamegenerator import getCorrelatedEquilibria, getMixedNashEquilibria, getPureNashEquilibria 

def NashEquilibriaSolver(A,B=None):
    """
    Given payoff matrix/matrices, return a list of existing Nash equilibria:
    [(nash1_p1, nash1_p2), (nash2_p1, nash2_p2), ...]
    """
    if B is not None:
        rps = nash.Game(A, B)
    else:
        rps = nash.Game(A)  # zero-sum game: unimatrix  
    eqs = rps.support_enumeration()
    return list(eqs)

def NashEquilibriumSolver(A, B=None):
    """
    Quickly solve *one* Nash equilibrium with Lemke Howson algorithm.
    Ref: https://nashpy.readthedocs.io/en/stable/reference/lemke-howson.html#lemke-howson

    TODO: sometimes give nan or wrong dimensions
    """
    if B is not None:
        rps = nash.Game(A, B)
    else:
        rps = nash.Game(A)  # zero-sum game: unimatrix  
    eq = rps.lemke_howson(initial_dropped_label=0) # The initial_dropped_label is an integer between 0 and sum(A.shape) - 1
    return eq

if __name__ == "__main__":
    A = np.array([[0, -1, 1], [2, 0, -1], [-1, 1, 0]])
#     A=np.array([[ 0.594,  0.554,  0.552,  0.555,  0.567,  0.591],
#  [ 0.575,  0.579,  0.564,  0.568,  0.574,  0.619],
#  [-0.036,  0.28,   0.53,   0.571,  0.57,  -0.292],
#  [ 0.079, -0.141, -0.2,    0.592,  0.525, -0.575],
#  [ 0.545,  0.583,  0.585,  0.562,  0.537,  0.606],
#  [ 0.548,  0.576,  0.58,   0.574,  0.563,  0.564]])
    nes = NashEquilibriaSolver(A)
    print(nes)

    # ne = NashEquilibriumSolver(A)
    # print(ne)

    # cce = getCorrelatedEquilibria(A, coarse=False)
    # print(cce)

    # ne = getMixedNashEquilibria(A)
    # print(ne)

