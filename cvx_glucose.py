import cvxpy as cp
import numpy as np

Ad = np.array([[1.00000000e+00, -1.28649765e+03, -4.63416638e-02],
               [0.00000000e+00,  8.67621256e-01,  4.53400039e-05],
               [0.00000000e+00,  0.00000000e+00,  5.99646534e-02]])
Bd = np.array([[-1.86359159e-01],
               [ 3.35854493e-04],
               [ 3.34162632e+00]])

n, m = Ad.shape[0], Bd.shape[1]
X = cp.Variable((n,n), symmetric=True)
Y = cp.Variable((m,n))
rho = 1.0        # set to e.g. 0.98 for contraction margin
eps = 1e-8

AX_BY = Ad @ X + Bd @ Y
if rho < 1.0:
    LMI = cp.bmat([[rho*X,        AX_BY],
                   [AX_BY.T,      rho*X]])
else:
    LMI = cp.bmat([[X,            AX_BY],
                   [AX_BY.T,      X]])

# (optional) soft bound on gain size
t = cp.Variable(nonneg=True)
gain_bound = cp.bmat([[t*np.eye(m), Y],
                      [Y.T,         X]])

prob = cp.Problem(
    cp.Minimize(cp.trace(X) + 1e-3*t),
    constraints=[
        X >> eps*np.eye(n),
        LMI >> eps*np.eye(2*n),
        gain_bound >> 0
    ]
)
prob.solve(solver=cp.MOSEK, verbose=False)  # or SCS if MOSK isn’t available

K = Y.value @ np.linalg.inv(X.value)
print("K =", K)
print("spectral radius of Acl =", max(abs(np.linalg.eigvals(Ad + Bd @ K))))
