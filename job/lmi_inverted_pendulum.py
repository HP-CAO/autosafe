import numpy as np
import cvxpy as cp

A = np.array([[1, 0.03333333, 0, 0],
                [0, 1, -0.05649123, 0],
                [0, 0, 1, 0.03333333],
                [0, 0, 0.89802632, 1]])

B = np.array([[0],
              [0.03341688],
              [0],
              [-0.0783208]])


D = np.array([[1/0.4,   0,        0,        0],
                [0,     1/4.5,       0,     0],
                [0,     0,    1/0.4,         0],
                [0,     0,       0,     1/4.5]])


n = 4
alpha = 0.96

Q = cp.Variable((n,n), symmetric=True)
R = cp.Variable((1,n))

constraints = [cp.bmat([[alpha * Q, Q @ A.T + R.T @ B.transpose()],
                        [A @ Q + B @ R, Q]]) >> 0,
               np.eye(4) - D @ Q @ D.T >> 0]

problem = cp.Problem(cp.Minimize(-1 * cp.log_det(Q)), constraints)
problem.solve(solver=cp.SCS)

P = np.linalg.inv(Q.value)
F = R @ P
A + B @ F

K = np.array(R.value) @ np.linalg.pinv(Q.value)
M = A + B @ K

print(np.linalg.eigvals(M))
# Check stability
assert np.all(np.linalg.eigvals(M) < 0)

print("Optimal Q:", Q.value)
print("Controller:", F)