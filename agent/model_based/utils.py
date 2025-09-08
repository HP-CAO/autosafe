def get_lyapunov_cost(tracking_error_curr, lyapunov_matrix):
    cur_lyapunov_value = tracking_error_curr @ lyapunov_matrix @ tracking_error_curr.T
    return cur_lyapunov_value # encouraging decrease