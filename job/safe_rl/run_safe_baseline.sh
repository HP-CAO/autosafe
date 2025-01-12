python job/safe_rl/ddpg_cart_pole_base.py JobParams.job_name=ddpg_cart_pole_safe_base_1 &
sleep 1
python job/safe_rl/ddpg_cart_pole_safe.py JobParams.job_name=ddpg_cart_pole_safe_simplex_1 &
sleep 1
python job/safe_rl/ddpg_cart_pole_base.py JobParams.job_name=ddpg_cart_pole_safe_base_2 &
sleep 1
python job/safe_rl/ddpg_cart_pole_safe.py JobParams.job_name=ddpg_cart_pole_safe_simplex_2

