python job/experiments/quadrotor.py JobParams.job_name=base_quadrotor_fix JobParams.experiment_name=base &
sleep 5
python job/experiments/quadrotor.py JobParams.job_name=mt_quadrotor JobParams.experiment_name=multi_task &
sleep 5
python job/experiments/quadrotor.py JobParams.job_name=mt_quadrotor_reg JobParams.experiment_name=multi_task_reg &
sleep 5
python job/experiments/quadrotor.py JobParams.job_name=model_based_quadrotor JobParams.experiment_name=model_based &
sleep 5
python job/experiments/quadrotor.py JobParams.job_name=mt_quadrotor_reg_apt JobParams.experiment_name=multi_task_reg_apt