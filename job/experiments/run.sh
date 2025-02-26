#python job/experiments/cartpole.py JobParams.job_name=base_cartpole_fix JobParams.experiment_name=base &
#sleep 5
python job/experiments/cartpole.py JobParams.job_name=mt_cartpole JobParams.experiment_name=multi_task &
sleep 5
python job/experiments/cartpole.py JobParams.job_name=mt_cartpole_reg JobParams.experiment_name=multi_task_reg &
sleep 5
python job/experiments/cartpole.py JobParams.job_name=model_based_cartpole JobParams.experiment_name=model_based &
sleep 5
python job/experiments/cartpole.py JobParams.job_name=mt_cartpole_reg_apt JobParams.experiment_name=multi_task_reg_apt