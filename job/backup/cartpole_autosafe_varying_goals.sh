python job/cartpole.py JobParams.job_name=autosafe_baseline JobParams.experiment_name=autosafe &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_1 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.05 &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_2 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.1 &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_3 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.15 &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_4 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.2 &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_5 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.25 &
sleep 3
python job/cartpole.py JobParams.job_name=autosafe_goal_6 JobParams.experiment_name=autosafe GymParams.TaskParams.control_goal_x=0.3