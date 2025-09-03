import dataclasses
from envs.quad_gym.task.goal_task import GoalTaskParams
from envs.quad_gym.task import move_forward_task, goal_task
from envs.quad_gym.task.move_forward_task import MoveForwardTaskParams

AllTasks = {
    "move_forward_task": move_forward_task.MoveForwardTask,
    "goal_task": goal_task.GoalTask,
}


@dataclasses.dataclass
class TaskConfig:
    task_type: str = "move_forward_task"
    task_params: object = None
    sub_goal: bool = False
    include_historic_sensors: bool = False
    num_history: int = 3
    domain_randomization: bool = False
    diagonal_act: bool = False
    random_dir: bool = False
    dir_update_interval: int = None
    curriculum: bool = False
    max_episode_steps: int = 1000
    random_seed: int = 1

    def __post_init__(self):
        if self.task_type == "goal_task":
            self.task_params = GoalTaskParams()
        else:
            self.task_params = MoveForwardTaskParams()
