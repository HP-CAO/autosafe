import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
import argparse

# Baselines
from job.base.sac_cart_pole import AllConfig as SACCartPole
from job.base.sac_cheetah import AllConfig as SACCheetah
from job.base.sac_hopper import AllConfig as SACHopper
from job.base.sac_walker import AllConfig as SACWalker
from job.base.sac_ant import AllConfig as SACAnt

# training with context-randomization
from job.DR.DR_hopper import AllConfig as DRHopper
from job.DR.DR_halfcheetah import AllConfig as DRHalfCheetah
from job.DR.DR_walker import AllConfig as DRWalker
from job.DR.DR_ant import AllConfig as DRAnt

# CartPole
from job.context.CartPole.vanilla_DR import AllConfig as CartPoleVanillaDR
from job.context.CartPole.cRL_IB import AllConfig as CartPoleIB
from job.context.IB.ib_learning import AllConfig as CartPoleTest

# training with context-randomization
# from job.context.DR.sac_quad_aug_obs import AllConfig as DR_SACQuadrotor
# from job.context.DR.sac_half_cheetah_aug_obs import AllConfig as DR_SACHalfCheetah
# from job.context.DR.sac_cart_pole_aug_obs import AllConfig as DR_SACCartPole

# training with context-MSE-model
# from job.context.DR_MSE.sac_mse_cart_pole import AllConfig as DR_MSE_SACCartPole
# from job.context.DR_MSE.sac_mse_halfcheetah import AllConfig as DR_MSE_SACtHalfCheetah
# from job.context.DR_MSE.sac_mse_quadrotor import AllConfig as DR_MSE_SACQuadrotor

# training with context-IB-model
# from job.context.DR_IB.sac_ib_cart_pole_decor import AllConfig as DR_IB_SACCartPole
# from job.context.DR_IB.sac_ib_quadrotor_decor import AllConfig as DR_IB_SACQuadrotor
# from job.context.DR_IB.sac_ib_halfcheetah_decor import AllConfig as DR_IB_SACHalfCheetah

ALL_JOBS = {
    # baselines
    "SACCartPole": SACCartPole,
    "SACCheetah": SACCheetah,
    "SACHopper": SACHopper,
    "SACWalker": SACWalker,
    "SACAnt": SACAnt,

    # training with context-randomization
    "DRHopper": DRHopper,
    "DRHalfCheetah": DRHalfCheetah,
    "DRWalker": DRWalker,
    "DRAnt": DRAnt,

    # CartPole
    "CartPoleVanillaDR": CartPoleVanillaDR,
    "CartPoleIB": CartPoleIB,
    "CartPoleTest": CartPoleTest,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='Listing all available jobs')
    parser.add_argument('--generate', default='', help='Generate a config file for a specific job')
    parser.add_argument('--all', action='store_true', help='Generate all config files')
    args = parser.parse_args()

    if args.list:
        print("Available jobs:")
        for job_name in ALL_JOBS.keys():
            print(f"\t{job_name}")
        exit(0)

    if args.generate != '':
        config_directory = "./config"
        if not os.path.exists(config_directory):
            os.makedirs(config_directory)

        if args.generate not in ALL_JOBS:
            print(f"Job {args.generate} not found.")
            exit(1)
        OmegaConf.save(ALL_JOBS[args.generate], config_directory + f'/{args.generate}.yaml')
        print(f"Config file generated at {config_directory}/{args.generate}.yaml")
        exit(0)

    if args.all:
        config_directory = "./config"
        if not os.path.exists(config_directory):
            os.makedirs(config_directory)
        for job_name in ALL_JOBS.keys():
            OmegaConf.save(ALL_JOBS[job_name], config_directory + f'/{job_name}.yaml')
            print(f"Config file generated at {config_directory}/{job_name}.yaml")
        exit(0)

    print("Please specify the job to generate configurations --list to list all available jobs.")