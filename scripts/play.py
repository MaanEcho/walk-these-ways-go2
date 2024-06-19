import isaacgym
assert isaacgym

import torch
import numpy as np

import glob
import pickle as pkl

from go2_gym.envs import *
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    # 加载 body 模型
    """
    这段代码的主要功能是从指定的路径加载一个预先编译好的 TorchScript 模型，并将其赋值给变量 body。这个模型可以用于后续的推理或进一步的训练。通过使用 TorchScript，模型可以在不依赖 Python 解释器的情况下运行，从而提高性能和可移植性。
    """
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')
    # 加载 adaptation_module 模型
    """
    这段代码的主要功能是从指定的路径加载一个预先编译好的 TorchScript 模型，并将其赋值给变量 adaptation_module。这个模型可以用于后续的推理或进一步的训练。通过使用 TorchScript，模型可以在不依赖 Python 解释器的情况下运行，从而提高性能和可移植性。
    """

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        # 调用 adaptation_module 模型，将 obs_history 输入模型，得到 latent 值
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        # 调用 body 模型，将 obs_history 和 latent 输入模型，得到 action 值（body其实就是policy吗？）
        info['latent'] = latent
        # 将 latent 值存入 info 字典中
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    # 根据给定的 label 值，获取上级目录中 runs 目录下所有匹配的文件和子目录的路径列表。
    # 具体来说，它会生成一个路径模式，并使用 glob.glob 函数来查找所有符合该模式的文件和子目录。
    logdir = sorted(dirs)[0]
    # 从 dirs 列表中获取排序后的第一个路径，并将其赋值给 logdir 变量。

    with open(logdir + "/parameters.pkl", 'rb') as file:
        # 在机器学习领域，.pkl 文件常用于保存训练好的模型，以便后续可以加载并进行预测
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    # DR = Domain Randomization
    Cfg.domain_rand.push_robots = False
    # 推机器人
    Cfg.domain_rand.randomize_friction = False
    # 摩擦力
    Cfg.domain_rand.randomize_gravity = False
    # 重力
    Cfg.domain_rand.randomize_restitution = False
    # 弹性系数
    Cfg.domain_rand.randomize_motor_offset = False
    # 电机偏移？
    Cfg.domain_rand.randomize_motor_strength = False
    # 电机强度？
    Cfg.domain_rand.randomize_friction_indep = False
    # 独立摩擦力？
    Cfg.domain_rand.randomize_ground_friction = False
    # 地面摩擦力
    Cfg.domain_rand.randomize_base_mass = False
    # 质量（什么是base mass?）
    Cfg.domain_rand.randomize_Kd_factor = False
    # K_d
    Cfg.domain_rand.randomize_Kp_factor = False
    # K_p
    Cfg.domain_rand.randomize_joint_friction = False
    # 关节摩擦力
    Cfg.domain_rand.randomize_com_displacement = False
    # COM位移？

    Cfg.env.num_recording_envs = 1
    # 记录的环境数目?
    Cfg.env.num_envs = 5
    # 环境数目
    Cfg.terrain.num_rows = 5
    # 地形的行数
    Cfg.terrain.num_cols = 5
    # 地形的列数
    Cfg.terrain.border_size = 0
    # 地形边界的大小（0表示没有边界）
    Cfg.terrain.center_robots = True
    # 是否将机器人放在地图的中心?
    Cfg.terrain.center_span = 1
    # 机器人中心的范围（1表示机器人位于地图的中心）
    Cfg.terrain.teleport_robots = True
    # 是否随机传送(teleport)机器人到地图的边缘

    Cfg.domain_rand.lag_timesteps = 6
    # 延迟时间步数？，默认为6，可以设置为0来禁用滞后补偿（什么是滞后补偿？）
    Cfg.domain_rand.randomize_lag_timesteps = True
    # 是否随机延迟时间步数
    Cfg.control.control_type = "actuator_net"
    # Cfg.control.control_type = "P"
    # default control_type is "actuator_net", you can also switch it to "P" to enable joint PD control
    # 控制类型，默认为"actuator_net"，可以设置为"P"来启用关节PD控制
    Cfg.asset.flip_visual_attachments = True
    # 是否翻转可视化附件？


    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    # VelocityTrackingEasyEnv：这是一个自定义的环境类，用于模拟机器人跟踪速度的任务
    """
    VelocityTrackingEasyEnv类是基于Isaac Gym库构建的，用于模拟机器人跟踪特定速度的任务。通过设置sim_device='cuda:0'，可以利用GPU加速模拟过程，提高计算效率。headless=False确保在模拟过程中可以看到图形界面，这对于调试和可视化非常有用。cfg=Cfg传递了环境的配置参数，这些参数定义了环境的各项属性，如机器人动力学、地形特征等。
    """
    env = HistoryWrapper(env)
    # HistoryWrapper：这是一个包装器类，用于记录环境的历史状态。通过记录历史状态，可以为策略提供更多的上下文信息，从而提高策略的性能。
    """
    VelocityTrackingEasyEnv类是基于Isaac Gym库构建的，用于模拟机器人跟踪特定速度的任务。通过设置sim_device='cuda:0'，可以利用GPU加速模拟过程，提高计算效率。headless=False确保在模拟过程中可以看到图形界面，这对于调试和可视化非常有用。cfg=Cfg传递了环境的配置参数，这些参数定义了环境的各项属性，如机器人动力学、地形特征等。
    """

    # load policy
    from ml_logger import logger
    from go2_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go2(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go2_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/pretrain-go2/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 2500 #250
    # 评估步数的数量
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}
    # 定义了四种步态的动作命令
    # pronking:弹跳步态。四条腿同时离地并同时着地，类似于弹跳。
    # trotting:对角步态。在这种步态中，动物的两条对角线上的腿同时移动，即左前腿和右后腿同时移动，接着是右前腿和左后腿。这种步态通常用于中等速度的移动。
    # bounding:跳跃步态。在这种步态中，动物的两条前腿和两条后腿分别同时离地并同时着地。这种步态通常用于快速移动，尤其是在开阔地带。
    # pacing:侧步态。在这种步态中，动物的两条同侧的前后腿同时移动。例如，左前腿和左后腿同时移动，接着是右前腿和右后腿。这种步态通常用于稳定性和舒适性较高的移动。
    # 每种步态的三个参数分别对应了左前腿、右前腿、右后腿的动作命令（GPT生成）？为什么参数的值只能是0或0.5？

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    # 初始速度命令
    body_height_cmd = 0.0
    # 身体高度命令
    step_frequency_cmd = 3.0 #3.0
    # 步频命令
    # gait = torch.tensor(gaits["pronking"])
    # gait = torch.tensor(gaits["trotting"])
    # gait = torch.tensor(gaits["bounding"])
    gait = torch.tensor(gaits["pacing"])
    footswing_height_cmd = 0.1
    # 足摆动高度命令
    pitch_cmd = 0.0
    # 俯仰角命令
    roll_cmd = 0.0
    # 翻滚角命令
    stance_width_cmd = 0.25
    # 立定宽度命令

    measured_x_vels = np.zeros(num_eval_steps)
    # 记录测量的x速度
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    # 记录目标的x速度
    joint_positions = np.zeros((num_eval_steps, 12))
    # 记录关节位置
    ###### -----------ldt---------------
    joint_torques = np.zeros((num_eval_steps, 12))
    # 记录关节力矩

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        ###### -----------ldt---------------
        # joint_torques[i] = env.torques.detach().cpu().numpy()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_torques, linestyle="-", label="Measured")
    axs[2].set_title("Joint Torques")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Torques (Nm)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go2(headless=False)
