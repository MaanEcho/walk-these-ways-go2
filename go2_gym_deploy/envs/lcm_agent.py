import time

import lcm
import numpy as np
import torch
# import cv2

from go2_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def class_to_dict(obj) -> dict:  # 阅读完成 √
    """
    功能：将一个Python类实例（obj）转换为一个字典（dict）。具体来说，它递归地将类的属性及其嵌套对象转换为字典形式，并忽略以下两个内容：
    ①私有属性：以 _ 开头的属性会被忽略；②特定属性：名为 terrain 的属性会被忽略。
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

class LCMAgent():
    def __init__(self, cfg, se, command_profile):   # 阅读完成 但还没完全理解函数内容
        if not isinstance(cfg, dict):
        # 判断变量cfg是否是一个字典（dict）。如果不是字典，则调用class_to_dict()函数将cfg转换为一个字典。
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        # self.se 等价于 StateEstimator(lc)
        self.command_profile = command_profile
        # self.command_profile 等价于 RCControllerProfile(dt=0.02, state_estimator=se, x_scale=2.5, y_scale=0.6, yaw_scale=5.0)

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        # dt还是控制步长的含义吗？
        self.timestep = 0
        # timestep是指当前的步数吗？

        self.num_obs = self.cfg["env"]["num_observations"]
        # 观测observations的数量
        self.num_envs = 1
        # 环境的数量
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        # 特权观测privileged_observations的数量
        self.num_actions = self.cfg["env"]["num_actions"]
        # 动作actions的数量
        self.num_commands = self.cfg["commands"]["num_commands"]
        # 命令commands的数量
        self.device = 'cpu'
        # 设备device的类型

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.commands_scale = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"], self.obs_scales["body_height_cmd"], 1, 1, 1, 1, 1,
             self.obs_scales["footswing_height_cmd"], self.obs_scales["body_pitch_cmd"],
             # 0, self.obs_scales["body_pitch_cmd"],
             self.obs_scales["body_roll_cmd"], self.obs_scales["stance_width_cmd"],
             self.obs_scales["stance_length_cmd"], self.obs_scales["aux_reward_cmd"], 1, 1, 1, 1, 1, 1
             ])[:self.num_commands]

        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # 左前机身、大腿、小腿关节
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # 右前机身、大腿、小腿关节
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",  # 左后机身、大腿、小腿关节 
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]    # 右后机身、大腿、小腿关节
        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in joint_names])
        try:
            self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"]])
        except KeyError:
            self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        # 12个关节电机的kp？
        self.d_gains = np.zeros(12)
        # 12个关节电机的kd？
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(12)
        # 动作
        self.last_actions = torch.zeros(12)
        # 上一次的动作
        self.gravity_vector = np.zeros(3)
        # 重力向量
        self.dof_pos = np.zeros(12)
        # 关节位置
        self.dof_vel = np.zeros(12)
        # 关节速度
        self.body_linear_vel = np.zeros(3)
        # 机器人线速度
        self.body_angular_vel = np.zeros(3)
        # 机器人角速度
        self.joint_pos_target = np.zeros(12)
        # 关节目标位置
        self.joint_vel_target = np.zeros(12)
        # 关节目标速度
        self.torques = np.zeros(12)
        # 扭矩
        self.contact_state = np.ones(4)
        # 足端接触状态

        self.joint_idxs = self.se.joint_idxs
        # 关节电机索引

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        # 步态索引
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)
        # 时钟输入？（目前不知道什么意思）

        # if "obs_scales" in self.cfg.keys():
        #     self.obs_scales = self.cfg["obs_scales"]
        # else:
        #     self.obs_scales = self.cfg["normalization"]["obs_scales"]
        # 这段代码跟上面的重复了。

        self.is_currently_probing = False
        # 是否正在探测？（目前不知道什么意思）

    def set_probing(self, is_currently_probing):    # 阅读完成，但还是不知道probe在这里的含义是什么
        self.is_currently_probing = is_currently_probing

    def get_obs(self):  # 阅读完成

        self.gravity_vector = self.se.get_gravity_vector()  # √
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)  # √
        self.commands[:, :] = cmds[:self.num_commands]  # √
        if reset_timer:  # √
            self.reset_gait_indices()
        #else:
        #    self.commands[:, 0:3] = self.command_profile.get_command(self.timestep * self.dt)[0:3]
        self.dof_pos = self.se.get_dof_pos()    # √
        self.dof_vel = self.se.get_dof_vel()    # √
        self.body_linear_vel = self.se.get_body_linear_vel()    # √
        self.body_angular_vel = self.se.get_body_angular_vel()  # √

        ob = np.concatenate((self.gravity_vector.reshape(1, -1),
                             self.commands * self.commands_scale,
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.obs_scales["dof_pos"],
                             self.dof_vel.reshape(1, -1) * self.obs_scales["dof_vel"],
                             torch.clip(self.actions, -self.cfg["normalization"]["clip_actions"],
                                        self.cfg["normalization"]["clip_actions"]).cpu().detach().numpy().reshape(1, -1)
                             ), axis=1)  # √

        if self.cfg["env"]["observe_two_prev_actions"]:  # √
            ob = np.concatenate((ob,
                            self.last_actions.cpu().detach().numpy().reshape(1, -1)), axis=1)

        if self.cfg["env"]["observe_clock_inputs"]:  # √
            ob = np.concatenate((ob,
                            self.clock_inputs), axis=1)
            # print(self.clock_inputs)

        if self.cfg["env"]["observe_vel"]:  # √
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 self.body_angular_vel.reshape(1, -1) * self.obs_scales["ang_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_only_lin_vel"]:  # √
            ob = np.concatenate(
                (self.body_linear_vel.reshape(1, -1) * self.obs_scales["lin_vel"],
                 ob), axis=1)

        if self.cfg["env"]["observe_yaw"]:  # √
            heading = self.se.get_yaw()
            ob = np.concatenate((ob, heading.reshape(1, -1)), axis=-1)

        self.contact_state = self.se.get_contact_state()    # √
        if "observe_contact_states" in self.cfg["env"].keys() and self.cfg["env"]["observe_contact_states"]:    # √
            ob = np.concatenate((ob, self.contact_state.reshape(1, -1)), axis=-1)

        if "terrain" in self.cfg.keys() and self.cfg["terrain"]["measure_heights"]:  # √
            robot_height = 0.25
            self.measured_heights = np.zeros(
                (len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"]))).reshape(
                1, -1)
            heights = np.clip(robot_height - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales["height_measurements"]
            ob = np.concatenate((ob, heights), axis=1)

        return torch.tensor(ob, device=self.device).float()  # √

    def get_privileged_observations(self):  # 阅读完成
        """部署时无需特权观测"""
        return None

    # "pd_plustau_targets" channel
    def publish_action(self, action, hard_reset=False): # 阅读完成
        """将关节电机指令发送到‘pd_plustau_targets’ channel中"""

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        # self.joint_pos_target[[0, 3, 6, 9]] *= -1
        self.joint_pos_target = self.joint_pos_target   # ？等号左右两边相同，这是为什么？
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)
        # print(f'cjp {self.joint_pos_target}')

        command_for_robot.q_des = joint_pos_target
        # 关节目标位置
        command_for_robot.qd_des = self.joint_vel_target
        # 关节目标速度
        command_for_robot.kp = self.p_gains
        # 关节电机的kp
        command_for_robot.kd = self.d_gains
        # 关节电机的kd
        command_for_robot.tau_ff = np.zeros(12)
        # 关节电机的前馈扭矩
        command_for_robot.se_contactState = np.zeros(4)
        # 足端接触状态
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        # 时间戳
        command_for_robot.id = 0
        # ID（不知道是什么的ID）

        if hard_reset:
            command_for_robot.id = -1
        # 不知道这里hard_reset是什么意思


        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        # 由lcm将神经网络输出的action传入c++ sdk
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):    # 阅读完成
        self.actions = torch.zeros(12)
        # 初始化动作actions为0
        self.time = time.time()
        # 时间戳self.time
        self.timestep = 0
        # 初始化时间步timestep为0
        return self.get_obs()

    def reset_gait_indices(self):   # 阅读完成
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        if self.num_commands == 8:
            bounds = 0
            durations = self.commands[:, 7]
        else:
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[3])

# 注释掉了下面camera相关代码
# --------------------------------------------------------------------
        # images = {'front': self.se.get_camera_front(),
        #           'bottom': self.se.get_camera_bottom(),
        #           'rear': self.se.get_camera_rear(),
        #           'left': self.se.get_camera_left(),
        #           'right': self.se.get_camera_right()
        #           }
        # downscale_factor = 2
        # temporal_downscale = 3

        # for k, v in images.items():
        #     if images[k] is not None:
        #         images[k] = cv2.resize(images[k], dsize=(images[k].shape[0]//downscale_factor, images[k].shape[1]//downscale_factor), interpolation=cv2.INTER_CUBIC)
        #     if self.timestep % temporal_downscale != 0:
        #         images[k] = None
        #print(self.commands)

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                #  -------------------------------------------
                #  "camera_image_front": images['front'],
                #  "camera_image_bottom": images['bottom'],
                #  "camera_image_rear": images['rear'],
                #  "camera_image_left": images['left'],
                #  "camera_image_right": images['right'],
                 }

        self.timestep += 1
        return obs, None, None, infos
