import copy
import time
import os

import numpy as np
import torch

from go2_gym_deploy.utils.logger import MultiLogger


class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):   # 阅读完成
        self.agents = {}
        self.policy = None
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.triggered_commands = {i: None for i in range(4)} # command profiles for each action button on the controller
        # 遥控器上四个按键X、Y、B、A（顺序不一定对）的命令配置文件
        self.button_states = np.zeros(4)
        # 按键状态

        self.is_currently_probing = False
        # 是否正在探测？（目前不知道是什么意思）
        self.is_currently_logging = [False, False, False, False]
        # 是否正在记录？（目前不知道是什么意思）

    def init_log_filename(self):    # 阅读完成
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")
        # 生成当前日期和时间的字符串格式

        for i in range(100):
        # 尝试创建一个带有时间戳和索引的日志文件夹（最多尝试100次）
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                # 设置日志文件名，并存储在self.log_filename中
                return
            except FileExistsError:
                continue

    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):    # 阅读完成
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):    # 阅读完成
        self.policy = policy

    def add_command_profile(self, command_profile):  # 阅读完成
        self.command_profile = command_profile

    def calibrate(self, wait=True, low=False):  # 
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        # 首先，如果机器人不在标准姿态（nominal pose），则缓慢地将其移动到标准姿态。
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,])
                else:
                    final_goal = np.zeros(12)
                nominal_joint_pos = agent.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    # 如果R2键被按下，则开始校准。
                        print(">>>>>>>>>>>>>>> R2 is pressed <<<<<<<<<<<<<")
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = joint_pos - nominal_joint_pos
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                # 逐步调整target使其接近final_goal，直到两者的最大差异小于0.01。
                for target in target_sequence:
                    next_target = target
                    if isinstance(agent.cfg, dict):
                        hip_reduction = agent.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.cfg["control"]["action_scale"]
                    else:
                        hip_reduction = agent.cfg.control.hip_scale_reduction
                        action_scale = agent.cfg.control.action_scale

                    next_target[[0, 3, 6, 9]] /= hip_reduction
                    next_target = next_target / action_scale
                    cal_action[:, 0:12] = next_target
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        print(">>>>>>>>>>>>>>> R2 is pressed again <<<<<<<<<<<<<")
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs

    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        control_obs = self.calibrate(wait=True)

        # now, run control loop

        try:
            for i in range(max_steps):

                policy_info = {}
                action = self.policy(control_obs, policy_info)
                # 输入观测和潜在变量，输出动作

                for agent_name in self.agents.keys():
                    obs, ret, done, info = self.agents[agent_name].step(action)

                    info.update(policy_info)
                    info.update({"observation": obs, "reward": ret, "done": done, "timestep": i,
                                 "time": i * self.agents[self.control_agent_name].dt, "action": action, "rpy": self.agents[self.control_agent_name].se.get_rpy(), "torques": self.agents[self.control_agent_name].torques})

                    if logging: self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        control_obs, control_ret, control_done, control_info = obs, ret, done, info

                # bad orientation emergency stop
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                # check for logging command
                prev_button_states = self.button_states[:]
                # 记录上一次的按键状态
                self.button_states = self.command_profile.get_buttons()
                # 读取当前的按键状态

                if self.command_profile.state_estimator.left_lower_left_switch_pressed:
                # 如果L2键被按下，则开始记录。
                    if not self.is_currently_probing:
                        print("START LOGGING")
                        self.is_currently_probing = True
                        self.agents[self.control_agent_name].set_probing(True)
                        self.init_log_filename()
                        self.logger.reset()
                    else:
                        print("SAVE LOG")
                        self.is_currently_probing = False
                        self.agents[self.control_agent_name].set_probing(False)
                        # calibrate, log, and then resume control
                        control_obs = self.calibrate(wait=False)
                        self.logger.save(self.log_filename)
                        self.init_log_filename()
                        self.logger.reset()
                        time.sleep(1)
                        control_obs = self.agents[self.control_agent_name].reset()
                    self.command_profile.state_estimator.left_lower_left_switch_pressed = False
                    # 重置L2键状态

                for button in range(4):
                    if self.command_profile.currently_triggered[button]:
                        if not self.is_currently_logging[button]:
                            print("START LOGGING")
                            self.is_currently_logging[button] = True
                            self.init_log_filename()
                            self.logger.reset()
                    else:
                        if self.is_currently_logging[button]:
                            print("SAVE LOG")
                            self.is_currently_logging[button] = False
                            # calibrate, log, and then resume control
                            control_obs = self.calibrate(wait=False)
                            self.logger.save(self.log_filename)
                            self.init_log_filename()
                            self.logger.reset()
                            time.sleep(1)
                            control_obs = self.agents[self.control_agent_name].reset()

                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                # 如果R2键被按下，则开始校准。
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    # 重置R2键状态
                    # self.button_states = self.command_profile.get_buttons()
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                        # self.button_states = self.command_profile.get_buttons()
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)
            self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            self.logger.save(self.log_filename)
