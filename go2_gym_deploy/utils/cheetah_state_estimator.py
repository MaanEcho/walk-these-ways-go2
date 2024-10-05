import math
import select
import threading
import time

import numpy as np

from go2_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go2_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go2_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
# 不调用相机 !!!
# from go1_gym_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
# from go1_gym_deploy.lcm_types.camera_message_rect_wide import camera_message_rect_wide


def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class StateEstimator:
    def __init__(self, lc, use_cameras=False): # default use_cameras=True 阅读完成
        
        # reverse legs
        # 这里腿的顺序为什么要转换？
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # 关节电机索引
        self.contact_idxs = [1, 0, 3, 2]
        # 四个足端索引
        # self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.lc = lc
        # 跟LCM相关

        self.joint_pos = np.zeros(12)
        # 关节位置
        self.joint_vel = np.zeros(12)
        # 关节速度
        self.tau_est = np.zeros(12)
        # 估计的扭矩
        self.world_lin_vel = np.zeros(3)
        # 世界坐标系下的线速度
        self.world_ang_vel = np.zeros(3)
        # 世界坐标系下的角速度
        self.euler = np.zeros(3)
        # 欧拉角
        self.R = np.eye(3)
        # 旋转矩阵
        self.buf_idx = 0
        # 缓冲区索引（目前不知道用途）

        self.smoothing_length = 12
        # 表示用作角速度平滑处理的历史数据长度
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        # 包含历史欧拉角变化的数组，表示某个时间点的欧拉角变化，用于计算角速度的平滑值
        self.dt_history = np.zeros((self.smoothing_length, 1))
        # 包含self.deuler_history对应时间间隔的数组，每个元素表示self.deuler_history中相应欧拉角变化所经历的时间间隔，用于计算角速度的平滑值
        self.euler_prev = np.zeros(3)
        # 上一次的欧拉角？
        self.timuprev = time.time()
        # 生成时间戳timuprev（目前不知道用途）

        self.body_lin_vel = np.zeros(3)
        # 机身坐标系下的线速度
        self.body_ang_vel = np.zeros(3)
        # 机身坐标系下的角速度
        self.smoothing_ratio = 0.2
        # 平滑系数，用于计算角速度的平滑值

        self.contact_state = np.ones(4)
        # 足端接触状态

        self.mode = 0
        # 某种模式（目前不知道用途）
        self.ctrlmode_left = 0
        # 左摇杆控制模式Mode A-C
        self.ctrlmode_right = 0
        # 右摇杆控制模式Mode D-F
        self.left_stick = [0, 0]
        # 左摇杆。两个值分别代表左摇杆x方向上的值和y方向上的值，范围是[-1,1]。
        self.right_stick = [0, 0]
        # 右摇杆。两个值分别代表右摇杆x方向上的值和y方向上的值，范围是[-1,1]。
        self.left_upper_switch = 0
        # L1键是否松开
        self.left_lower_left_switch = 0
        # L2键是否松开
        self.left_lower_right_switch = 0
        # MIT Cheetah项目用的不是宇树的遥控器，宇树的没有这个按键。

        self.right_upper_switch = 0
        # R1键是否松开
        self.right_lower_left_switch = 0
        # MIT Cheetah项目用的不是宇树的遥控器，宇树的没有这个按键。
        self.right_lower_right_switch = 0
        # R2键是否松开

        self.left_upper_switch_pressed = 0
        # L1键是否按下
        self.left_lower_left_switch_pressed = 0
        # L2键是否按下
        self.left_lower_right_switch_pressed = 0
        # MIT Cheetah项目用的不是宇树的遥控器，宇树的没有这个按键。

        self.right_upper_switch_pressed = 0
        # R1键是否按下
        self.right_lower_left_switch_pressed = 0
        # MIT Cheetah项目用的不是宇树的遥控器，宇树的没有这个按键。
        self.right_lower_right_switch_pressed = 0
        # R2键是否按下

        # default trotting gait
        self.cmd_freq = 3.0
        # 控制频率？
        self.cmd_phase = 0.5
        # 控制相位？（我一直没有理解相位phase的含义）
        self.cmd_offset = 0.0
        # 控制偏移？（应该是跟相位phase有关的）
        self.cmd_duration = 0.5
        # 控制持续时间？（应该也是跟相位phase有关的）


        self.init_time = time.time()
        # 生成时间戳init_time（目前不知道用途）
        self.received_first_legdata = False
        # 是否是第一次接收到来自lcm_position_go2的机器人状态信息

        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        # 订阅机器人的状态信息
        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        # 订阅机器人姿态信息
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        # 订阅遥控器信号
        # 上面的三句代码和lcm_position_go2中Custom::lcm_send()的最后三句代码相对应
        
        # -----------------------------------------------------------------------------------------------------
        # if use_cameras:
        #     for cam_id in [1, 2, 3, 4, 5]:
        #         self.camera_subscription = self.lc.subscribe(f"camera{cam_id}", self._camera_cb)
        #     self.camera_names = ["front", "bottom", "left", "right", "rear"]
        #     for cam_name in self.camera_names:
        #         self.camera_subscription = self.lc.subscribe(f"rect_image_{cam_name}", self._rect_camera_cb)
        # -----------------------------------------------------------------------------------------------------
        self.camera_image_left = None
        # 左侧相机图像
        self.camera_image_right = None
        # 右侧相机图像
        self.camera_image_front = None
        # 前侧相机图像
        self.camera_image_bottom = None
        # 底部相机图像
        self.camera_image_rear = None
        # 后侧相机图像

        self.body_loc = np.array([0, 0, 0])
        # 机身在空间中的位置？（相对于什么坐标系不确定）
        self.body_quat = np.array([0, 0, 0, 1])
        # 机器人的四元数？（相对于什么坐标系不确定）

    def get_body_linear_vel(self):   # 阅读完成
        """获取机身坐标系下的线速度"""
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_body_angular_vel(self):  # 阅读完成
        """计算机身坐标系下的角速度（作平滑处理）"""
        self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
                    1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_gravity_vector(self):   # 阅读完成
        """计算局部（机身）坐标系中的重力向量"""
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

    def get_contact_state(self):
        """获取足端接触状态"""
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):   # 阅读完成
        """获取机器人的欧拉角"""
        return self.euler

    def get_command(self):  # 阅读完成
        MODES_LEFT = ["body_height", "lat_vel", "stance_width"]
        # 与README中的左摇杆Mode A-C相对应
        MODES_RIGHT = ["step_frequency", "footswing_height", "body_pitch"]
        # 与README中的右摇杆Mode D-F相对应

        if self.left_upper_switch_pressed:
        # 如果L1键按下
            self.ctrlmode_left = (self.ctrlmode_left + 1) % 3
            # 按照"body_height"->"lat_vel"->"stance_width"->"body_height"->...的顺序切换控制模式 A-C
            self.left_upper_switch_pressed = False
            # 重置L1键按下标志
        if self.right_upper_switch_pressed:
        # 如果R1键按下
            self.ctrlmode_right = (self.ctrlmode_right + 1) % 3
            # 按照"step_frequency"->"footswing_height"->"body_pitch"->"step_frequency"->...的顺序切换控制模式 D-F
            self.right_upper_switch_pressed = False
            # 重置R1键按下标志

        MODE_LEFT = MODES_LEFT[self.ctrlmode_left]
        # 切换左摇杆控制模式
        MODE_RIGHT = MODES_RIGHT[self.ctrlmode_right]
        # 切换右摇杆控制模式

        # always in use
        cmd_x = 1 * self.left_stick[1]
        # 机器人前进速度指令
        cmd_yaw = -1 * self.right_stick[0]
        # 机器人偏航角速度指令

        # default values
        cmd_y = 0.  # -1 * self.left_stick[0]
        # 机器人横向速度指令
        cmd_height = 0.
        # 机身离地高度指令
        cmd_footswing = 0.08
        # 抬腿高度指令（没完全理解）我在阅读一个四足机器人强化学习控制的开源项目，其中有一部分跟步态相关的代码，如图片所示。我不知道图片中的phase、offset、bound、duration是什么意思，含义是什么，也不清楚Bound、Trot、Pace、Pronk四种步态对应的具体动作是怎样的（还没见过这些步态的实际运行场景，想象不出来），请你分别解释一下
        cmd_stance_length = 0.40
        # 站立前后距离指令
        cmd_ori_pitch = 0.
        # 俯仰指令（没完全理解）
        cmd_ori_roll = 0.
        # 横滚指令（没完全理解）
        cmd_freq = 3.0
        # 步频指令（没完全理解）

        # joystick commands
        # 遥控器摇杆指令
        if MODE_LEFT == "body_height":
            cmd_height = 0.3 * self.left_stick[0]
            # 机身离地高度指令
        elif MODE_LEFT == "lat_vel":
            cmd_y = 0.6 * self.left_stick[0]
            # 机器人横向速度指令
        elif MODE_LEFT == "stance_width":
            cmd_stance_width = 0.275 + 0.175 * self.left_stick[0]
            # 站立宽度指令
        if MODE_RIGHT == "step_frequency":
            min_freq = 2.0
            # 步频指令的最小值
            max_freq = 4.0
            # 步频指令的最大值
            cmd_freq = (1 + self.right_stick[1]) / 2 * (max_freq - min_freq) + min_freq
            # 步频指令
        elif MODE_RIGHT == "footswing_height":
            cmd_footswing = max(0, self.right_stick[1]) * 0.32 + 0.03
            # 抬腿高度指令
        elif MODE_RIGHT == "body_pitch":
            cmd_ori_pitch = -0.4 * self.right_stick[1]
            # 俯仰指令

        # gait buttons
        # 步态按键
        if self.mode == 0: # Press Button 'A' -> 'Bound'
            self.cmd_phase = 0.5
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 1: # Press Button 'B' -> 'Trot'
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 2: # Press Button 'X' -> 'Pace'
            self.cmd_phase = 0.0
            self.cmd_offset = 0.5
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 3: # Press Button 'Y' -> 'Pronk'
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.5
            self.cmd_duration = 0.5
        else: # Default Gait -> 'Trot'
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5

        return np.array([cmd_x, cmd_y, cmd_yaw, cmd_height, cmd_freq, self.cmd_phase, self.cmd_offset, self.cmd_bound,
                         self.cmd_duration, cmd_footswing, cmd_ori_pitch, cmd_ori_roll, cmd_stance_width,
                         cmd_stance_length, 0, 0, 0, 0, 0])
        # 返回控制指令

    def get_buttons(self):  # 阅读完成
        return np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch])
        # 返回[L2, L1, R2, R1]按键状态

    def get_dof_pos(self):   # 阅读完成
        """获取关节电机位置"""
        # print("dofposquery", self.joint_pos[self.joint_idxs])
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):   # 阅读完成
        """获取关节电机速度"""
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):   # 阅读完成
        """获取机器人的偏航角"""
        return self.euler[2]

    def get_body_loc(self):
        return np.array(self.body_loc)

    def get_body_quat(self):
        return np.array(self.body_quat)

    def get_camera_front(self):
        return self.camera_image_front

    def get_camera_bottom(self):
        return self.camera_image_bottom

    def get_camera_rear(self):
        return self.camera_image_rear

    def get_camera_left(self):
        return self.camera_image_left

    def get_camera_right(self):
        return self.camera_image_right

    def _legdata_cb(self, channel, data):
        # print("update legdata")
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}")

        msg = leg_control_data_lcmt.decode(data)
        # print(msg.q)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)
        # print(f"update legdata {msg.id}")

    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)

        self.euler = np.array(msg.rpy)

        self.R = get_rotation_matrix_from_rpy(self.euler)

        self.contact_state = 1.0 * (np.array(msg.contact_estimate) > 200)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)

    def _sensor_cb(self, channel, data):
        pass

    def _rc_command_cb(self, channel, data):

        msg = rc_command_lcmt.decode(data)


        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        # print(self.right_stick, self.left_stick)

# 是否要删除下面的camera相关函数？
# --------------------------------------------------
    # def _camera_cb(self, channel, data):
    #     msg = camera_message_lcmt.decode(data)

    #     img = np.fromstring(msg.data, dtype=np.uint8)
    #     img = img.reshape((3, 200, 464)).transpose(1, 2, 0)

    #     cam_id = int(channel[-1])
    #     if cam_id == 1:
    #         self.camera_image_front = img
    #     elif cam_id == 2:
    #         self.camera_image_bottom = img
    #     elif cam_id == 3:
    #         self.camera_image_left = img
    #     elif cam_id == 4:
    #         self.camera_image_right = img
    #     elif cam_id == 5:
    #         self.camera_image_rear = img
    #     else:
    #         print("Image received from camera with unknown ID#!")

    #     #im = Image.fromarray(img).convert('RGB')

    #     #im.save("test_image_" + channel + ".jpg")
    #     #print(channel)
            
    # def _rect_camera_cb(self, channel, data):
    #     message_types = [camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide,
    #                      camera_message_rect_wide, camera_message_rect_wide]
    #     image_shapes = [(116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3)]

    #     cam_name = channel.split("_")[-1]
    #     # print(f"received py from {cam_name}")
    #     cam_id = self.camera_names.index(cam_name) + 1

    #     msg = message_types[cam_id - 1].decode(data)

    #     img = np.fromstring(msg.data, dtype=np.uint8)
    #     img = np.flip(np.flip(
    #         img.reshape((image_shapes[cam_id - 1][2], image_shapes[cam_id - 1][1], image_shapes[cam_id - 1][0])),
    #         axis=0), axis=1).transpose(1, 2, 0)
    #     # print(img.shape)
    #     # img = np.flip(np.flip(img.reshape(image_shapes[cam_id - 1]), axis=0), axis=1)[:, :,
    #     #       [2, 1, 0]]  # .transpose(1, 2, 0)

    #     if cam_id == 1:
    #         self.camera_image_front = img
    #     elif cam_id == 2:
    #         self.camera_image_bottom = img
    #     elif cam_id == 3:
    #         self.camera_image_left = img
    #     elif cam_id == 4:
    #         self.camera_image_right = img
    #     elif cam_id == 5:
    #         self.camera_image_rear = img
    #     else:
    #         print("Image received from camera with unknown ID#!")
# --------------------------------------------------
            

    def poll(self, cb=None):    # 阅读完成
        t = time.time()
        # 时间戳t
        try:
            while True:
                timeout = 0.01
                # 超时时间为0.01s
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self): # 阅读完成
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        # 创建运行线程run_thread，回调函数为poll，并设置为非守护线程。
        # daemon=False参数指定了这个线程不是守护线程。这意味着当主线程结束时，这个线程不会自动被终止，它会继续运行直到回调函数执行完毕。
        self.run_thread.start()
        # 启动运行线程run_thread。

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    se.poll()
