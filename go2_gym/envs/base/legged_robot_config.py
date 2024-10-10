# License: see [LICENSE, LICENSES/legged_gym/LICENSE]
from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):  # 阅读完成
    # cli=False在这里不是常规的python语法。应该是跟params_proto库有关，可能这个库允许在类的定义中传递额外的参数。

    # 环境
    class env(PrefixProto, cli=False):
        num_envs = 4096 # 环境数量
        num_observations = 235  # 观测空间维度
        num_scalar_observations = 42    # 标量观测空间维度？（不确定理解是否正确）
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        # 如果不是None，step()函数会返回一个特权观测缓冲区（用于非对称训练的critic观测）。否则，返回None。
        num_privileged_obs = 18 # 特权观测维度
        privileged_future_horizon = 1   # 不清楚具体含义
        num_actions = 12    # 动作空间维度
        num_observation_history = 15    # 观测历史长度
        env_spacing = 3.    # 不清楚具体含义
        # not used with heightfields/trimeshes
        # heightfields/trimeshes不使用
        send_timeouts = True    # 是否发送超时信息给算法
        # send time out information to the algorithm
        # 是否发送超时信息给算法
        episode_length_s = 20   # 回合长度（以秒为单位）
        # episode length in seconds
        # 回合长度（以秒为单位）
        observe_vel = True  # 是否观测速度
        observe_only_ang_vel = False    # 是否仅观测角速度
        observe_only_lin_vel = False    # 是否仅观测线速度
        observe_yaw = False  # 是否观测偏航角
        observe_contact_states = False  # 是否观测接触状态
        observe_command = True  # 是否观测指令
        observe_height_command = False  # 是否观测高度指令
        observe_gait_commands = False   # 是否观测步态指令
        observe_timing_parameter = False    # 是否观测时间参数（不理解具体含义）
        observe_clock_inputs = False    # 是否观测时钟输入？（不确定理解是否正确）
        observe_two_prev_actions = False    # 是否观测前两个时间步的动作
        observe_imu = False  # 是否观测IMU
        record_video = True # 是否录制视频
        recording_width_px = 360    # 录制视频的宽度（像素）
        recording_height_px = 240   # 录制视频的高度（像素）
        recording_mode = "COLOR"    # 录制视频的模式
        num_recording_envs = 1  # 录制视频的环境数量
        debug_viz = False   # 是否显示调试视觉？（不理解具体含义）
        all_agents_share = False    # 是否所有智能体共享参数？（不确定理解是否正确）

        priv_observe_friction = True    # 是否特权观测摩擦力
        priv_observe_friction_indep = True  # 不理解具体含义
        priv_observe_ground_friction = False    # 是否特权观测地面摩擦力
        priv_observe_ground_friction_per_foot = False   # 是否特权观测每个足端的地面摩擦力
        priv_observe_restitution = True # 是否特权观测恢复力
        priv_observe_base_mass = True   # 是否特权观测机身质量
        priv_observe_com_displacement = True    # 是否特权观测质心位移（不理解具体含义）
        priv_observe_motor_strength = False # 是否特权观测电机强度？（不确定理解是否正确）
        priv_observe_motor_offset = False   # 是否特权观测电机偏移？（不理解具体含义）
        priv_observe_joint_friction = True   # 是否特权观测关节摩擦力
        priv_observe_Kp_factor = True    # 是否特权观测Kp因子
        priv_observe_Kd_factor = True    # 是否特权观测Kd因子
        priv_observe_contact_forces = False  # 是否特权观测足端接触力
        priv_observe_contact_states = False  # 是否特权观测足端接触状态
        priv_observe_body_velocity = False  # 是否特权观测机身速度
        priv_observe_foot_height = False    # 是否特权观测足端高度
        priv_observe_body_height = False    # 是否特权观测机身高度
        priv_observe_gravity = False      # 是否特权观测重力
        priv_observe_terrain_type = False    # 是否特权观测地形类型
        priv_observe_clock_inputs = False    # 是否特权观测时钟输入？（不理解具体含义）
        priv_observe_doubletime_clock_inputs = False    # 是否特权观测双时间步时钟输入？（不理解具体含义）
        priv_observe_halftime_clock_inputs = False    # 是否特权观测半时间步时钟输入？（不理解具体含义）
        priv_observe_desired_contact_states = False  # 是否特权观测期望足端接触状态
        priv_observe_dummy_variable = False  # 是否特权观测虚拟变量？（不理解具体含义）

    # 地形
    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'   # 地形类型？
        # "heightfield" # none, plane, heightfield or trimesh
        # “高度场” # 无，平面，高度场或三角网格
        horizontal_scale = 0.1  # 水平方向尺度大小
        # [m] 0.1
        vertical_scale = 0.005  # 竖直方向尺度大小
        # [m]
        border_size = 0 # 边界尺寸
        # 25 # [m]
        curriculum = True   # 是否使用课程学习
        static_friction = 1.0   # 静摩擦力
        dynamic_friction = 1.0  # 动摩擦力
        restitution = 0.0   # 弹性系数
        terrain_noise_magnitude = 0.1   # 地形噪声大小
        # rough terrain only:
        # 仅适用于粗糙地形：
        terrain_smoothness = 0.005  # 地形平滑度
        measure_heights = True  # 是否测量高度
        # 1mx1.6m rectangle (without center line)
        # 1m x 1.6m矩形（不含中心线）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # x方向上的测量点
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # y方向上的测量点
        selected = False    # 是否选择一个独特的地形类型并传递所有参数
        # select a unique terrain type and pass all arguments
        # 选择一个独特的地形类型并传递所有参数
        terrain_kwargs = None   # 所选择的地形类型的参数字典
        # Dict of arguments for selected terrain
        min_init_terrain_level = 0  # 最小初始地形级别
        max_init_terrain_level = 5  # 最大初始地形级别
        # starting curriculum state
        # 初始课程状态
        terrain_length = 0.5    # 地形长度
        # defaul = 8.
        terrain_width = 0.5    # 地形宽度
        # default = 8.
        num_rows = 10   # 地形行数
        # number of terrain rows (levels)
        # 地形行数（等级）
        num_cols = 20   # 地形列数
        # number of terrain cols (types)
        # 地形列数（类型）
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 地形类型：[平滑斜坡，粗糙斜坡，上升台阶，下降台阶，离散]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]   # 地形比例
        # trimesh only:
        # 仅适用于三角网格：
        slope_treshold = 0.75   # 斜坡阈值，单位：弧度
        # slopes above this threshold will be corrected to vertical surfaces
        # 斜坡的阈值，超过该阈值的斜坡将被修正为垂直表面
        difficulty_scale = 1.   # 难度缩放因子
        x_init_range = 1.   # x方向的初始范围
        y_init_range = 1.   # y方向的初始范围
        yaw_init_range = 0. # 偏航角初始范围
        x_init_offset = 0.  # x方向的初始偏移
        y_init_offset = 0.  # y方向的初始偏移
        teleport_robots = True  # 是否随机传送机器人
        teleport_thresh = 2.0   # 传送阈值
        max_platform_height = 0.2    # 最大平台高度
        center_robots = False    # 是否在地形中心放置机器人（不确定理解是否正确）
        center_span = 5 # 中心范围

    # 命令
    class commands(PrefixProto, cli=False):
        command_curriculum = False  # 是否使用课程学习（不确定理解是否正确）
        max_reverse_curriculum = 1. # 最大反向课程学习？（不确定理解是否正确）
        max_forward_curriculum = 1. # 最大正向课程学习？（不确定理解是否正确）
        yaw_command_curriculum = False  # 是否使用偏航角指令课程学习？（不确定理解是否正确）
        max_yaw_curriculum = 1.  # 最大偏航角指令课程学习？（不确定理解是否正确）
        exclusive_command_sampling = False  # 是否使用独占指令采样？（不确定理解是否正确）
        num_commands = 3    # 指令数量
        resampling_time = 10.   # 指令重采样时间
        # time before command are changed[s]
        # 指令改变之前的时间[秒]
        subsample_gait = False   # 是否使用步态子采样？（不确定理解是否正确）
        gait_interval_s = 10.   # 步态间隔时间？（不确定理解是否正确）
        # time between resampling gait params
        # 重采样步态参数之间的时间
        vel_interval_s = 10.    # 速度指令间隔时间？（不确定理解是否正确）
        jump_interval_s = 20.   # 跳跃指令间隔时间？（不确定理解是否正确）
        # time between jumps
        # 跳跃之间的时间
        jump_duration_s = 0.1   # 跳跃持续时间？（不确定理解是否正确）
        # duration of jump
        # 跳跃持续时间
        jump_height = 0.3   # 跳跃高度
        heading_command = True  # 是否使用航向指令
        # if true: compute ang vel command from heading error
        # 如果为真：根据航向误差计算角速度指令
        global_reference = False    # 是否使用全局参考？（不确定理解是否正确）
        observe_accel = False   # 是否观测加速度
        distributional_commands = False # 是否使用分布式指令？（不确定理解是否正确）
        curriculum_type = "RewardThresholdCurriculum"   # 课程学习类型
        lipschitz_threshold = 0.9   # lipschitz阈值
        # Lipschitz连续性：|f(x)-f(y)| <= L * |x-y|，lipschitz_threshold应该指的是公式中的L

        num_lin_vel_bins = 20   # 线速度分箱数量？（不确定理解是否正确）
        lin_vel_step = 0.3  # 线速度分箱步长？（不确定理解是否正确）
        num_ang_vel_bins = 20   # 角速度分箱数量？（不确定理解是否正确）
        ang_vel_step = 0.3  # 角速度分箱步长？（不确定理解是否正确）
        distribution_update_extension_distance = 1  # 指令分布更新扩展距离？（不确定理解是否正确）
        curriculum_seed = 100   # 课程学习种子

        lin_vel_x = [-1.0, 1.0] # x方向的线速度范围
        # min max [m/s]
        lin_vel_y = [-1.0, 1.0] # y方向的线速度范围 
        # min max [m/s]
        ang_vel_yaw = [-1, 1]   # 偏航角速度范围
        # min max [rad/s]
        body_height_cmd = [-0.05, 0.05] # 机身高度指令范围（相对于机身默认离地高度）
        impulse_height_commands = False # 是否使用冲量高度指令？（不确定理解是否正确）

        limit_vel_x = [-10.0, 10.0] # x方向的线速度范围（和lin_vel_x有什么区别？）*
        limit_vel_y = [-0.6, 0.6]   # y方向的线速度范围（和lin_vel_y有什么区别？）*
        limit_vel_yaw = [-10.0, 10.0]   # 偏航角速度范围（和ang_vel_yaw有什么区别？）*
        limit_body_height = [-0.05, 0.05]   # 机身高度指令范围（和body_height_cmd有什么区别？）*
        #----------------------------------------------------
        # 之后需要重点关注
        limit_gait_phase = [0, 0.01]    # 步态相位范围*
        limit_gait_offset = [0, 0.01]   # 步态偏移范围*
        limit_gait_bound = [0, 0.01]    # 步态边界范围*
        limit_gait_frequency = [2.0, 2.01]   # 步态频率范围*
        limit_gait_duration = [0.49, 0.5]    # 步态持续时间范围*
        # 之后需要重点关注
        #----------------------------------------------------
        limit_footswing_height = [0.06, 0.061]  # 抬腿高度范围*
        limit_body_pitch = [0.0, 0.01]  # 机身俯仰角范围*
        limit_body_roll = [0.0, 0.01]   # 机身滚转角范围*
        limit_aux_reward_coef = [0.0, 0.01] # 辅助奖励系数范围*
        limit_compliance = [0.0, 0.01]  # 合规范围？（不理解具体含义）
        limit_stance_width = [0.0, 0.01]    # 站立宽度范围*
        limit_stance_length = [0.0, 0.01]   # 站立长度范围*

        num_bins_vel_x = 25 # x方向的线速度分箱数量？（不理解具体含义）
        num_bins_vel_y = 3  # y方向的线速度分箱数量？（不理解具体含义）
        num_bins_vel_yaw = 25   # 偏航角速度分箱数量？（不理解具体含义）
        num_bins_body_height = 1    # 机身高度指令分箱数量？（不理解具体含义）
        num_bins_gait_frequency = 11    # 步态频率分箱数量？（不理解具体含义）
        num_bins_gait_phase = 11    # 步态相位分箱数量？（不理解具体含义）
        num_bins_gait_offset = 2    # 步态偏移分箱数量？（不理解具体含义）
        num_bins_gait_bound = 2 # 步态边界分箱数量？（不理解具体含义）
        num_bins_gait_duration = 3  # 步态持续时间分箱数量？（不理解具体含义）
        num_bins_footswing_height = 1   # 抬腿高度分箱数量？（不理解具体含义）
        num_bins_body_pitch = 1 # 机身俯仰角分箱数量？（不理解具体含义）
        num_bins_body_roll = 1  # 机身滚转角分箱数量？（不理解具体含义）
        num_bins_aux_reward_coef = 1    # 辅助奖励系数分箱数量？（不理解具体含义）
        num_bins_compliance = 1    # 合规分箱数量？（不理解具体含义）
        num_bins_stance_width = 1    # 站立宽度分箱数量？（不理解具体含义）
        num_bins_stance_length = 1   # 站立长度分箱数量？（不理解具体含义）

        heading = [-3.14, 3.14] # 航向指令范围（不确定理解是否正确）

        #----------------------------------------------------
        # 之后需要重点关注
        gait_phase_cmd_range = [0.0, 0.01]  # 步态相位指令范围
        gait_offset_cmd_range = [0.0, 0.01]  # 步态偏移指令范围
        gait_bound_cmd_range = [0.0, 0.01]  # 步态边界指令范围
        gait_frequency_cmd_range = [2.0, 2.01]  # 步态频率指令范围
        gait_duration_cmd_range = [0.49, 0.5]    # 步态持续时间指令范围
        # 之后需要重点关注
        #----------------------------------------------------
        footswing_height_range = [0.06, 0.061]  # 抬腿高度指令范围
        body_pitch_range = [0.0, 0.01]  # 机身俯仰角指令范围
        body_roll_range = [0.0, 0.01]   # 机身滚转角指令范围
        aux_reward_coef_range = [0.0, 0.01]  # 辅助奖励系数指令范围
        compliance_range = [0.0, 0.01]  # 合规指令范围？（不理解具体含义）
        stance_width_range = [0.0, 0.01]    # 站立宽度指令范围
        stance_length_range = [0.0, 0.01]    # 站立长度指令范围
        # 感觉和前面的，从limit_footswing_height到limit_stance_length，内容重复了，可以考虑合并

        exclusive_phase_offset = True    # 是否独占步态相位和步态偏移？（不确定理解是否正确）
        binary_phases = False    # 是否二进制步态相位？（不确定理解是否正确）
        pacing_offset = False    # 是否使用步态偏移？（不确定理解是否正确）
        balance_gait_distribution = True    # 是否平衡步态分布？（不确定理解是否正确）
        gaitwise_curricula = True    # 是否使用步态课程学习？（不确定理解是否正确）

    # 课程门槛
    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # 跟踪线速度？（不理解具体含义）
        # closer to 1 is tighter
        # 靠近1越紧张？（不理解具体含义）
        tracking_ang_vel = 0.5  # 跟踪角速度？（不理解具体含义）
        tracking_contacts_shaped_force = 0.8    # 跟踪接触形变力？（不理解具体含义）
        # closer to 1 is tighter
        # 靠近1越紧张？（不理解具体含义）
        tracking_contacts_shaped_vel = 0.8  # 跟踪接触形变速度？（不理解具体含义）

    # 初始状态
    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]    # 位置
        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # 姿态（四元数）
        # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # 线速度
        # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # 角速度
        # x,y,z [rad/s]
        # target angles when action = 0.0
        # 当action=0.0时的目标角度
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}   # 默认关节角度

    # 控制
    class control(PrefixProto, cli=False):
        control_type = 'actuator_net'   # 控制类型（执行器网络）
        #'P'  # P: position, V: velocity, T: torques
        #'P'  # P: 角度控制，V：角速度控制，T：力矩控制
        # PD Drive parameters:
        # PD控制参数：
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}   # 刚度
        # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # 阻尼
        # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0   # 蹲下时，hip的力矩缩减比例？（不确定理解是否正确）
        # decimation: Number of control action updates @ sim DT per policy DT
        # 降采样：在每个策略时间步（policy DT）内，仿真时间步（sim DT）上控制动作更新的次数
        decimation = 4  # 降采样（不理解具体含义）
        # GPT-4o：在这个参数配置文件中，`decimation`参数表示控制决策更新的频率。具体而言，它指的是在模拟时间步长（`sim DT`）下，每个策略时间步长（`policy DT`）进行的控制动作更新的次数。简单来说，`decimation`用于控制动作的时间分辨率：数值越高，控制更新越频繁。

    # 资产
    class asset(PrefixProto, cli=False):
        file = ""   # 资产文件路径
        foot_name = "None"  # 足端名称
        # name of the feet bodies, used to index body state and contact force tensors
        # 足端名称，用于索引机器人状态和接触力张量
        penalize_contacts_on = []   # 惩罚接触的关节/足端名称
        terminate_after_contacts_on = []    # 关节/足端接触后终止的特定关节/足端名称
        disable_gravity = False # 禁用重力？（不确定理解是否正确）
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # 合并固定关节连接的体。可以通过将“<... dont_collapse="true">”添加到特定的固定关节中来保留它们
        collapse_fixed_joints = True    # 崩溃固定关节？（不确定理解是否正确）
        fix_base_link = False   # 是否固定机器人的基座
        # fixe the base of the robot
        # 是否固定机器人的基座
        default_dof_drive_mode = 3  # 默认关节驱动模式
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 见GymDofDriveModeFlags（0为无，1为位置目标，2为速度目标，3为力矩）
        self_collisions = 0 # 自碰撞
        # 1 to disable, 0 to enable...bitwise filter
        # 1为禁用，0为启用...位元过滤器
        #-----------------------------------------------------------------------
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        # 用胶囊代替圆柱，可以加快/更稳定的模拟
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        # Some .obj meshes must be flipped from y-up to z-up
        # 有些.obj网格必须从y-up翻转到z-up

        density = 0.001 # 密度
        angular_damping = 0.    # 角阻尼
        linear_damping = 0.    # 线阻尼
        max_angular_velocity = 1000.    # 最大角速度
        max_linear_velocity = 1000.    # 最大线速度
        armature = 0.    # 电枢？（不理解具体含义）
        thickness = 0.01    # 厚度？（不理解具体含义）

    # 域随机化
    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10    # 随机化间隔（秒）
        randomize_rigids_after_start = True # 是否在开始时随机化刚体？（不确定理解是否正确）
        randomize_friction = True    # 是否随机化摩擦力？（不确定理解是否正确）
        friction_range = [0.5, 1.25]    # 摩擦力范围
        # increase range
        # 增加范围
        randomize_restitution = False    # 是否随机化恢复力？
        restitution_range = [0, 1.0]    # 恢复力范围
        randomize_base_mass = False # 是否随机化基座质量？
        # add link masses, increase range, randomize inertia, randomize joint properties
        # 添加连杆质量，增加范围，随机化惯性，随机化关节属性
        added_mass_range = [-1., 1.]     # 附加质量范围
        randomize_com_displacement = False    # 是否随机化质心位移？
        # add link masses, increase range, randomize inertia, randomize joint properties
        # 增加连杆质量，增加范围，随机惯性，随机关节属性
        com_displacement_range = [-0.15, 0.15]  # 质心位移范围
        randomize_motor_strength = False    # 是否随机化电机强度？
        motor_strength_range = [0.9, 1.1]    # 电机强度范围
        randomize_Kp_factor = False    # 是否随机化Kp因子？
        Kp_factor_range = [0.8, 1.3]    # Kp因子范围
        randomize_Kd_factor = False    # 是否随机化Kd因子？
        Kd_factor_range = [0.5, 1.5]    # Kd因子范围
        gravity_rand_interval_s = 7  # 重力随机化间隔（秒）
        gravity_impulse_duration = 1.0  # 重力冲量持续时间（秒）
        randomize_gravity = False    # 是否随机化重力？
        gravity_range = [-1.0, 1.0]    # 重力范围
        push_robots = True   # 是否随机推撞机器人？
        push_interval_s = 15    # 推撞间隔（秒）
        max_push_vel_xy = 1.    # 最大推撞速度（xy方向）
        randomize_lag_timesteps = True   # 是否随机延迟时间步？（不理解具体含义）
        lag_timesteps = 6   # 延迟时间步？（不理解具体含义）

    # 奖励
    class rewards(PrefixProto, cli=False):
        only_positive_rewards = True    # 是否只包含正奖励
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 如果为真，负的总奖励将被剪切为零（避免早停问题）
        only_positive_rewards_ji22_style = False    # 是否只包含正奖励，JI22风格？（不确定理解是否正确）
        sigma_rew_neg = 5   # 负奖励的标准差？（不确定理解是否正确）
        reward_container_name = "CoRLRewards"   # 奖励容器名称
        tracking_sigma = 0.25   # 跟踪奖励的标准差
        # tracking reward = exp(-error^2/sigma)
        # 跟踪奖励 R = exp(-(error)^2/sigma)
        tracking_sigma_lat = 0.25   # 跟踪奖励的标准差（lat是指y方向吗？）
        # tracking reward = exp(-error^2/sigma)
        # 跟踪奖励 R = exp(-(error)^2/sigma)
        tracking_sigma_long = 0.25  # 跟踪奖励的标准差（long是指x方向吗？）
        # tracking reward = exp(-error^2/sigma)
        # 跟踪奖励 R = exp(-(error)^2/sigma)
        tracking_sigma_yaw = 0.25    # 跟踪奖励的标准差（yaw是指z方向角速度吗？）
        # tracking reward = exp(-error^2/sigma)
        # 跟踪奖励 R = exp(-(error)^2/sigma)
        soft_dof_pos_limit = 1. # 软关节位置限制
        # percentage of urdf limits, values above this limit are penalized
        # 软关节位置限制，超过这个限制的关节位置将被惩罚
        soft_dof_vel_limit = 1. # 软关节速度限制
        soft_torque_limit = 1.  # 软力矩限制
        base_height_target = 1. # 基座高度目标
        max_contact_force = 100.    # 最大足端接触力
        # forces above this value are penalized
        # 超过这个值的足端接触力将被惩罚
        use_terminal_body_height = False    # 是否使用终止高度？
        terminal_body_height = 0.20 # 终止高度
        use_terminal_foot_height = False    # 是否使用终止足端高度？
        terminal_foot_height = -0.005    # 终止足端高度
        use_terminal_roll_pitch = False # 是否使用终止俯仰/滚转角？
        terminal_body_ori = 0.5 # 终止身体姿态
        kappa_gait_probs = 0.07 # 步态概率？（不理解具体含义）
        gait_force_sigma = 50.  # 步态力矩标准差？（不理解具体含义）
        gait_vel_sigma = 0.5    # 步态速度标准差？（不理解具体含义）
        footswing_height = 0.09 # 抬腿高度？（不确定理解是否正确）

    # 奖励权重
    class reward_scales(ParamsProto, cli=False):
        termination = -0.0  # 终止奖励权重
        tracking_lin_vel = 1.0  # 线速度跟踪奖励权重
        tracking_ang_vel = 0.5  # 角速度跟踪奖励权重
        lin_vel_z = -2.0    # z方向线速度奖励权重
        ang_vel_xy = -0.05  # xy方向角速度奖励权重
        orientation = -0.   # 姿态奖励权重
        torques = -0.00001  # 力矩奖励权重
        dof_vel = -0.   # 关节速度奖励权重
        dof_acc = -2.5e-7   # 关节加速度奖励权重
        base_height = -0.   # 基座高度奖励权重
        feet_air_time = 1.0 # 足端悬空时间奖励权重
        collision = -1. # 碰撞奖励权重
        feet_stumble = -0.0 # 足端摔倒奖励权重
        action_rate = -0.01 # 动作速率奖励权重
        stand_still = -0.   # 静止站立奖励权重
        tracking_lin_vel_lat = 0.   # 线速度（lat）跟踪奖励权重
        tracking_lin_vel_long = 0.   # 线速度（long）跟踪奖励权重
        tracking_contacts = 0.   # 接触奖励权重
        tracking_contacts_shaped = 0.   # 接触奖励权重（带有形状）
        tracking_contacts_shaped_force = 0.   # 接触奖励权重（带有形状和力）
        tracking_contacts_shaped_vel = 0.   # 接触奖励权重（带有形状和速度）
        jump = 0.0   # 跳跃奖励权重
        energy = 0.0    # 能量奖励权重
        energy_expenditure = 0.0    # 能量消耗奖励权重
        survival = 0.0  # 生存奖励权重
        dof_pos_limits = 0.0    # 关节位置限制奖励权重
        feet_contact_forces = 0.    # 足端接触力奖励权重
        feet_slip = 0.  # 足端滑移奖励权重
        feet_clearance_cmd_linear = 0.  # 足端清除指令（线性）奖励权重？（感觉理解不太对）
        dof_pos = 0.    # 关节位置奖励权重
        action_smoothness_1 = 0.    # 动作平滑度奖励权重1
        action_smoothness_2 = 0.    # 动作平滑度奖励权重2
        base_motion = 0.    # 基座运动奖励权重
        feet_impact_vel = 0.0    # 足端撞击速度奖励权重
        raibert_heuristic = 0.0 # raibert启发式奖励权重（不理解具体含义）

    # 归一化
    class normalization(PrefixProto, cli=False):
        clip_observations = 100.    # 观测值截断范围
        clip_actions = 100. # 动作值截断范围

        friction_range = [0.05, 4.5]    # 摩擦力范围
        ground_friction_range = [0.05, 4.5] # 地面摩擦力范围
        restitution_range = [0, 1.0]    # 恢复力范围
        added_mass_range = [-1., 3.]     # 附加质量范围
        com_displacement_range = [-0.1, 0.1]    # 质心位移范围
        motor_strength_range = [0.9, 1.1]    # 电机强度范围
        motor_offset_range = [-0.05, 0.05]    # 电机偏移范围
        Kp_factor_range = [0.8, 1.3]    # Kp因子范围
        Kd_factor_range = [0.5, 1.5]    # Kd因子范围
        joint_friction_range = [0.0, 0.7]    # 关节摩擦力范围
        contact_force_range = [0.0, 50.0]    # 接触力范围
        contact_state_range = [0.0, 1.0]    # 接触状态范围
        body_velocity_range = [-6.0, 6.0]   # 机身速度范围
        foot_height_range = [0.0, 0.15] # 足端高度范围
        body_height_range = [0.0, 0.60] # 机身高度范围
        gravity_range = [-1.0, 1.0] # 重力范围
        motion = [-0.01, 0.01]  # 运动奖励范围

    # 观测权重？
    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0   # 线速度观测权重
        ang_vel = 0.25  # 角速度观测权重
        dof_pos = 1.0   # 关节位置观测权重
        dof_vel = 0.05  # 关节速度观测权重
        imu = 0.1   # IMU观测权重
        height_measurements = 5.0   # 高度测量观测权重
        friction_measurements = 1.0 # 摩擦力测量观测权重
        body_height_cmd = 2.0   # 机身高度指令观测权重
        gait_phase_cmd = 1.0    # 步态相位指令观测权重
        gait_freq_cmd = 1.0 # 步态频率指令观测权重
        footswing_height_cmd = 0.15 # 抬腿高度指令观测权重
        body_pitch_cmd = 0.3    # 机身俯仰指令观测权重
        body_roll_cmd = 0.3 # 机身滚转指令观测权重
        aux_reward_cmd = 1.0    # 辅助奖励指令观测权重
        compliance_cmd = 1.0    # 合规指令观测权重？（不理解具体含义）
        stance_width_cmd = 1.0  # 站立宽度指令观测权重
        stance_length_cmd = 1.0 # 站立长度指令观测权重
        segmentation_image = 1.0    # 分割图像观测权重？（不理解具体含义）
        rgb_image = 1.0 # RGB图像观测权重
        depth_image = 1.0   # 深度图像观测权重

    # 噪声
    class noise(PrefixProto, cli=False):
        add_noise = True    # 是否添加噪声
        noise_level = 1.0   # 噪声水平
        # scales other values
        # 缩放其他值

    # 噪声权重
    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01  # 关节位置噪声权重
        dof_vel = 1.5   # 关节速度噪声权重
        lin_vel = 0.1   # 线速度噪声权重
        ang_vel = 0.2   # 角速度噪声权重
        imu = 0.1   # IMU噪声权重
        gravity = 0.05  # 重力噪声权重
        contact_states = 0.05   # 接触状态噪声权重
        height_measurements = 0.1   # 高度测量噪声权重
        friction_measurements = 0.0 # 摩擦力测量噪声权重
        segmentation_image = 0.0    # 分割图像噪声权重
        rgb_image = 0.0  # RGB图像噪声权重
        depth_image = 0.0   # 深度图像噪声权重

    # viewer camera:
    # 观察者相机设置
    class viewer(PrefixProto, cli=False):
        ref_env = 0 # 参考环境
        pos = [10, 0, 6]    # 位置*
        # [m]
        lookat = [11., 5, 3.]   # 观察目标*
        # [m]

    # 仿真参数设置
    class sim(PrefixProto, cli=False):
        dt = 0.005  # 仿真步长*
        substeps = 1    # 子步数*
        gravity = [0., 0., -9.81]   # 重力向量*
        # [m/s^2]
        up_axis = 1 # 竖直方向坐标轴*
        # 0 is y, 1 is z
        # 0：y轴，1：z轴

        use_gpu_pipeline = True # 是否使用GPU管线*

        class physx(PrefixProto, cli=False):
            num_threads = 10    # 线程数*
            solver_type = 1 # 求解器类型*
            # 0: pgs, 1: tgs
            num_position_iterations = 4 # 位置迭代次数*
            num_velocity_iterations = 0 # 速度迭代次数*
            contact_offset = 0.01   # 接触偏移量*
            # [m]
            rest_offset = 0.0   # 静止偏移量？（不确定理解是否正确）*
            # [m]
            bounce_threshold_velocity = 0.5 # 弹跳阈值速度？（不理解具体含义）*
            # 0.5 [m/s]
            max_depenetration_velocity = 1.0    # 最大depenetration速度*
            max_gpu_contact_pairs = 2 ** 23 # 最大GPU接触对数*
            # 2**24 -> needed for 8000 envs and more
            # 8000个环境（甚至更多）需要24位接触对数
            default_buffer_size_multiplier = 5  # *
            # 默认缓冲区大小倍数
            contact_collection = 2  # 接触收集类型*
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # 0：从不，1：最后一个子步，2：所有子步（默认=2）
