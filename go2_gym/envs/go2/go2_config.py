from typing import Union

from params_proto import Meta

from go2_gym.envs.base.legged_robot_config import Cfg


def config_go2(Cnfg: Union[Cfg, Meta]): # 阅读完成
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        # 当action=0时，各关节角度的目标值
        'FL_hip_joint': 0.1,  # [rad]   左前机身关节
        'RL_hip_joint': 0.1,  # [rad]   左后机身关节
        'FR_hip_joint': -0.1,  # [rad]  右前机身关节
        'RR_hip_joint': -0.1,  # [rad]  右后机身关节

        'FL_thigh_joint': 0.8,  # [rad] 左前大腿关节
        'RL_thigh_joint': 1.,  # [rad]  左后大腿关节
        'FR_thigh_joint': 0.8,  # [rad] 右前大腿关节
        'RR_thigh_joint': 1.,  # [rad]  右后大腿关节

        'FL_calf_joint': -1.5,  # [rad]  左前小腿关节
        'RL_calf_joint': -1.5,  # [rad]  左后小腿关节
        'FR_calf_joint': -1.5,  # [rad]  右前小腿关节
        'RR_calf_joint': -1.5  # [rad]  右后小腿关节
    }

    _ = Cnfg.control
    _.control_type = 'P' # P
    _.stiffness = {'joint': 25.}  # [N*m/rad] 关节PD参数有待调整
    _.damping = {'joint': 0.6}  # [N*m*s/rad] 关节PD参数有待调整
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    # Unitree Go2: _.flip_visual_attachments = True
    # Unitree Go1: _.flip_visual_attachments = False
    _.flip_visual_attachments = True
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.34

    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 42
    _.observe_vel = False
    _.num_envs = 4096

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]

    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.added_mass_range = [-1, 3]
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.friction_range = [0.05, 4.5]
    _.randomize_restitution = True
    _.restitution_range = [0.0, 1.0]
    _.restitution = 0.5  # 地形恢复系数（这是类实例化后新添加的属性）
    # default terrain restitution
    # 默认地形恢复系数
    _.randomize_com_displacement = True
    _.com_displacement_range = [-0.1, 0.1]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.9, 1.1]
    _.randomize_Kp_factor = False
    _.Kp_factor_range = [0.8, 1.3]
    _.randomize_Kd_factor = False
    _.Kd_factor_range = [0.5, 1.5]
    _.rand_interval_s = 6
