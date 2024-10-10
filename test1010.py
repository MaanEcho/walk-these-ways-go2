from typing import Union
from params_proto import PrefixProto, Meta

class Cfg(PrefixProto, cli=False):
    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}
    
    class commands(PrefixProto, cli=False):
        num_lin_vel_bins = 10
        num_ang_vel_bins = 10
    
    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_ang_vel = 0.5
        tracking_lin_vel = 0.5
        tracking_contacts_shaped_vel = 0.5
        tracking_contacts_shaped_force = 0.5

def config_go2(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

def train_go2(headless=True):
    config_go2(Cfg)

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90
    Cfg.init_state.test = True

if __name__ == '__main__':
    train_go2(headless=False)