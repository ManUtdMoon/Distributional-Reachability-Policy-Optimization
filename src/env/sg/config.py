from pathlib import Path
import sys

from src.defaults import ROOT_DIR
ROOT_DIR = Path(ROOT_DIR)

assert ROOT_DIR.is_dir(), ROOT_DIR

XML_DIR = ROOT_DIR / 'src' / 'env' / 'sg' / 'xmls'

point_goal_config = {
    'robot_base': str(XML_DIR / 'point.xml'),

    'task': 'goal',

    'lidar_num_bins': 16,
    'lidar_alias': True,

    'constrain_hazards': True,
    'constrain_indicator': False,

    'hazards_num': 4,
    'hazards_keepout': 0.4,
    'hazards_size': 0.15,
    'hazards_cost': 1.0,

    'goal_keepout': 0.4,
    'goal_size': 0.3,
    
    'reward_goal': 0.0,

    '_seed': None
}

car_goal_config = {
    **point_goal_config,
    'robot_base': str(XML_DIR / 'car.xml'),
}

doggo_goal_config = {
    **point_goal_config,
    'robot_base': str(XML_DIR / 'doggo.xml'),
    'sensors_obs': 
        ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'] +
        [
            'touch_ankle_1a', 'touch_ankle_2a', 
            'touch_ankle_3a', 'touch_ankle_4a',
            'touch_ankle_1b', 'touch_ankle_2b', 
            'touch_ankle_3b', 'touch_ankle_4b'
        ]
}