import copy
import pickle as pkl

import numpy as np
import torch


def class_to_dict(obj) -> dict:  # 阅读完成
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
            print(key)
            element = class_to_dict(val)
        result[key] = element
    return result


class MultiLogger:
    def __init__(self): # 阅读完成
        self.loggers = {}

    def add_robot(self, name, cfg): # 阅读完成
        print(name, cfg)
        self.loggers[name] = EpisodeLogger(cfg)

    def log(self, name, info):
        self.loggers[name].log(info)

    def save(self, filename):   # 阅读完成
        """将日志信息保存到指定的文件中"""
        with open(filename, 'wb') as file:
            logdict = {}
            for key in self.loggers.keys():
                logdict[key] = [class_to_dict(self.loggers[key].cfg), self.loggers[key].infos]
            pkl.dump(logdict, file)
            print(f"Saved log! Number of timesteps: {[len(self.loggers[key].infos) for key in self.loggers.keys()]}; Path: {filename}")

    def read_metric(self, metric, robot_name=None):
        if robot_name is None:
            robot_name = list(self.loggers.keys())[0]
        logger = self.loggers[robot_name]

        metric_arr = []
        for info in logger.infos:
            metric_arr += [info[metric]]
        return np.array(metric_arr)

    def reset(self):
        for key, log in self.loggers.items():
            log.reset()


class EpisodeLogger:
    def __init__(self, cfg):    # 阅读完成
        self.infos = []
        self.cfg = cfg

    def log(self, info):     # 阅读完成
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].detach().cpu().numpy()

            if isinstance(info[key], dict):
                continue
            elif "image" not in key:
                info[key] = copy.deepcopy(info[key])

        self.infos += [dict(info)]

    def reset(self):       # 阅读完成
        self.infos = []
