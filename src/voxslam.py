from multiprocessing.managers import BaseManager
from time import sleep

import torch
import torch.multiprocessing as mp

from loggers import BasicLogger
from mapping import Mapping
from share import ShareData, ShareDataProxy
from tracking import Tracking
from utils.import_util import get_dataset
from visualization import Visualizer


class VoxSLAM:
    def __init__(self, args):
        self.args = args
        
        # logger (optional)
        self.logger = BasicLogger(args)  #用于创建日志
        # visualizer (optional)
        self.visualizer = Visualizer(args, self)  #用于显示出来

        # shared data 
        mp.set_start_method('spawn', force=True)  # multiprocessing包的初始化
        BaseManager.register('ShareData', ShareData, ShareDataProxy) #注册multiprocessing的BaseManager
        manager = BaseManager()
        manager.start()  #启动Manager
        self.share_data = manager.ShareData()  #用manager存放用于共享的东西
        # keyframe buffer 
        self.kf_buffer = mp.Queue(maxsize=1) #用队列来当做共享的buffer空间
        # data stream
        self.data_stream = get_dataset(args)  # get_dataset读取数据，相当于DataLoader
        # tracker 
        self.tracker = Tracking(args, self.data_stream, self.logger, self.visualizer)  #跟踪线程
        # mapper
        self.mapper = Mapping(args, self.logger, self.visualizer)  #建图线程
        # initialize map with first frame
        self.tracker.process_first_frame(self.kf_buffer)  #初始化地图的第一帧
        self.processes = []

    def start(self):
        mapping_process = mp.Process(
            target=self.mapper.spin, args=(self.share_data, self.kf_buffer))  # 先启动Mapping进行初始化
        mapping_process.start()
        print("initializing the first frame ...")
        sleep(5)
        tracking_process = mp.Process(
            target=self.tracker.spin, args=(self.share_data, self.kf_buffer))  #再启动Tracking
        tracking_process.start()

        vis_process = mp.Process(
            target=self.visualizer.spin, args=(self.share_data,)) #最后保存Mapping和Tracking的数据并显示出来
        self.processes = [tracking_process, mapping_process]

        if self.args.enable_vis:
            vis_process.start()   #如果开启显示模型，则显示
            self.processes += [vis_process]

    def wait_child_processes(self):
        for p in self.processes:  #多线程协同
            p.join()

    @torch.no_grad()
    def get_raw_trajectory(self):
        return self.share_data.tracking_trajectory   #在share的空间内获取跟踪轨迹信息

    @torch.no_grad()
    def get_keyframe_poses(self):  #在Mapper线程内获取关键帧的相机位姿
        keyframe_graph = self.mapper.keyframe_graph
        poses = []
        for keyframe in keyframe_graph:
            poses.append(keyframe.get_pose().detach().cpu().numpy())
        return poses
