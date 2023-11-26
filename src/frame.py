import torch
import torch.nn as nn
import numpy as np
from se3pose import OptimizablePose
from utils.sample_util import *

rays_dir = None

class RGBDFrame(nn.Module): #RGBDFrame类，用于存放一帧的RGBD数据
    def __init__(self, fid, rgb, depth, K, pose=None) -> None:
        super().__init__()
        self.stamp = fid #帧的id
        self.h, self.w = depth.shape #深度图的高和宽
        self.rgb = rgb.cuda()
        self.depth = depth.cuda() #/ 2
        self.K = K #相机内参
        # self.register_buffer("rgb", rgb)
        # self.register_buffer("depth", depth)

        if pose is not None:  #如果有位姿，则将位姿转换为优化位姿
            pose[:3, 3] += 10
            pose = torch.tensor(pose, requires_grad=True, dtype=torch.float32)
            self.pose = OptimizablePose.from_matrix(pose) #将位姿转换为优化位姿(李代数形式)
            self.optim = torch.optim.Adam(self.pose.parameters(), lr=1e-3) #将位姿塞入Adam
        else:
            self.pose = None
            self.optim = None
        self.precompute() #如果没有射线方向，则计算射线方向

    def get_pose(self): #获取位姿
        return self.pose.matrix()

    def get_translation(self): #获取位姿的平移
        return self.pose.translation()

    def get_rotation(self): #获取位姿的旋转
        return self.pose.rotation()

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None): #获取采样射线
        w = self.w if w == None else w #如果没有输入w，则使用默认的w
        h = self.h if h == None else h #如果没有输入h，则使用默认的h
        if K is None: #K 为相机内参，如果没有输入K，则使用自己计算的K
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy') #计算网格
        rays_d = torch.stack(
                    [(ix-K[0, 2]) / K[0,0],
                    (iy-K[1,2]) / K[1,1],
                    torch.ones_like(ix)], -1).float() #计算射线方向
        return rays_d

    @torch.no_grad()
    def precompute(self): #预计算
        global rays_dir
        if rays_dir is None: #如果没有射线方向，则计算射线方向
            rays_dir = self.get_rays(K=self.K).cuda()
        self.rays_d = rays_dir

    @torch.no_grad()
    def get_points(self): #获取深度图中的点
        vmap = self.rays_d * self.depth[..., None]
        return vmap[self.depth > 0].reshape(-1, 3)

    @torch.no_grad()
    def sample_rays(self, N_rays): #在采样射线中采样点，N_rays为采样射线的数量，返回有效采样点的mask
        self.sample_mask = sample_rays(
            torch.where(self.depth > 0, torch.ones_like(self.depth)[None, ...], torch.zeros_like(self.depth)[None, ...]), N_rays)[0, ...]
