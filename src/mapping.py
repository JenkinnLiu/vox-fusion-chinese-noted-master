from copy import deepcopy
import random
from time import sleep
import numpy as np

import torch
import trimesh

from criterion import Criterion
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property
from variations.render_helpers import bundle_adjust_frames
from utils.mesh_util import MeshExtractor

torch.classes.load_library(
    "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")


def get_network_size(net):
    size = 0
    for param in net.parameters():
        size += param.element_size() * param.numel()
    return size / 1024 / 1024


class Mapping:
    def __init__(self, args, logger: BasicLogger, vis=None, **kwargs):
        super().__init__()
        self.args = args
        self.logger = logger  #记录具体数据的日志生成器log
        self.visualizer = vis
        self.decoder = get_decoder(args).cuda() #decoder，MLP

        self.loss_criteria = Criterion(args) #初始化四个loss的具体参数
        self.keyframe_graph = []  # 存储关键帧
        self.initialized = False

        mapper_specs = args.mapper_specs

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)   #checkpoint_frequency采样点的频率？默认为-1
        self.final_iter = get_property(mapper_specs, "final_iter", 0) #0
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)  #8
        self.save_data_freq = get_property(
            args.debug_args, "save_data_freq", 0)  #  保存data的频率，默认为0

        # required args
        # self.overlap_th = mapper_specs["overlap_th"]
        self.voxel_size = mapper_specs["voxel_size"]  #体素大小，0.2
        self.window_size = mapper_specs["window_size"] #优化滑动窗口大小,4
        self.num_iterations = mapper_specs["num_iterations"] # 建图迭代次数，10
        self.n_rays = mapper_specs["N_rays_each"]  #每次采样所用采样射线的个数？1024
        self.sdf_truncation = args.criteria["sdf_truncation"] #SDF终止距离？0.1
        self.max_voxel_hit = mapper_specs["max_voxel_hit"] #采样射线最多碰撞的体素个数？10
        self.step_size = mapper_specs["step_size"]  #步长0.1
        self.step_size = self.step_size * self.voxel_size  #步长*体素大小，应该是每个体素处理10步的意思
        self.max_distance = args.data_specs["max_depth"]  #八叉树最深距离还是采样射线最大距离？

        embed_dim = args.decoder_specs["in_dim"] #decoder输入维度，16维（论文中提到过）
        use_local_coord = mapper_specs["use_local_coord"]  #在本地建图？
        self.embed_dim = embed_dim - 3 if use_local_coord else embed_dim #如果use_local_coord，则输入13维
        num_embeddings = mapper_specs["num_embeddings"]  #emdeddings的数量，20000
        self.mesh_freq = args.debug_args["mesh_freq"]# 建立mesh网格的频率,50
        self.mesher = MeshExtractor(args)  #初始化Mesh网格

        self.embeddings = torch.zeros(
            (num_embeddings, self.embed_dim),
            requires_grad=True, dtype=torch.float32,  #初始化特征嵌入embeddings，20000个16维的体素，
            device=torch.device("cuda"))
        torch.nn.init.normal_(self.embeddings, std=0.01)  #初始化为normal分布
        self.embed_optim = torch.optim.Adam([self.embeddings], lr=5e-3)  #将embeddings放入Adam
        self.model_optim = torch.optim.Adam(self.decoder.parameters(), lr=5e-3)  #将decoder放入Adam

        self.svo = torch.classes.svo.Octree()  #初始化八叉树
        self.svo.init(256, embed_dim, self.voxel_size)  #

        self.frame_poses = []  #每个帧的位姿
        self.depth_maps = [] #深度图
        self.last_tracked_frame_id = 0  #最后跟踪的帧id

    def spin(self, share_data, kf_buffer):  #开始建图
        print("mapping process started!")
        while True:
            # torch.cuda.empty_cache()
            if not kf_buffer.empty():   #如果存储关键帧的buffer里有东西
                tracked_frame = kf_buffer.get()  #获取buffer里的跟踪的关键帧
                # self.create_voxels(tracked_frame)

                if not self.initialized:  #如果还没初始化，那就初始化一下
                    if self.mesher is not None:  #如果已经建好网格mesh了
                        self.mesher.rays_d = tracked_frame.get_rays()  #获取当前帧采样射线的方向
                    self.create_voxels(tracked_frame)  #根据当前跟踪的帧来创建体素
                    self.insert_keyframe(tracked_frame)  #插入关键帧
                    while kf_buffer.empty(): #一直利用跟踪线程分享的数据建图，直到存储关键帧的buffer不为空
                        self.do_mapping(share_data)
                        # self.update_share_data(share_data, tracked_frame.stamp)
                    self.initialized = True  #初始化完成
                else:  #如果已经初始化
                    self.do_mapping(share_data, tracked_frame)  #正常利用跟踪线程分享的数据建图
                    self.create_voxels(tracked_frame)  #根据当前跟踪的帧来创建体素

                    #下面就是论文里的，避免在已建图的场景不插入关键帧的情况，这里每隔50帧必须插入关键帧
                    # if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                    if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                        self.insert_keyframe(tracked_frame)
                        print(
                            f"********** current num kfs: { len(self.keyframe_graph) } **********")

                # self.create_voxels(tracked_frame)
                tracked_pose = tracked_frame.get_pose().detach()  #得到当前跟踪帧的相机位姿
                ref_pose = self.current_keyframe.get_pose().detach()  #当前关键帧的的位姿
                rel_pose = torch.linalg.inv(ref_pose) @ tracked_pose  #rel_pose=tracked_pose/ref_pose，存储位姿的相对变化量（可以理解为李代数）
                self.frame_poses += [(len(self.keyframe_graph) -   #存储位姿
                                      1, rel_pose.cpu())]
                self.depth_maps += [tracked_frame.depth.clone().cpu()]  #存储当前跟踪帧的深度

                if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:  #如建立网格mesh的频率>0并且轮到建立网格mesh的时候了
                    self.logger.log_mesh(self.extract_mesh(  #建立网格mesh，并将其数据生成log日志
                        res=self.mesh_res, clean_mesh=True), name=f"mesh_{tracked_frame.stamp:05d}.ply")

                if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                    self.save_debug_data(tracked_frame)  #如果到了存储data的时机，则save_debug_data
            elif share_data.stop_mapping: #如果收到share_data停止建图的指令，则停止建图
                break

        print(f"********** post-processing {self.final_iter} steps **********")  #停止建图后还要做一些事情
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(share_data, tracked_frame=None,  #最后再建个图，把体素网格优化一下
                            update_pose=False, update_decoder=False)

        print("******* extracting final mesh *******") #提取最终的mesh网格
        pose = self.get_updated_poses()  #更新位姿
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False)  #提取mesh网格
        self.logger.log_ckpt(self)  #生成log日志，将数据存进日志里
        self.logger.log_numpy_data(np.asarray(pose), "frame_poses")
        self.logger.log_mesh(mesh)
        self.logger.log_numpy_data(self.extract_voxels(), "final_voxels")
        print("******* mapping process died *******")

    def do_mapping(   #进行一次建图操作
            self,
            share_data,  #共享data
            tracked_frame=None,   #当前跟踪的帧
            update_pose=True,  #更新的位姿
            update_decoder=True  #更新得的decoder
    ):
        # self.map.create_voxels(self.keyframe_graph[0])
        self.decoder.train()  #调用torch的train函数进行训练
        optimize_targets = self.select_optimize_targets(tracked_frame) #选择优化目标(优化目标为关键帧)
        # optimize_targets = [f.cuda() for f in optimize_targets]

        bundle_adjust_frames(  #BA优化的帧,有如下几个参数
            optimize_targets,  #优化目标
            self.map_states, #map的状态
            self.decoder, # decoder， MLP
            self.loss_criteria, #loss的具体参数
            self.voxel_size, #体素大小
            self.step_size, #步长
            self.n_rays, #每次采样所用采样射线的个数
            self.num_iterations, #建图迭代次数
            self.sdf_truncation, #SDF终止距离?
            self.max_voxel_hit, #采样射线最多碰撞的体素个数
            self.max_distance, #八叉树最深距离还是采样射线最大距离？
            learning_rate=[1e-2, 1e-3], #学习率
            embed_optim=self.embed_optim, #优化器
            model_optim=self.model_optim if update_decoder else None,   #模型的优化器
            update_pose=update_pose, #更新位姿
        )

        # optimize_targets = [f.cpu() for f in optimize_targets]
        self.update_share_data(share_data)
        # sleep(0.01)

    def select_optimize_targets(self, tracked_frame=None): #选择优化目标
        # TODO: better ways
        targets = []
        selection_method = 'random'
        if len(self.keyframe_graph) <= self.window_size:#如果关键帧的数量小于优化滑动窗口大小
            targets = self.keyframe_graph[:] #则优化目标为所有关键帧
        elif selection_method == 'random': #如果选择方法不变（为随机）
            targets = random.sample(self.keyframe_graph, self.window_size) #则优化目标为随机选择的关键帧
        elif selection_method == 'overlap': #如果选择方法为overlap
            raise NotImplementedError( #则抛出异常
                f"seletion method {selection_method} unknown") #选择方法未知

        if tracked_frame is not None and tracked_frame != self.current_keyframe: #如果当前跟踪帧不为空且不是当前关键帧
            targets += [tracked_frame] #当前跟踪帧纳入优化目标
        return targets

    def update_share_data(self, share_data, frameid=None): #更新共享数据share_data
        share_data.decoder = deepcopy(self.decoder).cpu()
        tmp_states = {}
        for k, v in self.map_states.items():
            tmp_states[k] = v.detach().cpu()
        share_data.states = tmp_states
        # self.last_tracked_frame_id = frameid

    def insert_keyframe(self, frame): #将当前帧插入关键帧
        # kf check
        print("insert keyframe")
        self.current_keyframe = frame
        self.keyframe_graph += [frame]
        # self.update_grid_features()

    def create_voxels(self, frame): #根据当前帧来创建体素
        points = frame.get_points().cuda() #获取当前帧的采样点（=rays_d*depth）
        pose = frame.get_pose().cuda()  #获取当前帧的位姿
        points = points@pose[:3, :3].transpose(-1, -2) + pose[:3, 3] #将采样点变换到世界坐标系下
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor') #将采样点变换到体素坐标系下
        # 等价于voxels = points / self.voxel_size #将采样点变换到体素坐标系下
        self.svo.insert(voxels.cpu().int()) #将体素插入八叉树
        self.update_grid_features() #更新特征网格

    @torch.enable_grad()
    def update_grid_features(self): #更新特征网格
        voxels, children, features = self.svo.get_centres_and_children() #获取八叉树的体素，子节点，特征
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size #计算体素的中心点
        children = torch.cat([children, voxels[:, -1:]], -1) #将子节点和体素的特征拼接起来

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxel_vertex_idx"] = features.cuda() #将特征，中心点，子节点等放入map_states
        map_states["voxel_center_xyz"] = centres
        map_states["voxel_structure"] = children
        map_states["voxel_vertex_emb"] = self.embeddings
        self.map_states = map_states

    @torch.no_grad()
    def get_updated_poses(self): #获取更新后的位姿
        frame_poses = []
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i] #获取参考帧的id和相对位姿
            ref_frame = self.keyframe_graph[ref_frame_ind]
            ref_pose = ref_frame.get_pose().detach().cpu() #获取参考帧的位姿
            pose = ref_pose @ rel_pose #计算绝对位姿=参考帧位姿*相对位姿
            frame_poses += [pose.detach().cpu().numpy()]
        return frame_poses

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False): #提取mesh网格
        sdf_network = self.decoder
        sdf_network.eval() #调用torch的eval函数进行评估

        voxels, _, features = self.svo.get_centres_and_children() #获取八叉树的体素，子节点，特征
        index = features.eq(-1).any(-1) #获取特征中为-1的体素
        voxels = voxels[~index, :] #将特征中为-1的体素去掉
        features = features[~index, :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = features.cuda() #将特征，中心点，等放入encoder_states
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["voxel_vertex_emb"] = self.embeddings

        frame_poses = self.get_updated_poses() #获取更新后的位姿
        mesh = self.mesher.create_mesh( #创建mesh网格
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=frame_poses[-1], depth_maps=self.depth_maps[-1],
            clean_mseh=clean_mesh, require_color=True, offset=-10, res=res)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, offset=-10):
        voxels, _, features = self.svo.get_centres_and_children() #获取八叉树的体素，子节点，特征
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
            self.voxel_size + offset
        print(torch.max(features)-torch.count_nonzero(index))
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame, offset=-10): #保存debug数据
        """
        save per-frame voxel, mesh and pose  保存每帧的体素，网格和位姿
        """
        pose = tracked_frame.get_pose().detach().cpu().numpy()
        pose[:3, 3] += offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels().detach().cpu().numpy()
        keyframe_poses = [p.get_pose().detach().cpu().numpy()
                          for p in self.keyframe_graph]

        for f in frame_poses:
            f[:3, 3] += offset
        for kf in keyframe_poses:
            kf[:3, 3] += offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": keyframe_poses,
            "is_keyframe": (tracked_frame == self.current_keyframe)
        }, tracked_frame.stamp)
