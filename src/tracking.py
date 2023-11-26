import torch
from tqdm import tqdm

from criterion import Criterion
from frame import RGBDFrame
from utils.import_util import get_property
from utils.profile_util import Profiler
from variations.render_helpers import fill_in, render_rays, track_frame


class Tracking:
    def __init__(self, args, data_stream, logger, vis, **kwargs):
        self.args = args
        self.last_frame_id = 0 #上一帧的id
        self.last_frame = None  #上一帧的数据

        self.data_stream = data_stream #数据流(相当于DataLoader)
        self.logger = logger
        self.visualizer = vis #可视化
        self.loss_criteria = Criterion(args) #loss函数

        self.render_freq = args.debug_args["render_freq"] #渲染频率50
        self.render_res = args.debug_args["render_res"] #渲染分辨率，(640, 480)

        self.voxel_size = args.mapper_specs["voxel_size"] #体素大小，0.2
        self.N_rays = args.tracker_specs["N_rays"] #采样射线数量，1024
        self.num_iterations = args.tracker_specs["num_iterations"] #优化迭代次数，30
        self.sdf_truncation = args.criteria["sdf_truncation"] #截断距离，0.1
        self.learning_rate = args.tracker_specs["learning_rate"] #学习率，0.01
        self.start_frame = args.tracker_specs["start_frame"] #开始帧，0
        self.end_frame = args.tracker_specs["end_frame"] #结束帧，-1
        self.show_imgs = args.tracker_specs["show_imgs"] #是否显示图片，False，为啥不显示？我可不可以把它改成True？
        self.step_size = args.tracker_specs["step_size"] #步长，0.1
        self.keyframe_freq = args.tracker_specs["keyframe_freq"] #关键帧频率，10
        self.max_voxel_hit = args.tracker_specs["max_voxel_hit"] #最大体素命中数，10
        self.max_distance = args.data_specs["max_depth"] #最大深度，10
        self.step_size = self.step_size * self.voxel_size #步长=步长乘以体素大小，0.02

        if self.end_frame <= 0:
            self.end_frame = len(self.data_stream)#如果结束帧小于等于0，则结束帧等于数据流的长度

        # sanity check on the lower/upper bounds
        self.start_frame = min(self.start_frame, len(self.data_stream))
        self.end_frame = min(self.end_frame, len(self.data_stream)) #开始帧和结束帧都不能超过数据流的长度

        # profiler（性能分析）
        verbose = get_property(args.debug_args, "verbose", False) #是否显示详细信息，False
        self.profiler = Profiler(verbose=verbose) #性能分析器
        self.profiler.enable()

    def process_first_frame(self, kf_buffer): #处理第一帧
        init_pose = self.data_stream.get_init_pose() #获取初始位姿
        fid, rgb, depth, K, _ = self.data_stream[self.start_frame] #获取第一帧的数据（id，rgb，depth，K，_）
        first_frame = RGBDFrame(fid, rgb, depth, K, init_pose) #创建第一帧
        first_frame.pose.requires_grad_(False) #第一帧的位姿不需要计算梯度
        first_frame.optim = torch.optim.Adam(first_frame.pose.parameters(), lr=1e-3) #第一帧的优化器

        print("******* initializing first_frame:", first_frame.stamp)
        kf_buffer.put(first_frame, block=True) #将第一帧放入关键帧缓冲区
        self.last_frame = first_frame #将第一帧作为上一帧
        self.start_frame += 1 #开始帧加1

    def spin(self, share_data, kf_buffer): #启动跟踪线程
        print("******* tracking process started! *******")
        progress_bar = tqdm(
            range(self.start_frame, self.end_frame), position=0)
        progress_bar.set_description("tracking frame")
        for frame_id in progress_bar:
            if share_data.stop_tracking: #如果share_data中的stop_tracking为True，则跳出循环
                break
            try:
                data_in = self.data_stream[frame_id] #从文件中读取帧
    
                if self.show_imgs:
                    import cv2
                    img = data_in[1]
                    depth = data_in[2]
                    cv2.imshow("img", img.cpu().numpy())
                    cv2.imshow("depth", depth.cpu().numpy())
                    cv2.waitKey(1)

                current_frame = RGBDFrame(*data_in) #创建当前帧
                self.do_tracking(share_data, current_frame, kf_buffer) #做一次do_tracking跟踪当前帧

                if self.render_freq > 0 and (frame_id + 1) % self.render_freq == 0: #如果到了渲染debug图片的时机
                    self.render_debug_images(share_data, current_frame) #渲染图片，用于debug，不是必须的
            except Exception as e:
                        print("error in dataloading: ", e,
                            f"skipping frame {frame_id}")
                            

        share_data.stop_mapping = True
        print("******* tracking process died *******")

    def check_keyframe(self, check_frame, kf_buffer):
        try:
            kf_buffer.put(check_frame, block=True) #如果当前帧为关键帧，则将当前帧放入关键帧缓冲区
        except:
            pass

    def do_tracking(self, share_data, current_frame, kf_buffer): #跟踪当前帧
        decoder = share_data.decoder.cuda()
        map_states = share_data.states #从share_data中获取decoder和states
        for k, v in map_states.items():
            map_states[k] = v.cuda() #加载map_states到gpu上

        self.profiler.tick("track frame")
        # 跟踪当前帧，计算采样射线，渲染，算loss，backward(),返回优化后的位姿，优化器和命中掩码mask
        frame_pose, optim, hit_mask = track_frame(
            self.last_frame.pose,
            current_frame,
            map_states,
            decoder,
            self.loss_criteria,
            self.voxel_size,
            self.N_rays,
            self.step_size,
            self.num_iterations,
            self.sdf_truncation,
            self.learning_rate,
            self.max_voxel_hit,
            self.max_distance,
            profiler=self.profiler,
            depth_variance=True
        )
        self.profiler.tok("track frame") #性能分析

        current_frame.pose = frame_pose #将优化后的位姿放入当前帧
        current_frame.optim = optim #将优化器和位姿放入当前帧
        current_frame.hit_ratio = hit_mask.sum() / self.N_rays #计算采样射线命中率
        self.last_frame = current_frame #将当前帧作为上一帧

        self.profiler.tick("transport frame")
        self.check_keyframe(current_frame, kf_buffer) #如果当前帧为关键帧，则将当前帧放入关键帧缓冲区
        self.profiler.tok("transport frame")

        share_data.push_pose(frame_pose.translation().detach().cpu().numpy()) #将优化后的位姿放入share_data中

    @torch.no_grad()
    def render_debug_images(self, share_data, current_frame): #渲染图片，用于debug，不是必须的，可以不看，我也没看，哈
        rgb = current_frame.rgb
        depth = current_frame.depth #获取当前帧的rgb和depth
        rotation = current_frame.get_rotation() #获取当前帧的旋转矩阵
        ind = current_frame.stamp #获取当前帧的id
        w, h = self.render_res #获取渲染分辨率
        final_outputs = dict()

        decoder = share_data.decoder.cuda() #从share_data中获取decoder
        map_states = share_data.states #从share_data中获取地图states
        for k, v in map_states.items():
            map_states[k] = v.cuda()

        rays_d = current_frame.get_rays(w, h).cuda() #计算当前帧的采样射线的方向
        rays_d = rays_d @ rotation.transpose(-1, -2) #将采样射线旋转到世界坐标系下，因为采样射线是相机坐标系下的，而地图是世界坐标系下的

        rays_o = current_frame.get_translation()
        rays_o = rays_o.unsqueeze(0).expand_as(rays_d) #将采样射线的原点扩展成和采样射线方向一样的形状

        rays_o = rays_o.reshape(1, -1, 3).contiguous()
        rays_d = rays_d.reshape(1, -1, 3) #将采样射线的原点和方向reshape成(1, -1, 3)的形状

        final_outputs = render_rays( #渲染图片
            rays_o,
            rays_d,
            map_states,
            decoder,
            self.step_size,
            self.voxel_size,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            chunk_size=20000,
            return_raw=True
        )

        rdepth = fill_in((h, w, 1),
                         final_outputs["ray_mask"].view(h, w),
                         final_outputs["depth"], 0) #将深度图填充到渲染分辨率的大小
        rcolor = fill_in((h, w, 3),
                         final_outputs["ray_mask"].view(h, w),
                         final_outputs["color"], 0) #将颜色图填充到渲染分辨率的大小
        # self.logger.log_raw_image(ind, rcolor, rdepth)

        # raw_surface=fill_in((h, w, 1),
        #                  final_outputs["ray_mask"].view(h, w),
        #                  final_outputs["raw"], 0)
        # self.logger.log_data(ind, raw_surface, "raw_surface")
        self.logger.log_images(ind, rgb, depth, rcolor, rdepth) #将rgb，depth，rcolor，rdepth保存到logger中
