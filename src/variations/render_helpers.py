from copy import deepcopy
import torch
import torch.nn.functional as F

from .voxel_helpers import ray_intersect, ray_sample


def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)

    point_feats = (weights * point_feats).sum(1)
    return point_feats


def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
            point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel


@torch.enable_grad()
def get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size):
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / voxel_size + 0.5).unsqueeze(1)
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5
    feats = trilinear_interp(p, q, point_feats).float()
    # if self.args.local_coord:
    # feats = torch.cat([(p-.5).squeeze(1).float(), feats], dim=-1)
    return feats


@torch.enable_grad()
def get_features(samples, map_states, voxel_size):
    # encoder states
    point_feats = map_states["voxel_vertex_idx"].cuda()
    point_xyz = map_states["voxel_center_xyz"].cuda()
    values = map_states["voxel_vertex_emb"].cuda()

    # ray point samples
    sampled_idx = samples["sampled_point_voxel_idx"].long()
    sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)
    sampled_dis = samples["sampled_point_distance"]

    point_xyz = F.embedding(sampled_idx, point_xyz)
    point_feats = F.embedding(F.embedding(
        sampled_idx, point_feats), values).view(point_xyz.size(0), -1)
    feats = get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size)
    inputs = {"dists": sampled_dis, "emb": feats}
    return inputs


@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]

    chunk_size = 32
    res = bits  # -1

    @torch.no_grad()
    def get_scores_once(feats, points, values):
        # sample points inside voxels
        start = -.5
        end = .5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        xx, yy, zz = torch.meshgrid(x, y, z)
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features(
            {
                "sampled_point_xyz": sampled_xyz,
                "sampled_point_voxel_idx": sampled_idx,
                "sampled_point_ray_direction": None,
                "sampled_point_distance": None,
            },
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points,
                "voxel_vertex_emb": values,
            },
            voxel_size
        )

        # evaluation with density
        sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
        return sdf_values.reshape(-1, res ** 3, 4).detach().cpu()

    return torch.cat([
        get_scores_once(feats[i: i + chunk_size],
                        points[i: i + chunk_size], values)
        for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


@torch.no_grad()
def eval_points(sdf_network, map_states, sampled_xyz, sampled_idx, voxel_size):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]

    # sampled_xyz = sampled_xyz.reshape(1, 3) + points.unsqueeze(1)
    # sampled_idx = sampled_idx[None, :].expand(*sampled_xyz.size()[:2])
    sampled_idx = sampled_idx.reshape(-1)
    sampled_xyz = sampled_xyz.reshape(-1, 3)

    if sampled_xyz.shape[0] == 0:
        return

    field_inputs = get_features(
        {
            "sampled_point_xyz": sampled_xyz,
            "sampled_point_voxel_idx": sampled_idx,
            "sampled_point_ray_direction": None,
            "sampled_point_distance": None,
        },
        {
            "voxel_vertex_idx": feats,
            "voxel_center_xyz": points,
            "voxel_vertex_emb": values,
        },
        voxel_size
    )

    # evaluation with density
    sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
    return sdf_values.reshape(-1, 4)[:, :3].detach().cpu()

    # return torch.cat([
    #     get_scores_once(feats[i: i + chunk_size],
    #                     points[i: i + chunk_size], values)
    #     for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


def render_rays( #渲染射线
        rays_o, #射线起点
        rays_d, #射线方向
        map_states, #地图
        sdf_network, #decoder
        step_size, #步长
        voxel_size, #体素大小
        truncation, #截断距离
        max_voxel_hit, #最大体素命中数
        max_distance, #最大距离
        chunk_size=20000, #chunk大小
        profiler=None, #性能分析器
        return_raw=False #是否返回原始z值
):
    centres = map_states["voxel_center_xyz"] #体素中心
    childrens = map_states["voxel_structure"] #体素结构

    if profiler is not None:
        profiler.tick("ray_intersect")
    intersections, hits = ray_intersect(
        rays_o, rays_d, centres,
        childrens, voxel_size, max_voxel_hit, max_distance) #计算射线与体素的交点
    if profiler is not None:
        profiler.tok("ray_intersect")
    assert(hits.sum() > 0)

    ray_mask = hits.view(1, -1)
    intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items() #过滤掉不在视野内的无效交点
    }

    rays_o = rays_o[ray_mask].reshape(-1, 3)
    rays_d = rays_d[ray_mask].reshape(-1, 3) #过滤掉不在视野内的无效射线

    if profiler is not None:
        profiler.tick("ray_sample")
    samples = ray_sample(intersections, step_size=step_size) #采样射线
    if profiler is not None:
        profiler.tok("ray_sample")

    sampled_depth = samples['sampled_point_depth']
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits
    # 只有当射线命中时才计算
    sample_mask = sampled_idx.ne(-1)
    if sample_mask.sum() == 0:  # miss everything ski
        return None, 0

    sampled_xyz = ray(rays_o.unsqueeze(
        1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2)) # 计算采样点的坐标
    sampled_dir = rays_d.unsqueeze(1).expand(
        *sampled_depth.size(), rays_d.size()[-1]) #计算采样点的射线方向
    sampled_dir = sampled_dir / \
        (torch.norm(sampled_dir, 2, -1, keepdim=True) + 1e-8) #归一化
    samples['sampled_point_xyz'] = sampled_xyz
    samples['sampled_point_ray_direction'] = sampled_dir

    # apply mask
    samples_valid = {name: s[sample_mask] for name, s in samples.items()} #过滤掉不在视野内的无效采样点

    num_points = samples_valid['sampled_point_depth'].shape[0] #采样点的个数
    field_outputs = []
    if chunk_size < 0:
        chunk_size = num_points #chunk大小为采样点的个数

    for i in range(0, num_points, chunk_size):
        chunk_samples = {name: s[i:i+chunk_size]
                         for name, s in samples_valid.items()}

        # get encoder features as inputs
        # 获取编码器的特征作为输入
        if profiler is not None:
            profiler.tick("get_features")
        chunk_inputs = get_features(chunk_samples, map_states, voxel_size) #获取采样点的特征，作为输入
        if profiler is not None:
            profiler.tok("get_features")

        # forward implicit fields
        # 前向隐式场
        if profiler is not None:
            profiler.tick("render_core")
        chunk_outputs = sdf_network(chunk_inputs) #送入神经网络，输出sdf值和颜色，mask等信息
        if profiler is not None:
            profiler.tok("render_core")

        field_outputs.append(chunk_outputs) #将输出放入列表

    field_outputs = {name: torch.cat(
        [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}

    outputs = {'sample_mask': sample_mask}

    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1) #过滤掉不在视野内的无效采样点，获得sdf值
    colour = masked_scatter(sample_mask, field_outputs['color']) #过滤掉不在视野内的无效采样点
    # colour = torch.sigmoid(colour)
    sample_mask = outputs['sample_mask']

    valid_mask = torch.where(
        sample_mask, torch.ones_like(
            sample_mask), torch.zeros_like(sample_mask)
    )

    # convert sdf to weight
    def sdf2weights(sdf_in, trunc):
        weights = torch.sigmoid(sdf_in / trunc) * \
            torch.sigmoid(-sdf_in / trunc)   #论文里的公式(2),算权重wi

        signs = sdf_in[:, 1:] * sdf_in[:, :-1]
        mask = torch.where(
            signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs)
        )
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)
        mask = torch.where(
            z_vals < z_min + trunc,
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        weights = weights * mask * valid_mask
        # weights = weights * valid_mask
        return weights / (torch.sum(weights, dim=-1, keepdims=True) + 1e-8), z_min #论文里的公式(3),归一化

    z_vals = samples["sampled_point_depth"]

    weights, z_min = sdf2weights(sdf, truncation)
    rgb = torch.sum(weights[..., None] * colour, dim=-2)
    depth = torch.sum(weights * z_vals, dim=-1)

    return {
        "weights": weights, #权重
        "color": rgb, #颜色
        "depth": depth, #深度
        "z_vals": z_vals, #z值
        "sdf": sdf, #sdf值
        "weights": weights, #权重
        "ray_mask": ray_mask, #射线mask
        "raw": z_min if return_raw else None
    } #返回权重，颜色，深度，z值，sdf值，采样mask，原始z值


def bundle_adjust_frames(
    keyframe_graph, #优化目标：关键帧
    map_states,
    sdf_network, #decoder
    loss_criteria, #loss
    voxel_size,
    step_size,
    N_rays=512, #采样射线的个数
    num_iterations=10, #迭代次数10
    truncation=0.1,
    max_voxel_hit=10,
    max_distance=10,
    learning_rate=[1e-2, 5e-3],
    embed_optim=None,  #torch.optim.Adam([self.embeddings], lr=5e-3)  #将embeddings放入Adam
    model_optim=None, #torch.optim.Adam(self.decoder.parameters(), lr=5e-3)
    update_pose=True, #True
):

    #确定优化器optimizers需要优化的参数

    # optimize_params = [{'params': embeddings, 'lr': learning_rate[0]}]
    optimizers = [embed_optim] #torch.optim.Adam([self.embeddings], lr=5e-3) #将embeddings放入Adam
    if model_optim is not None:
        # optimize_params += [{'params': sdf_network.parameters(),
        #                      'lr': learning_rate[0]}]
        optimizers += [model_optim] #torch.optim.Adam(self.decoder.parameters(), lr=5e-3)

    # optimize_params=[]
    for keyframe in keyframe_graph: #遍历关键帧
        if keyframe.stamp != 0 and update_pose:  #如果不是第一帧并且需要更新位姿
            optimizers += [keyframe.optim] #将关键帧的优化器放入优化器列表
            # keyframe.pose.requires_grad_(True)
            # optimize_params += [{
            #     'params': keyframe.pose.parameters(), 'lr': learning_rate[1]
            # }]
    
    # if len(optimize_params) != 0:
    #     pose_optim = torch.optim.Adam(optimize_params)
    #     optimizers += [pose_optim]

    for _ in range(num_iterations): #迭代10次

        rays_o = []
        rays_d = []
        rgb_samples = []
        depth_samples = []

        for frame in keyframe_graph: #遍历关键帧
            pose = frame.get_pose().cuda() #获取位姿
            frame.sample_rays(N_rays) #采样射线

            sample_mask = frame.sample_mask.cuda() #采样结果的mask
            sampled_rays_d = frame.rays_d[sample_mask].cuda() #采样射线的方向（过滤掉不在视野内的无效射线）

            R = pose[: 3, : 3].transpose(-1, -2) #旋转矩阵
            sampled_rays_d = sampled_rays_d@R #旋转射线方向
            sampled_rays_o = pose[: 3, 3].reshape( #平移射线起点
                1, -1).expand_as(sampled_rays_d)

            rays_d += [sampled_rays_d]
            rays_o += [sampled_rays_o]
            rgb_samples += [frame.rgb.cuda()[sample_mask]] #采样的rgb信息
            depth_samples += [frame.depth.cuda()[sample_mask]] #采样的深度信息

        rays_d = torch.cat(rays_d, dim=0).unsqueeze(0) #将所有采样的射线方向拼接起来
        rays_o = torch.cat(rays_o, dim=0).unsqueeze(0) #将所有采样的射线起点拼接起来
        rgb_samples = torch.cat(rgb_samples, dim=0).unsqueeze(0)
        depth_samples = torch.cat(depth_samples, dim=0).unsqueeze(0)

        final_outputs = render_rays(
            rays_o,
            rays_d,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            # chunk_size=-1
        ) #渲染射线

        loss, _ = loss_criteria( #计算loss
            final_outputs, (rgb_samples, depth_samples))

        for optim in optimizers: #遍历优化器
            optim.zero_grad() #梯度清零
        loss.backward() #反向传播
        for optim in optimizers:
            optim.step() #优化器更新参数


def track_frame( #跟踪帧
    frame_pose, #帧的位姿
    curr_frame, #当前帧
    map_states, #地图
    sdf_network, #decoder
    loss_criteria, #loss
    voxel_size, #体素大小
    N_rays=512, #采样射线的个数
    step_size=0.05, #步长
    num_iterations=10, #跟踪迭代次数
    truncation=0.1, #截断距离
    learning_rate=1e-3, #学习率
    max_voxel_hit=10, #最大体素命中数
    max_distance=10, #最大距离
    profiler=None, #性能分析器,none
    depth_variance=False #是否包含深度方差
):

    init_pose = deepcopy(frame_pose).cuda() #深拷贝帧的位姿
    init_pose.requires_grad_(True) #需要梯度
    optim = torch.optim.Adam(init_pose.parameters(), lr=learning_rate) #将位姿参数放入Adam

    for iter in range(num_iterations): #迭代10次
        if iter == 0 and profiler is not None:
            profiler.tick("sample_rays")
        curr_frame.sample_rays(N_rays)
        if iter == 0 and profiler is not None:
            profiler.tok("sample_rays")

        sample_mask = curr_frame.sample_mask #采样结果的mask
        ray_dirs = curr_frame.rays_d[sample_mask].unsqueeze(0).cuda() #采样射线的方向（过滤掉不在视野内的无效射线）
        rgb = curr_frame.rgb[sample_mask].cuda()
        depth = curr_frame.depth[sample_mask].cuda() #得到采样的rgb和深度信息
        # 计算采样射线
        ray_dirs_iter = ray_dirs.squeeze( #计算采样射线的方向
            0) @ init_pose.rotation().transpose(-1, -2) #将采样射线旋转到世界坐标系下，因为采样射线是相机坐标系下的，而地图是世界坐标系下的
        ray_dirs_iter = ray_dirs_iter.unsqueeze(0)
        ray_start_iter = init_pose.translation().reshape(
            1, 1, -1).expand_as(ray_dirs_iter).cuda().contiguous() #将采样射线的原点扩展成和采样射线方向一样的形状

        if iter == 0 and profiler is not None:
            profiler.tick("render_rays")
        final_outputs = render_rays(  #根据采样射线来渲染，返回权重，颜色，深度，z值，sdf值，采样mask，原始z值
            ray_start_iter,
            ray_dirs_iter,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            # chunk_size=-1,
            profiler=profiler if iter == 0 else None
        )
        if iter == 0 and profiler is not None:
            profiler.tok("render_rays")

        hit_mask = final_outputs["ray_mask"].view(N_rays)
        final_outputs["ray_mask"] = hit_mask #将命中的射线mask放入final_outputs

        if iter == 0 and profiler is not None:
            profiler.tick("loss_criteria")
        loss, _ = loss_criteria(
            final_outputs, (rgb, depth), weight_depth_loss=depth_variance) #计算loss
        if iter == 0 and profiler is not None:
            profiler.tok("loss_criteria")

        if iter == 0 and profiler is not None:
            profiler.tick("backward step")
        optim.zero_grad() #梯度清零
        loss.backward() #反向传播
        optim.step() #优化器更新参数
        if iter == 0 and profiler is not None:
            profiler.tok("backward step")

    return init_pose, optim, hit_mask #返回优化后的位姿，优化器，命中的射线mask
