import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.rgb_weight = args.criteria["rgb_weight"] # 5
        self.depth_weight = args.criteria["depth_weight"] # 1
        self.sdf_weight = args.criteria["sdf_weight"] # 5000
        self.fs_weight = args.criteria["fs_weight"] # 10
        self.truncation = args.criteria["sdf_truncation"] # 0.1
        self.max_dpeth = args.data_specs["max_depth"]  # 10

    def forward(self, outputs, obs, use_color_loss=True,
                use_depth_loss=True, compute_sdf_loss=True,
                weight_depth_loss=False):
                
        img, depth = obs
        loss = 0
        loss_dict = {}

        pred_depth = outputs["depth"] #预测的深度
        pred_color = outputs["color"] #预测的颜色
        pred_sdf = outputs["sdf"] #预测的sdf
        z_vals = outputs["z_vals"] #预测的z值
        ray_mask = outputs["ray_mask"] #采样射线掩码mask
        weights = outputs["weights"] # 权重

        gt_depth = depth[ray_mask] #真实的深度
        gt_color = img[ray_mask] #真实的颜色

        # color_loss = self.compute_loss(
        #     gt_color, pred_color, loss_type='l1')
        if use_color_loss:
            color_loss = (gt_color - pred_color).abs().mean()
            loss += self.rgb_weight * color_loss #计算颜色损失
            loss_dict["color_loss"] = color_loss.item()

        if use_depth_loss:
            valid_depth = (gt_depth > 0.01) & (gt_depth < self.max_dpeth)
            depth_loss = (gt_depth - pred_depth).abs() #计算深度损失

            if weight_depth_loss: #计算带权重的深度损失，因为存在方差，所以这样可以减小方差较大的点的权重
                depth_var = weights*((pred_depth.unsqueeze(-1) - z_vals)**2)
                depth_var = torch.sum(depth_var, -1)
                tmp = depth_loss/torch.sqrt(depth_var+1e-10)
                valid_depth = (tmp < 10*tmp.median()) & valid_depth
            depth_loss = depth_loss[valid_depth].mean()
            loss += self.depth_weight * depth_loss
            loss_dict["depth_loss"] = depth_loss.item()

        if compute_sdf_loss: #计算sdf损失
            fs_loss, sdf_loss = self.get_sdf_loss(
                z_vals, gt_depth, pred_sdf,
                truncation=self.truncation,
                loss_type='l2'
            ) #计算free-space loss 和 sdf loss
            loss += self.fs_weight * fs_loss
            loss += self.sdf_weight * sdf_loss #加上权重
            # loss += self.bs_weight * back_loss
            loss_dict["fs_loss"] = fs_loss.item()
            # loss_dict["bs_loss"] = back_loss.item()
            loss_dict["sdf_loss"] = sdf_loss.item()

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def compute_loss(self, x, y, mask=None, loss_type="l2"):
        if mask is None:
            mask = torch.ones_like(x).bool()
        if loss_type == "l1":
            return torch.mean(torch.abs(x - y)[mask])
        elif loss_type == "l2":
            return torch.mean(torch.square(x - y)[mask])

    def get_masks(self, z_vals, depth, epsilon): #计算mask

        #不能距离预测深度太远（超过epsilon），否则不计算损失，mask掉
        front_mask = torch.where(
            z_vals < (depth - epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        back_mask = torch.where(
            z_vals > (depth + epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        #深度不能为0，也不能太大，否则不计算损失，mask掉
        depth_mask = torch.where(
            (depth > 0.0) & (depth < self.max_dpeth), torch.ones_like(
                depth), torch.zeros_like(depth)
        )
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask  #计算sdf mask（同时满足三个mask条件才能计算，相当于算个交集）

        num_fs_samples = torch.count_nonzero(front_mask).float() #计算free-space sample的数量（即mask为1的数量）
        num_sdf_samples = torch.count_nonzero(sdf_mask).float() #计算sdf sample的数量（即mask为1的数量）
        num_samples = num_sdf_samples + num_fs_samples #计算总的sample数量
        fs_weight = 1.0 - num_fs_samples / num_samples #计算free-space sample的权重，即sample数量越多，权重越小
        sdf_weight = 1.0 - num_sdf_samples / num_samples #计算sdf sample的权重

        return front_mask, sdf_mask, fs_weight, sdf_weight #返回三个mask和权重

    def get_sdf_loss(self,
                     z_vals,
                     depth,
                     predicted_sdf,
                     truncation,
                     loss_type="l2"): #计算sdf损失

        front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks( #计算三种mask和权重
            z_vals, depth.unsqueeze(-1).expand(*z_vals.shape), truncation
        )
        fs_loss = (self.compute_loss(predicted_sdf * front_mask, torch.ones_like( #计算free-space loss
            predicted_sdf) * front_mask, loss_type=loss_type,) * fs_weight) #公式（6）
        sdf_loss = (self.compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask,
                    depth.unsqueeze(-1).expand(*z_vals.shape) * sdf_mask, loss_type=loss_type,) * sdf_weight) #计算sdf loss，公式（7）
        # back_loss = (self.compute_loss(predicted_sdf * back_mask, -torch.ones_like(
        #     predicted_sdf) * back_mask, loss_type=loss_type,) * back_weight)

        return fs_loss, sdf_loss
