import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.rgb_weight = args.criteria["rgb_weight"]
        self.depth_weight = args.criteria["depth_weight"]
        self.sdf_weight = args.criteria["sdf_weight"]
        self.fs_weight = args.criteria["fs_weight"]
        self.truncation = args.criteria["sdf_truncation"]
        self.max_dpeth = args.data_specs["max_depth"]

    def forward(self, outputs, obs, use_color_loss=True,
                use_depth_loss=True, compute_sdf_loss=True,
                weight_depth_loss=False):
                
        img, depth = obs
        loss = 0
        loss_dict = {}

        pred_depth = outputs["depth"]
        pred_color = outputs["color"]
        pred_sdf = outputs["sdf"]
        z_vals = outputs["z_vals"]
        ray_mask = outputs["ray_mask"]
        weights = outputs["weights"]

        gt_depth = depth[ray_mask]
        gt_color = img[ray_mask]

        # color_loss = self.compute_loss(
        #     gt_color, pred_color, loss_type='l1')
        if use_color_loss:
            color_loss = (gt_color - pred_color).abs().mean()
            loss += self.rgb_weight * color_loss
            loss_dict["color_loss"] = color_loss.item()

        if use_depth_loss:
            valid_depth = (gt_depth > 0.01) & (gt_depth < self.max_dpeth)
            depth_loss = (gt_depth - pred_depth).abs()

            if weight_depth_loss:
                depth_var = weights*((pred_depth.unsqueeze(-1) - z_vals)**2)
                depth_var = torch.sum(depth_var, -1)
                tmp = depth_loss/torch.sqrt(depth_var+1e-10)
                valid_depth = (tmp < 10*tmp.median()) & valid_depth
            depth_loss = depth_loss[valid_depth].mean()
            loss += self.depth_weight * depth_loss
            loss_dict["depth_loss"] = depth_loss.item()

        if compute_sdf_loss:
            fs_loss, sdf_loss = self.get_sdf_loss(
                z_vals, gt_depth, pred_sdf,
                truncation=self.truncation,
                loss_type='l2'
            )
            loss += self.fs_weight * fs_loss
            loss += self.sdf_weight * sdf_loss
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

    def get_masks(self, z_vals, depth, epsilon):

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
        depth_mask = torch.where(
            (depth > 0.0) & (depth < self.max_dpeth), torch.ones_like(
                depth), torch.zeros_like(depth)
        )
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        num_fs_samples = torch.count_nonzero(front_mask).float()
        num_sdf_samples = torch.count_nonzero(sdf_mask).float()
        num_samples = num_sdf_samples + num_fs_samples
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, sdf_mask, fs_weight, sdf_weight

    def get_sdf_loss(self, z_vals, depth, predicted_sdf, truncation, loss_type="l2"):

        front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(
            z_vals, depth.unsqueeze(-1).expand(*z_vals.shape), truncation
        )
        fs_loss = (self.compute_loss(predicted_sdf * front_mask, torch.ones_like(
            predicted_sdf) * front_mask, loss_type=loss_type,) * fs_weight)
        sdf_loss = (self.compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask,
                    depth.unsqueeze(-1).expand(*z_vals.shape) * sdf_mask, loss_type=loss_type,) * sdf_weight)
        # back_loss = (self.compute_loss(predicted_sdf * back_mask, -torch.ones_like(
        #     predicted_sdf) * back_mask, loss_type=loss_type,) * back_weight)

        return fs_loss, sdf_loss
