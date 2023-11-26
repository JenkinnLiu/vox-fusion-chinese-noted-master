import torch
import torch.nn as nn
import torch.nn.functional as F

# 这里的代码是基于Nerf的实现进行修改的，主要是修改了embedding的方式，以及输出的方式，NICE-SLAM也有类似的实现
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.
    基于高斯傅里叶特征映射的实现进行修改。“傅里叶特征让网络在低维域中学习高频函数”：
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.embedding_size = mapping_size

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.
    Nerf位置嵌入。
    """

    def __init__(self, in_dim, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos] #正弦余弦函数sin和cos
        self.max_freq_log2 = multires-1 #最大频率
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs
        self.embedding_size = multires*in_dim*2 + in_dim

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs) #这里的频率是指傅里叶变换中的频率，即sin和cos中的参数，这里的频率是指2的幂次方
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq)) #sin(x*2^i)和cos(x*2^i),其中，i为0,1,2,...,self.max_freq
        ret = torch.cat(output, dim=1)
        return ret


class Same(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embedding_size = in_dim

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self,
                 depth=8,
                 width=256,
                 in_dim=3,
                 sdf_dim=128,
                 skips=[4],
                 multires=6,
                 embedder='nerf',
                 local_coord=False,
                 **kwargs):
        """
        """
        super().__init__()
        self.D = depth
        self.W = width
        self.skips = skips
        if embedder == 'nerf':
            self.pe = Nerf_positional_embedding(in_dim, multires) #嵌入方式
        elif embedder == 'none':
            self.pe = Same(in_dim)
        elif embedder == 'gaussian':
            self.pe = GaussianFourierFeatureTransform(in_dim)
        else:
            raise NotImplementedError("unknown positional encoder")

        self.pts_linears = nn.ModuleList( #这里就是论文里的网络结构，输入是嵌入后的点，输出是sdf和rgb
            [nn.Linear(self.pe.embedding_size, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + self.pe.embedding_size, width) for i in range(depth-1)])
        self.sdf_out = nn.Linear(width, 1+sdf_dim)
        self.color_out = nn.Sequential(
            nn.Linear(sdf_dim+self.pe.embedding_size, width),
            nn.ReLU(),
            nn.Linear(width, 3),
            nn.Sigmoid())
        # self.output_linear = nn.Linear(width, 4)

    def get_values(self, x):
        x = self.pe(x)
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # outputs = self.output_linear(h)
        # outputs[:, :3] = torch.sigmoid(outputs[:, :3])
        sdf_out = self.sdf_out(h)
        sdf = sdf_out[:, :1]
        sdf_feat = sdf_out[:, 1:]

        h = torch.cat([sdf_feat, x], dim=-1)
        rgb = self.color_out(h)
        outputs = torch.cat([rgb, sdf], dim=-1)

        return outputs  #输出sdf, rgb

    def get_sdf(self, inputs):
        return self.get_values(inputs['emb'])[:, 3]

    def forward(self, inputs):
        outputs = self.get_values(inputs['emb'])

        return {
            'color': outputs[:, :3],
            'sdf': outputs[:, 3]
        }
