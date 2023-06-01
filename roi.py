import torch
import torch.nn as nn

device = 'cuda'
    
class TorchROIPooling(torch.nn.Module):

    def __init__(self, output_size):
        super(TorchROIPooling, self).__init__()

        self.output_size = output_size
        self.scaling_factor = 0.242

        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )


    def _max_pool(self, features, scaled_proposal):
        
        num_channels, _, _, _ = features.shape

        interp_features = torch.zeros((num_channels, self.output_size, self.output_size, self.output_size))

        if torch.all(scaled_proposal == 0):
            return interp_features
        
        xp0, yp0, zp0, xp1, yp1, zp1 = scaled_proposal.type(torch.LongTensor)
        xp1 += 1
        yp1 += 1
        zp1 += 1

        size = (self.output_size, self.output_size, self.output_size)
        
        interp_features = torch._adaptive_avg_pool3d(features[:,xp0:xp1,yp0:yp1,zp0:zp1], size)
        
        return interp_features
    
    def _roi_pool(self, x, proposals):
        batch, num_channels, _, _, _ = x.shape

        # first scale proposals down by self.scaling factor
        scaled_proposals = torch.zeros_like(proposals)

        scaled_proposals[:,:,0] = torch.floor(proposals[:,:, 0] * self.scaling_factor)
        scaled_proposals[:,:,1] = torch.floor(proposals[:,:, 1] * self.scaling_factor)
        scaled_proposals[:,:,2] = torch.floor(proposals[:,:, 2] * self.scaling_factor)
        scaled_proposals[:,:,3] = torch.ceil(proposals[:,:, 3] * self.scaling_factor)
        scaled_proposals[:,:,4] = torch.ceil(proposals[:,:, 4] * self.scaling_factor)
        scaled_proposals[:,:,5] = torch.ceil(proposals[:,:, 5] * self.scaling_factor)


        res = torch.zeros((batch, proposals.shape[1], num_channels, self.output_size,
                        self.output_size, self.output_size)).to(device)
        
        for batch_id in range(batch):
            for idx in range(scaled_proposals.shape[1]):
                proposal = scaled_proposals[batch_id, idx,:]
                res[batch_id][idx] = self._max_pool(x[batch_id], proposal)

        return res

    def forward(self, x, proposals):
        x = self.conv_layers(x)
        x = self._roi_pool(x, proposals)
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        return x


# def bilinear_interpolate_2d(img, x, y):
#     """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
#     Taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

#     Args:
#         img (torch.Tensor): Tensor of size cxwxh. Usually one channel of feature layer
#         x (torch.Tensor): Float dtype, x axis location for sampling
#         y (torch.Tensor): Float dtype, y axis location for sampling

#     Returns:
#         torch.Tensor: interpolated value
#     """

#     img = torch.squeeze(img)
#     # print("img:", img.shape)

#     # img = img.reshape((img.shape[1], img.shape[2]))

#     # x0 = torch.floor(x).type(torch.LongTensor)
#     x0 = torch.floor(x).type(torch.cuda.LongTensor)
#     x1 = x0 + 1

#     y0 = torch.floor(y).type(torch.cuda.LongTensor)
#     # y0 = torch.floor(y).type(torch.LongTensor)
#     y1 = y0 + 1

#     x0 = torch.clamp(x0, 0, img.shape[1]-1)
#     x1 = torch.clamp(x1, 0, img.shape[1]-1)
#     y0 = torch.clamp(y0, 0, img.shape[2]-1)
#     y1 = torch.clamp(y1, 0, img.shape[2]-1)

#     Ia = torch.squeeze(img[:, x0, y0])
#     Ib = torch.squeeze(img[:, x0, y1])
#     Ic = torch.squeeze(img[:, x1, y0])
#     Id = torch.squeeze(img[:, x1, y1])

#     # print("Ia:", Ia.shape)

#     # norm_const = 1/((x1.type(floattype) - x0.type(floattype))*(y1.type(floattype) - y0.type(floattype)))

#     wa = (x1.type(floattype) - x) * (y1.type(floattype) - y)
#     wb = (x1.type(floattype) - x) * (y-y0.type(floattype))
#     wc = (x-x0.type(floattype)) * (y1.type(floattype) - y)
#     wd = (x-x0.type(floattype)) * (y - y0.type(floattype))

#     return torch.t(torch.t(Ia)*wa) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

# def bilinear_interpolate_3d(grid, x, y, z):
#     """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
#     Taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

#     Args:
#         grid (torch.Tensor): Tensor of size cxwxhxl. Usually one channel of feature layer
#         x (torch.Tensor): Float dtype, x axis location for sampling
#         y (torch.Tensor): Float dtype, y axis location for sampling
#         z (torch.Tensor): Float dtype, z axis location for sampling

#     Returns:
#         torch.Tensor: interpolated value
#     """
#     # print("grid:", grid.shape)
#     x0 = torch.floor(x).type(torch.cuda.LongTensor)
#     # x0 = torch.floor(x).type(torch.LongTensor)

#     x1 = x0 + 1

#     # print(x0.device)
#     # print(y.device)

#     x0 = torch.clamp(x0, 0, grid.shape[1]-1)
#     x1 = torch.clamp(x1, 0, grid.shape[1]-1)
    
#     Ia = bilinear_interpolate_2d(grid[:,x0,:,:], y, z)

#     if Ia.isnan().any():
#         exit()

#     Ib = bilinear_interpolate_2d(grid[:,x1,:,:], y, z)

#     # print(grid)
#     # print(Ia)

#     wa = x1.type(floattype) - x
#     wb = x - x0.type(floattype)

#     return torch.t(torch.t(Ia)*wa) + torch.t(torch.t(Ib)*wb)

# class TorchROIAlign(object):

#     def __init__(self, output_size, scaling_factor):
#         self.output_size = output_size
#         self.scaling_factor = scaling_factor

#     def _roi_align(self, features, scaled_proposal):
#         """Given feature layers and scaled proposals return bilinear interpolated
#         points in feature layer

#         Args:
#             features (torch.Tensor): Tensor of shape channels x width x height x length
#             scaled_proposal (list of torch.Tensor): Each tensor is a bbox by which we
#             will extract features from features Tensor
#         """
        
#         num_channels, _, _, _ = features.shape

#         # print("roi_align:", features.shape)

#         xp0, yp0, zp0, xp1, yp1, zp1 = scaled_proposal
#         p_width = xp1 - xp0
#         p_height = yp1 - yp0
#         p_length = zp1 - zp0

#         w_stride = p_width/self.output_size
#         h_stride = p_height/self.output_size
#         l_stride = p_length/self.output_size

#         interp_features = torch.zeros((num_channels, self.output_size, self.output_size, self.output_size))

#         for i in range(self.output_size):
#             for j in range(self.output_size):
#                 for k in range(self.output_size):
#                     x_bin_strt = i*w_stride + xp0
#                     y_bin_strt = j*h_stride + yp0
#                     z_bin_strt = k*l_stride + zp0

#                     # generate 4 points for interpolation
#                     # notice no rounding
#                     dev = 'cuda'
#                     x1 = torch.Tensor([x_bin_strt + 0.25*w_stride]).to(dev)
#                     y1 = torch.Tensor([y_bin_strt + 0.25*h_stride]).to(dev)
#                     z1 = torch.Tensor([z_bin_strt + 0.25*l_stride]).to(dev)
#                     x2 = torch.Tensor([x_bin_strt + 0.75*w_stride]).to(dev)
#                     y2 = torch.Tensor([y_bin_strt + 0.75*h_stride]).to(dev)
#                     z2 = torch.Tensor([z_bin_strt + 0.75*l_stride]).to(dev)

#                     # for c in range(num_channels):
#                     # img = features[0, c]
#                     grid = features
#                     v1 = bilinear_interpolate_3d(grid, x1, y1, z1)
#                     v2 = bilinear_interpolate_3d(grid, x1, y1, z2)
#                     v3 = bilinear_interpolate_3d(grid, x1, y2, z1)
#                     v4 = bilinear_interpolate_3d(grid, x1, y2, z2)
#                     v5 = bilinear_interpolate_3d(grid, x2, y1, z1)
#                     v6 = bilinear_interpolate_3d(grid, x2, y1, z2)
#                     v7 = bilinear_interpolate_3d(grid, x2, y2, z1)
#                     v8 = bilinear_interpolate_3d(grid, x2, y2, z2)

#                     # print(v1)
#                     # print(v1.shape)
#                     interp_features[:, i, j, k] = (v1+v2+v3+v4+v5+v6+v7+v8)/8
        
#         return interp_features

#     def __call__(self, feature_layer, proposals):
#         """Given feature layers and a list of proposals, it returns aligned
#         representations of the proposals. Proposals are scaled by scaling factor
#         before pooling.

#         Args:
#             feature_layer (torch.Tensor): Feature layer of size (batch, num_channels, width,
#             height, length)
#             proposals (list of torch.Tensor): Each element of the list represents a
#             bounding box as (x0, y0, z0, x1, y1, z1)

#         Returns:
#             torch.Tensor: Shape len(proposals), channels, self.output_size,
#             self.output_size
#         """
#         # print("feature_layer:", feature_layer.shape)
#         batch, num_channels, _, _, _ = feature_layer.shape

#         # first scale proposals down by self.scaling factor
#         scaled_proposals = torch.zeros_like(proposals)
#         # print("scaled_proposals:", scaled_proposals.shape)

#         # notice no ceil or floor functions
#         scaled_proposals[:,:,0] = proposals[:,:, 0] * self.scaling_factor
#         scaled_proposals[:,:,1] = proposals[:,:, 1] * self.scaling_factor
#         scaled_proposals[:,:,2] = proposals[:,:, 2] * self.scaling_factor
#         scaled_proposals[:,:,3] = proposals[:,:, 3] * self.scaling_factor
#         scaled_proposals[:,:,4] = proposals[:,:, 4] * self.scaling_factor
#         scaled_proposals[:,:,5] = proposals[:,:, 5] * self.scaling_factor


#         res = torch.zeros((batch, proposals.shape[1], num_channels, self.output_size,
#                         self.output_size, self.output_size)).to('cuda')
        
#         for batch_id in range(batch):
#             for idx in range(scaled_proposals.shape[1]):
#                 proposal = scaled_proposals[batch_id, idx,:]
#                 # print("proposal:", proposal.shape)
#                 res[batch_id][idx] = self._roi_align(feature_layer[batch_id], proposal)

#         return res