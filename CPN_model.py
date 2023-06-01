import torch.nn as nn
import torch.nn.functional as F
import torch
from roi import TorchROIPooling
from torch_geometric.nn import GCNConv, NNConv, GATConv
from torch_geometric.data import Batch, Data
from baseline_model import BaselineModel

device = "cuda"


class CPNModel(torch.nn.Module):
    def __init__(self):
        super(CPNModel, self).__init__()

        # Convolution model
        self.conv_model = BaselineModel()
        
        # ROI model
        self.roi_model = TorchROIPooling(output_size=2).to(device)

        # GNN
        # self.gnn_layers = nn.Sequential(
        self.conv1 = GATConv(in_channels=16, out_channels=12) #in_channels = 128/8 
        self.conv2 = GATConv(in_channels=12, out_channels=8)
        # )

        self.fc_layers_node_feature = nn.Sequential(
            # nn.Linear(8*8*3, 128),
            nn.Linear(8*3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # fully coneected layers for local instances
        self.fc_layers_local = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # fully connected fusion layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
        )

    
    def build_graph(self, x, edge_index, edge_attr):
        b, n_objects, _ = x.shape

        l = []
        for i in range(b):
            unpadded_edge_index = torch.masked_select(edge_index[i], edge_index[i, 0, :] != edge_index[i, 1,:]).to(device)

            unpadded_edge_index = unpadded_edge_index.view(2, -1).type(torch.long)
            unpadded_edge_attr = edge_attr[i, :unpadded_edge_index.shape[1], :].to(device)
            # print(f'edge_index shape:{unpadded_edge_index.shape}')
            # print(f'edge_attr shape:{unpadded_edge_attr.shape}')

            l.append(Data(x=x[i], edge_index=unpadded_edge_index, edge_attr=unpadded_edge_attr))

        return Batch.from_data_list(l)

    def forward(self, x, proposals, node_feature, edge_index, edge_attr):
        # print(f'x shape: {x.shape}')
        # print(f'cpn shape: {cpn.shape}')
        conv_x = self.conv_model(x, proposals)


        # x = self.roi_model.conv_layers(cpn)
        # x = self.roi_model._roi_pool(x, proposals)
        b = node_feature.shape[0]
        node_feature = node_feature.view(b * 8, 24) # 
        x = self.fc_layers_node_feature(node_feature)

        # b, n_objects, c, _, _, _ = x.shape
        x = x.view(b, 8, -1)
        batch = self.build_graph(x, edge_index, edge_attr)
        # print(batch)
        # print(f'edge index:\n{batch.edge_index}')
        # print(f'edge attr:\n{batch.edge_attr}')
        # print(f'batch:\n{batch.batch}')
        # print(f'ptr:\n{batch.ptr}')
        y = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        y = self.conv2(y, batch.edge_index)

        # conv_x = conv_x.view((-1, conv_x.shape[1], 1))
        # flg_global = torch.ones(conv_x.shape).to(device)

        # y = y.view((-1,y.shape[1], 1))
        # flg_local = torch.zeros(y.shape).to(device)

        # conv_x = torch.concat([conv_x, flg_global], dim=-1)
        # y = torch.concat([y, flg_local], dim=-1)

        # conv_x = conv_x.reshape(b, -1)
        # y = y.reshape(b, -1)


        y = y.view(b, -1)
        # y = self.fc_layers_local(y)

        # print(f'conv_x shape: {conv_x.shape}')
        # print(f'y shape: {y.shape}')
        assert(conv_x.shape[1] == y.shape[1])

        x = torch.hstack((conv_x, y))
        x = self.fc_layers(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    # def __init__(self):
    #     super(SceneGraphModel, self).__init__()

    #     # Define 3D convolutional layers
    #     self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    #     self.bn1 = nn.BatchNorm3d(16)
    #     self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
    #     self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    #     self.bn2 = nn.BatchNorm3d(16)
    #     self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
    #     # self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    #     # self.bn3 = nn.BatchNorm3d(128)
    #     # self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

    #     # Define fully connected layers
    #     self.fc1 = nn.Linear(15744, 1024)
    #     self.bn4 = nn.BatchNorm1d(1024)
    #     self.fc2 = nn.Linear(1024, 256)
    #     self.bn5 = nn.BatchNorm1d(256)
    #     self.fc3 = nn.Linear(256, 64)
    #     self.bn6 = nn.BatchNorm1d(64)
    #     self.fc4 = nn.Linear(64, 10)

    #     # ROI
    #     # self.roi = TorchROIAlign(output_size=3, scaling_factor=0.242)
    #     self.roi = TorchROIPooling(output_size=2, scaling_factor=0.242)

    #     # self.gconv1 = GCNConv(16 * self.roi.output_size**3, 16 * self.roi.output_size**3 * 2)
    #     # self.gconv2 = GCNConv(16 * self.roi.output_size**3 * 2, 16 * self.roi.output_size**3)
    #     nn1 = nn.Identity()
    #     # self.gconv1 = NNConv(in_channels=128, out_channels=100, nn=nn1)
    #     self.gconv1 = GATConv(in_channels=128, out_channels=64)

    #     self.gconv2 = GATConv(in_channels=64, out_channels=64)

    # def forward(self, x, proposals):
    #     # Pass input through 3D convolutional layers
    #     b, _, _, _, _ = x.shape
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = F.relu(x)
    #     x = self.pool1(x)
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = F.relu(x)
    #     x = self.pool2(x)
        
    #     roi = self.roi(x, proposals)
    #     num = proposals.shape[1]
        
    #     roi = roi.view(-1, num, self.roi.output_size*self.roi.output_size*self.roi.output_size*16) #3456

    #     edge_index = torch.zeros((b, 2, num * num)).type(torch.LongTensor).to(device=device)
    #     edge_attr = torch.zeros((b, num * num)).to(device=device)
    #     pose = torch.zeros((b, num, 3)).to(device=device)
    #     for i in range(num):
    #         px = (proposals[: ,i ,0] + proposals[: ,i ,3]) / 2
    #         py = (proposals[: ,i ,1] + proposals[: ,i ,4]) / 2
    #         pz = (proposals[: ,i ,2] + proposals[: ,i ,5]) / 2
    #         pose[:,i] = torch.vstack([px, py, pz]).T
        
    #     idx = 0
    #     for i in range(num):
    #         for j in range(num):
    #             if i == j:
    #                 continue
    #             edge_index[:, 0, idx] = i
    #             edge_index[:, 1, idx] = j
    #             edge_attr[:, idx] = torch.norm(pose[:, i] - pose[:, j], dim=1)
    #             idx += 1

    #     l = []
    #     for i in range(b):
    #         l.append(Data(x=roi[i], edge_index=edge_index[i], edge_attr=edge_attr[i]))

    #     batch = Batch.from_data_list(l)
    #     # print(roi.shape)
    #     # print(batch.x.shape)
    #     # print(batch.edge_index.shape)
    #     # print(batch.edge_attr.shape)
    #     y = self.gconv1(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr).relu()
    #     # y = F.dropout(y, p=0.2)
    #     y = self.gconv2(x=y, edge_index=batch.edge_index)
    #     # y = F.dropout(y, p=0.2)
    #     # print(y.shape)
    #     y = y.view(b, -1)
    #     # print(y.shape)
    #     # print(x.shape)
    #     # print(y.shape)
    #     x = x.view(-1, 16 * 8 * 7 * 17)
    #     x = torch.concat((x, y), dim=1)
    #     # print(x.shape)

    #     x = self.fc1(x)
    #     x = self.bn4(x)
    #     x = F.relu(x)

    #     x = self.fc2(x)
    #     x = self.bn5(x)
    #     x = F.relu(x)
    #     x = self.fc3(x)
    #     x = self.bn6(x)
    #     x = F.relu(x)
    #     x = self.fc4(x)
    #     x = F.normalize(x, p=2, dim=1)  # L2-normalize output embeddings

    #     return x
