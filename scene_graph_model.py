import torch.nn as nn
import torch.nn.functional as F
import torch
from roi import TorchROIPooling
from torch_geometric.nn import GCNConv, NNConv, GATConv
from torch_geometric.data import Batch, Data
from baseline_model import BaselineModel

device = "cuda"

class SceneGraphModel(torch.nn.Module):
    def __init__(self):
        super(SceneGraphModel, self).__init__()

        # Convolution model
        self.conv_model = BaselineModel()
        
        # ROI model
        self.roi_model = TorchROIPooling(output_size=2).to(device)

        # GNN
        self.conv1 = GATConv(in_channels=512, out_channels=256)
        self.conv2 = GATConv(in_channels=256, out_channels=128)
        self.conv3 = GATConv(in_channels=128, out_channels=64)
        self.conv4 = GATConv(in_channels=64, out_channels=16)
        self.conv5 = GATConv(in_channels=16, out_channels=8)

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
    
    def build_graph(self, x, proposals):
        b, n_objects, _ = x.shape

        # compute 3d pose of each object
        pose = torch.zeros((b, n_objects, 3)).to(device=device)
        for i in range(n_objects):
            px = (proposals[: ,i ,0] + proposals[: ,i ,3]) / 2
            py = (proposals[: ,i ,1] + proposals[: ,i ,4]) / 2
            pz = (proposals[: ,i ,2] + proposals[: ,i ,5]) / 2
            pose[:,i] = torch.vstack([px, py, pz]).T
        
        # contruct graph
        edge_index = torch.zeros((b, 2, n_objects * n_objects)).type(torch.LongTensor).to(device=device)
        edge_attr = torch.zeros((b, n_objects * n_objects)).to(device=device)
        idx = 0
        for i in range(n_objects):
            for j in range(n_objects):
                if i == j:
                    continue
                # an directed edge from i -> j
                edge_index[:, 0, idx] = i
                edge_index[:, 1, idx] = j
                # attribute is the distance between the two objects
                edge_attr[:, idx] = torch.norm(pose[:, i] - pose[:, j], dim=1)
                idx += 1

        # convert to batch
        l = []
        for i in range(b):
            l.append(Data(x=x[i], edge_index=edge_index[i], edge_attr=edge_attr[i]))
        return Batch.from_data_list(l)

    def forward(self, x, proposals):
        # 3d convolution baseline
        conv_x = self.conv_model(x, proposals)

        # get roi feature of each objects
        roi_x = self.roi_model.conv_layers(x)
        roi_x = self.roi_model._roi_pool(roi_x, proposals)
        b, n_objects, _, _, _, _ = roi_x.shape
        roi_x = roi_x.view(b, n_objects, -1)
        
        # build scene graph
        batch = self.build_graph(roi_x, proposals)

        # gnn layers
        gnn_x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        gnn_x = self.conv2(gnn_x, batch.edge_index)
        gnn_x = self.conv3(gnn_x, batch.edge_index)
        gnn_x = self.conv4(gnn_x, batch.edge_index)
        gnn_x = self.conv5(gnn_x, batch.edge_index)
        gnn_x = gnn_x.view(b, -1)

        assert(conv_x.shape[1] == gnn_x.shape[1])

        # fusion fc layers
        x = torch.hstack((conv_x, gnn_x))
        x = self.fc_layers(x)

        # normalized output
        x = F.normalize(x, p=2, dim=1)

        return x

    # def forward(self, x, proposals): 
    #     conv_x = self.conv_model(x, proposals)

    #     x = self.roi_model.conv_layers(x)
    #     x = self.roi_model._roi_pool(x, proposals)

    #     b, n_objects, c, _, _, _ = x.shape
    #     x = x.view(b, n_objects, -1)
        
    #     batch = self.build_graph(x, proposals)
    #     y = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
    #     y = self.conv2(y, batch.edge_index)

    #     conv_x = conv_x.view((-1, conv_x.shape[1], 1))
    #     flg_global = torch.ones(conv_x.shape).to(device)

    #     y = y.view((-1,y.shape[1], 1))
    #     flg_local = torch.zeros(y.shape).to(device)

    #     conv_x = torch.concat([conv_x, flg_global], dim=-1)
    #     y = torch.concat([y, flg_local], dim=-1)

    #     conv_x = conv_x.reshape(b, -1)
    #     y = y.reshape(b, -1)


    #     # print(y.shape)
    #     y = y.view(b, -1)
    #     x = torch.hstack((conv_x, y))
    #     x = self.fc_layers(x)
    #     x = F.normalize(x, p=2, dim=1)

    #     return x

   