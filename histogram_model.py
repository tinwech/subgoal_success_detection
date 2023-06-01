import torch.nn as nn
import torch.nn.functional as F
import torch
from roi import TorchROIPooling
from torch_geometric.nn import GCNConv, NNConv, GATConv
from torch_geometric.data import Batch, Data
from baseline_model import BaselineModel

device = "cuda"


class HistogramModel(torch.nn.Module):
    def __init__(self):
        super(HistogramModel, self).__init__()

        # Convolution model
        self.conv_model = BaselineModel()
        
        # ROI model
        self.roi_model = TorchROIPooling(output_size=2).to(device)

        # GNN
        # self.gnn_layers = nn.Sequential(
        self.conv1 = GATConv(in_channels=3, out_channels=12, heads=8) #in_channels = 128/8 
        self.conv2 = GATConv(in_channels=12*8, out_channels=8, heads=1)
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
        
        x = node_feature.view(b, 8, -1)
        batch = self.build_graph(x, edge_index, edge_attr)
        
        y = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        y = self.conv2(y, batch.edge_index)

        y = y.view(b, -1)
        
        assert(conv_x.shape[1] == y.shape[1])

        x = torch.hstack((conv_x, y))
        x = self.fc_layers(x)
        x = F.normalize(x, p=2, dim=1)

        return x
