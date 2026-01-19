import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.0):
        super(GNNModel, self).__init__()
        self.dropout_ratio = dropout

        # 1. 输入投影层
        # ESOL原子特征很少(9维)，先映射到高维空间(64维)再卷积，效果通常更好
        self.input_proj = Linear(num_node_features, hidden_channels)

        # 2. GIN 卷积层堆叠
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for _ in range(num_layers):
            # GIN 的核心：每个节点都要过一个 MLP (全连接网络)
            # 这赋予了它极强的非线性拟合能力
            mlp = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )

            # train_eps=True 让模型自动学习自身特征的权重
            self.convs.append(GINConv(mlp, train_eps=True))
            # 这里的 BatchNorm 是加在卷积输出后的
            self.bns.append(BatchNorm1d(hidden_channels))

        # 3. 回归输出层
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, data):
        # 必须转 float，否则报错 mat1 and mat2 dtype mismatch
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # 步骤A: 输入投影
        x = self.input_proj(x)
        x = F.relu(x)

        # 步骤B: GIN 卷积 + BN + ReLU
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # 步骤C: Readout (GIN 理论上最适合 Sum Pooling)
        x = global_add_pool(x, batch)

        # 步骤D: MLP 输出预测值
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)

        return x