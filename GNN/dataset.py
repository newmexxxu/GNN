import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader


def get_dataloaders(root_dir='data', batch_size=32, seed=42):
    """
    下载并处理 ESOL 数据集，返回 train, val, test 的 DataLoader
    """
    # 1. 下载/加载数据集
    dataset = MoleculeNet(root=root_dir, name='ESOL')

    # 2. 设置随机种子，保证切分结果可复现 (重要!)
    torch.manual_seed(seed)
    dataset = dataset.shuffle()

    # 3. 按 8:1:1 比例切分
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    # 剩下的给测试集

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # 4. 封装成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 返回 loader 以及特征维度(用于初始化模型)
    return train_loader, val_loader, test_loader, dataset.num_features