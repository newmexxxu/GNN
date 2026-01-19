import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from GNN.dataset import get_dataloaders
from GNN.model import GNNModel  # 引用新模型
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'checkpoints/best_model.pth'


def test():
    # 1. 获取测试集
    _, _, test_loader, num_features = get_dataloaders(batch_size=64)

    # 2. 初始化模型 (参数必须和 train.py 一致)
    model = GNNModel(
        num_node_features=num_features,
        hidden_channels=64,
        num_layers=3,
        dropout=0.1
    ).to(DEVICE)

    # 3. 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("GIN Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    preds = []
    targets = []

    # 4. 推理
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = model(data)
            preds.append(out.view(-1).cpu().numpy())
            targets.append(data.y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 5. 指标
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)

    print("=" * 30)
    print(f"GIN Test Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("=" * 30)

    # 6. 画图
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5, c='green', label='Samples')  # 换个颜色代表新希望

    min_val = min(min(targets), min(preds))
    max_val = max(max(targets), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

    plt.xlabel('True Solubility')
    plt.ylabel('Predicted Solubility')
    plt.title(f'GIN Prediction (RMSE={rmse:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_scatter.png')
    plt.show()


if __name__ == '__main__':
    test()