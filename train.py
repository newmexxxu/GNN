import torch
import os
import matplotlib.pyplot as plt
from GAT.dataset import get_dataloaders
from GAT.model import GNNModel

# --- GIN 超参数 ---
EPOCHS = 1000
BATCH_SIZE = 64
LR = 0.0005
WEIGHT_DECAY = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'checkpoints/best_model.pth'

os.makedirs('checkpoints', exist_ok=True)


def train():
    # 1. 获取数据
    train_loader, val_loader, _, num_features = get_dataloaders(batch_size=BATCH_SIZE)

    # 2. 初始化模型
    model = GNNModel(
        num_node_features=num_features,
        hidden_channels=64,
        num_layers=3,
        dropout=0.1
    ).to(DEVICE)

    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    print(f"Start training GIN on {DEVICE}...")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)

            # 🔥🔥 关键修复：把 data.y 也展平成一维 🔥🔥
            loss = criterion(out.view(-1), data.y.view(-1))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)

                # 🔥🔥 关键修复：验证集也要展平 🔥🔥
                loss = criterion(out.view(-1), data.y.view(-1))

                total_val_loss += loss.item() * data.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1:04d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('GIN Training Curve')
    plt.savefig('loss_curve.png')
    plt.show()


if __name__ == '__main__':
    train()