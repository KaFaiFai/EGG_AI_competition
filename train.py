import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import numpy as np
from timeit import default_timer

from dataset import EGGDataset
from model import LinearNet, LSTMNet
from metric import ClassificationMetrics

torch.manual_seed(42)
np.random.seed(42)

mat_path = "/userhome/cs/u3568880/EGG_AI_competition/data_EEG_AI.mat"
# model_path = None
model_path = "EGG_LinearNet.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_to = "EGG_LinearNet.pt"
    num_epoch = 10000
    batch_size = 64
    lr = 3e-4

    dataset = EGGDataset(mat_path)
    train_length = len(dataset) * 4 // 5
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = random_split(
        dataset, [train_length, test_length])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    net = LinearNet(num_class=26, num_channel=24,
                    num_time_point=801).to(device)
    if model_path is not None:
        print(f"Loading model from {model_path}")
        net.load_state_dict(torch.load(model_path))

    optimizer = Adam(net.parameters(), lr=lr, weight_decay=0.005)
    criterion = CrossEntropyLoss()

    timers = [default_timer()]
    for epoch in range(num_epoch):
        print(f"----- Epoch {epoch:>5d}/{num_epoch} -----")

        all_losses, all_outs, all_labels = [], None, None
        for i, (data, labels) in enumerate(train_dataloader):
            net.train()
            data, labels = data.to(device), labels.to(device) - 1
            out = net(data)

            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            all_losses.append(loss.item())
            all_outs = torch.cat(
                (all_outs, out), 0) if all_outs is not None else out
            all_labels = torch.cat((all_labels, labels),
                                   0) if all_labels is not None else labels

            if i % 30 == 0:
                print(f"[Batch {i:>4d}/{len(train_dataloader)}]"
                      f" Loss: {loss.item():.4f}")

        net.eval()
        average_loss = np.mean(all_losses)
        metrics = ClassificationMetrics(truths=all_labels, outputs=all_outs)
        print("-"*20)
        print(f"Training loss: {average_loss:.4f}"
              f"| Training accuracy: {metrics.accuracy:.2%}")

        # evaluation on test set
        net.eval()
        all_test_outs, all_test_labels = None, None
        for i, (data, labels) in enumerate(test_dataloader):
            net.train()
            data, labels = data.to(device), labels.to(device) - 1
            out = net(data)

            all_test_outs = torch.cat(
                (all_test_outs, out), 0) if all_test_outs is not None else out
            all_test_labels = torch.cat(
                (all_test_labels, labels), 0) if all_test_labels is not None else labels

        test_metrics = ClassificationMetrics(
            truths=all_test_labels, outputs=all_test_outs)
        test_metrics.print_report()

        print(f"Saving model to {save_to}")
        torch.save(net.state_dict(), save_to)

        timers.append(default_timer())
        print(f"Epoch time: {timers[-1] - timers[-2]:.2f}s")

        print('\n')


if __name__ == "__main__":
    main()
