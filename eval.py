import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import numpy as np
from timeit import default_timer

from dataset import EGGDataset
from model import LinearNet, LSTMNet
from metric import ClassificationMetrics


mat_path = "/userhome/cs/u3568880/EGG_AI_competition/data_EEG_AI.mat"
# model_path = None
model_path = "EGG_LinearNet.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_to = "EGG_LinearNet.pt"
    batch_size = 64

    dataset = EGGDataset(mat_path)
    train_length = len(dataset) * 4 // 5
    test_length = len(dataset) - train_length
    torch.manual_seed(42)
    np.random.seed(42)
    _train_dataset, test_dataset = random_split(
        dataset, [train_length, test_length])

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    net = LinearNet(num_class=26, num_channel=24,
                    num_time_point=801).to(device)
    net.load_state_dict(torch.load(model_path))

    criterion = CrossEntropyLoss()

    timers = [default_timer()]
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

    timers.append(default_timer())
    print(f"Evaluation time: {timers[-1] - timers[-2]:.2f}s")


if __name__ == "__main__":
    main()
