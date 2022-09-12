from scipy.io import loadmat
import numpy as np
import torch

data_path = "/userhome/cs/u3568880/EGG_AI_competition/data_EEG_AI.mat"


def peek_data(mat):
    keys = ["label", "channel_labels", "time_points"]

    label_value = mat["label"]
    print(f"label: {label_value.shape} {label_value.dtype}")
    print(f"from {np.min(label_value)} to {np.max(label_value)}")
    print(f"{label_value[:5, 0]} ... {label_value[-5:, 0]}")
    print()

    channel_value = mat["channel_labels"]
    print(f"channel_labels: {channel_value.shape} {channel_value.dtype}")
    print(f"{[c[0] for c in channel_value[:, 0]]}")
    print()

    time_value = mat["time_points"]
    print(f"time_points: {time_value.shape} {time_value.dtype}")
    print(f"from {np.min(time_value)} to {np.max(time_value)}")
    print(f"{time_value[:5, 0]} ... {time_value[-5:, 0]}")
    print()

    data_value = mat["data"]
    print(f"data: {data_value.shape} {data_value.dtype}")
    print(f"from {np.min(data_value)} to {np.max(data_value)}")
    print()


def test_gpu():
    if torch.cuda.is_available():
        print("Using gpu")
    else:
        print("GPU not available")


def main():
    KEYS = {"channel_labels", "data", "label", "time_points"}

    mat = loadmat(data_path)
    peek_data(mat)
    test_gpu()


if __name__ == "__main__":
    main()
