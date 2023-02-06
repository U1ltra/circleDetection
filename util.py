
import os.path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from dataset import Dataset, dataset_generation
from typing import BinaryIO, List, Tuple
from circleGenerator import CircleParams

def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        return 0
    if d <= abs(r1 - r2):
        return 1
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    h1 = r1_sq * np.arccos(d1 / r1)
    h2 = d1 * np.sqrt(r1_sq - d1**2)
    h3 = r2_sq * np.arccos(d2 / r2)
    h4 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = h1 + h2 + h3 + h4
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union

def logger(output, fp: BinaryIO) -> None:
    """
    Helper function to record the output to a given file while printing
    """
    print(output)
    fp.write(bytes(output+'\n', 'utf-8'))

def save_losses(path: str, losses: List) -> None:
    """
    Save the recorded loss in .npy
    """
    with open(path, "wb") as f:
        np.save(f, np.array(losses, dtype=np.float32))

def test_model(
    net: Net, test_data: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return model's test loss and IOU of each predictions
    """
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size = 128
    )
    
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    
    test_losses = []
    ious = []
    for i, data in enumerate(test_loader, 0):
        x, y = data[0].to(device), data[1].to(device)
        
        pred = net(x)
        loss = criterion1(pred, y) + 0.1*criterion2(pred, y)
        
        test_losses.append(loss.item())
        
        for i in range(len(pred)):
            
            ious.append(
                iou(
                    CircleParams.from_tensor(pred[i]),
                    CircleParams.from_tensor(y[i])
                )
            )
    
    return tuple((np.array(test_losses), np.array(ious)))

def draw_figure(
    train: np.ndarray, valid: np.ndarray, loss_name: str
) -> None:
    """
    Plot a figure for training and validation loss
    """
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label="Train Loss")
    ax.plot(valid, label="Validation Loss")
    ax.legend(loc='best')
    ax.set_title("Loss During Training", fontsize=16)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Loss: {}".format(loss_name), fontsize=14)
    plt.savefig("./loss_figure.pdf")
    plt.show()

def print_metrics(test_res: tuple, noise_level: float) -> None:
    """
    test_res: a tuple that stores (test_loss, test_IOU)
    noise_level: noise level of the test dataset used

    Print the evaluation metrics for this test results
    """
    print(f"{'-'*50}")
    print(f"Test on noise level = {noise_level}")
    avg_loss = np.sum(test_res[0]) / len(test_res[0])
    avg_iou = np.sum(test_res[1]) / len(test_res[1])
    over_eighty = test_res[1] > 0.8
    over_ninty = test_res[1] > 0.9

    print(f"Average Test Loss {avg_loss}")
    print(f"Average IOU {avg_iou}")
    print(f"IOU over 80% {np.sum(over_eighty) / len(over_eighty)}")
    print(f"IOU over 90% {np.sum(over_ninty) / len(over_ninty)}")
    print(f"{'-'*50}")

def new_test(
    net: Net,
    device: torch.device,
    noise_level: float = 0.5
) -> None:
    """
    net: CNN model to be evaluated
    device: torch device object
    noise_level: noise level to be used to generate the new test dataset

    Run a new test on the given model be generating a new circle dataset using
    the specified noise level
    """
    idx = int(10*noise_level)    
    if os.path.isfile(f"./test_{idx}_feat.npy"):
        test_feat = np.load(f"./test_{idx}_feat.npy")
        test_lab = np.load(f"./test_{idx}_lab.npy")
        test_data = Dataset(test_feat, test_lab)
    else:
        test_datasize = 1e3
        test_data = dataset_generation(f"./test_{idx}", test_datasize, noise_level)
    test_res = test_model(net, test_data, device)
    print_metrics(test_res, noise_level)