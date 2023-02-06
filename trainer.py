

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.modules.loss import MSELoss

from model import Net
from util import logger, save_losses
from dataset import dataset_generation


# reproductivity 
torch.manual_seed(39)
np.random.seed(39)
fp = open("./train_log.log", "wb")

# model init
logger("Model Initialization ... ", fp)
net = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
print(net)

# hyperparameters
epoch_num = 5   # 200
batch_size = 128

lr = 1e-3
momentum = 0.9
weight_decay = 1e-5

criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

train_datasize = 10 # 3*1e4
valid_datasize = 10 # 1e3
test_datasize = 10  # 1e3


# dataset prep
logger("Dataset preparation ... ", fp)
train_data = dataset_generation("./train", train_datasize)
valid_data = dataset_generation("./valid", valid_datasize)
test_data = dataset_generation("./test", test_datasize)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size = batch_size
)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size = batch_size
)


# training setup
train_batch_losses, valid_batch_losses = [], []
train_losses, valid_losses = [], []
best_val = float("inf")
cnt_model = 0


# main loop
logger(
    f"Training Start at {time.strftime('%H:%M:%S', time.localtime())}",
    fp
)
for epoch in range(epoch_num): 
    time_start = time.time()
    
    # training
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        loss = criterion1(outputs, labels) + 0.1 * criterion2(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
        train_batch_losses.append(loss.item())
    # MSE-mean & L1-mean
    train_losses.append(running_loss / len(train_loader))
    
    # validation
    valid_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            pred = net(inputs)
            loss = criterion1(pred, labels) + 0.1 * criterion2(pred, labels)
            valid_loss += loss.item()
            valid_batch_losses.append(loss.item())
            
        if valid_loss < best_val:
            logger(
                f"{'-'*50}\nSaving Best Model cnt={cnt_model}\nLoss: {best_val:.5f} -> {valid_loss:.5f}\n{'-'*50}",
                fp
            )
            torch.save(net.state_dict, f"./best_model{cnt_model}.pt")
            best_val = valid_loss
            cnt_model+=1
    valid_losses.append(valid_loss / len(valid_loader))
    
            
    if epoch % 5 == 0:
        logger(
            f'\n\nTraining Loss Accumulate {running_loss:.5f}. Training Loss per Batch {running_loss / len(train_loader):.5f}',
            fp
        )
        logger(
            f'Validation Loss Accumulate {valid_loss:.5f}. Validation Loss per Batch {valid_loss / len(valid_loader):.5f}\n\n',
            fp
        )
    
    logger(
        f"Epoch {epoch+1} in {time.time() - time_start:.2f}",
        fp
    )
logger(
    f"Training End at {time.strftime('%H:%M:%S', time.localtime())}",
    fp
)


# state saving
save_losses("./train_batch_loss.npy", train_batch_losses)
save_losses("./train_loss.npy", train_losses)
save_losses("./valid_batch_loss.npy", valid_batch_losses)
save_losses("./valid_loss.npy", valid_losses)
logger("Loss saved...", fp)
fp.close()

    