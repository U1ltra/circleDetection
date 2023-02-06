
import torch
import numpy as np

from model import Net
from dataset import Dataset
from util import test_model, new_test, print_metrics


net = Net()
net.load_state_dict(
    torch.load(
        "./experiments/noise50/best_model25.pt",
        map_location = torch.device('cpu')
    )
)
net.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

test_feat = np.load("./experiments/noise50/test_feat.npy")
test_lab = np.load("./experiments/noise50/test_lab.npy")
test_data = Dataset(test_feat, test_lab)

test_res = test_model(net, test_data, device)
print("Test on original test data >>>")
print_metrics(test_res, 0.5)


torch.manual_seed(50)
np.random.seed(50)

new_test(net, device, 0.7)
new_test(net, device, 0.6)
new_test(net, device, 0.5)
new_test(net, device, 0.4)
new_test(net, device, 0.3)



