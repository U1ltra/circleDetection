import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets.vision import List
from torchvision import datasets, models, transforms

from circleGenerator import CircleParams, generate_examples

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class to hold the image array and corresponding 
    labels, the CircleParams object array.
    """
    def __init__(self, img_src: np.ndarray, circ_src: List[CircleParams]):
        self.images = img_src
        self.circles = circ_src

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            np.expand_dims(
                np.float32(self.images[idx]),
                axis = 0
            ), 
            self.circles[idx]
        )

def noise_removal():
    pass
    
def dataset_generation(
    store_path: str,
    num: int = 1e4,
    noise_level = 0.5
) -> Dataset:
    """
    store_path: path to store the generated numpy array object for
                failure recovery
    num: number of data points to generate
    
    Generate a dataset using the generator and store the feature and
    labels in the given path.
    
    Return a Dataset object ready to be wrapped by data loader
    """
    
    img_gen = generate_examples(noise_level = noise_level)

    feat = []
    lab = []
    for i in range(int(num)):
        data_point = next(img_gen)
        feat.append(
            data_point[0]
        )
        lab.append(data_point[1].to_numpy())
    
    feat = np.array(feat)
    lab = np.array(lab)
    
    with open(store_path + "_feat.npy", 'wb') as f:
        np.save(f, feat)
    with open(store_path + "_lab.npy", 'wb') as f:
        np.save(f, lab)
    
    return Dataset(
        feat,
        lab
    )



