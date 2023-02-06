
from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Generator

import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa



class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int
    
    def to_numpy(self) -> np.ndarray:
        """
        Return np.array([row, col, radius])
        """
        return np.array([
            self.row,
            self.col,
            self.radius
        ], dtype=np.float32)
    
    def from_tensor(tensor_cir: torch.Tensor) -> CircleParams:
        """
        Class Method
        ------------
        tensor_cir: tensor in shape [1,3]
        
        Return a CircleParams construct with the the three elements
        in tensor_cir, with [[row, col, radius]] correspondingly
        """
        return CircleParams(
            tensor_cir[0].item(),
            tensor_cir[1].item(),
            tensor_cir[2].item()
        )
    

def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img


def noisy_circle(
    img_size: int, min_radius: float, max_radius: float, noise_level: float
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """

    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)
    # show_circle(img)

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def show_circle(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Circle')
    plt.show()


def generate_examples(
    noise_level: float = 0.5,
    img_size: int = 100,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None,
    dataset_path: str = 'ds',
) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    assert max_radius > min_radius, "max_radius must be greater than min_radius"
    assert img_size > max_radius, "size should be greater than max_radius"
    assert noise_level >= 0, "noise should be non-negative"

    params = f"noise_level={noise_level}, img_size={img_size}, min_radius={min_radius}, max_radius={max_radius}, dataset_path={dataset_path}"
    print(f"Using parameters: {params}")
    while True:
        img, params = noisy_circle(
            img_size=img_size, min_radius=min_radius, max_radius=max_radius, noise_level=noise_level
        )
        yield img, params
