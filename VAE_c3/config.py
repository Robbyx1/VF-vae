import torch
import torch.nn.functional as F
# Initialize and store the static mask globally
def init_mask(device):
    mask = torch.tensor([
        [0., 0., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1.],
        [0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 1., 1., 1., 1., 0., 0.]
    ], dtype=torch.float32).to(device)

    return F.pad(mask, (2, 1, 2, 2), value=0.0)