import matplotlib
matplotlib.use('agg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os

def get_glimpse(x, l, output_size, k, device):
    """Transform image to retina representation

        Assume that width = height and channel = 1
    """
    batch_size, input_size = x.size(0), x.size(2) - 1
    #device = torch.device('cpu')
    assert output_size * 2**(k - 1) <= input_size, \
        "output_size * 2**(k-1) should smaller than or equal to input_size"

    # construct theta for affine transformation
    theta = torch.zeros(batch_size, 2, 3)
    theta[:, :, 2] = l

    scale = output_size / input_size
    osize = torch.Size([batch_size, 1, output_size, output_size])

    for i in range(k):
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        grid = F.affine_grid(theta, osize, align_corners=False).to(device)
        glimpse = F.grid_sample(x, grid, align_corners=False)

        if i==0:
            output = glimpse
        else:
            output = torch.cat((output, glimpse), dim=1)
        scale *= 2

    return output.detach()


def draw_locations(image, locations, weights=None, size=8, epoch=0, save_path='cifar10_rnn_adaptive_12'):
    image = np.transpose(image, (1,2,0))
    weights = weights.detach().cpu().numpy()


    if (epoch>50):
        for idx in range(len(weights[0])-1):
            if (weights[0][idx] < 0.5) and (weights[0][idx+1] < 0.5):
                break

        locations = locations[:idx+1]


    #print(locations.shape)
    locations = list(locations)
    fig, ax = plt.subplots(1, len(locations))
    for i, location in enumerate(locations):
        if len(locations) == 1:
            subplot = ax
        else:
            subplot = ax[i]

        subplot.axis('off')
        subplot.imshow(image, cmap='gray')
        loc = ((location[0] + 1) * image.shape[1] / 2 - size / 2,
           (location[1] + 1) * image.shape[0] / 2 - size / 2)

        rect = patches.Rectangle(
            loc, size, size, linewidth=1, edgecolor='r', facecolor='none')
        subplot.add_patch(rect)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)


    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path+ '/glimpse_%d.png'%epoch, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    img = np.ones((3,3,28,28))

    loc = np.ones((3,2))

    img = torch.Tensor(img).cuda()
    loc = torch.Tensor(loc).cuda()

    out = get_glimpse(img,loc,8,2)
    print(out.shape)