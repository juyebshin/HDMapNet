from matplotlib.cm import get_cmap
import numpy as np
from PIL import Image

colors_plt = ['r', 'b', 'g']
color_map = np.random.randint(0, 256, (256, 3))
color_map[0] = np.array([0, 0, 0])

def colorise(input, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    vmin = float(np.min(input)) if vmin is None else vmin
    vmax = float(np.max(input)) if vmax is None else vmax

    input = (input - vmin) / (vmax - vmin)
    if input.ndim > 1:
        return cmap(input)[..., :3] # :3 -> 0, 1, 2
    else:
        return cmap(input)[:3] # :3 -> 0, 1, 2