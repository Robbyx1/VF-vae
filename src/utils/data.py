import json
import numpy as np
from PIL import Image

'''
_plot_vf: create VF image
Parameters:
    vf_array - array-like of padded visual field test
Returns:
    im - PIL image
'''
def _plot_vf(vf_array):
    # convert to grayscale and account for 100 padding
    vf_unscaled = np.array(vf_array)
    padding_mask = vf_unscaled == 100
    vf_scaled = 255*(vf_unscaled - 0) / (50 - 0)
    vf_scaled[padding_mask] = 255

    # increase size of
    factor = 20
    vf_resized = np.repeat(vf_scaled, factor, axis=1)
    vf_resized = np.repeat(vf_resized, factor, axis=0)
    im = Image.fromarray(vf_resized).convert('RGB')
    
    return im

'''
show_vf: show VF image
parameters:
    vf_array - array-like of padded visual field test
'''
def show_vf(vf_array):
    im = _plot_vf(vf_array)
    im.show()

'''
save_vf: save VF image
parameters:
    vf_array - array-like of padded visual field test
    pth - save path
'''
def save_vf(vf_array, pth):
    im = _plot_vf(vf_array)
    im.save(pth)