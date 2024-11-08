'''
Read UWHVF data.
Based on example from https://github.com/uw-biomedical-ml/uwhvf.
JSON schema is provided here: https://github.com/uw-biomedical-ml/uwhvf/blob/master/schema.json.
Background on visual field data:
    Range of each point is 0-50 dB, measuring sensitivity to light. 100-padding is used to form 8x9 matrix.
'''

import sys
sys.path.append(r'/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion')
import json
import numpy as np
from PIL import Image
from src.utils.data import show_vf, save_vf
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--show_save',
                        default='save',
                        choices=['show', 'save'],
                        help='Show or save images.')
    parser.add_argument('--save_dir', default='images', type=str, help='Directory to save images.')
    args = parser.parse_args()

    if args.show_save == 'save':
        os.makedirs(args.save_dir, exist_ok=True)

    with open('../src/uwhvf/alldata.json') as f:
        dat = json.loads(f.read())

    # Initialize a counter for td_seq
    td_seq_count = 0

    # Loop through each patient in the dataset
    for patient_id, patient_data in dat['data'].items():

        for eye in ['R', 'L']:
            if eye in patient_data:
                for record in patient_data[eye]:
                    if 'td_seq' in record:
                        # td_seq_count += len(record['td_seq'])
                        td_seq_count += 1

    print(f"Total number of td_seq data entries: {td_seq_count}")
    # dat is loaded as a dictionary
    print(dat.keys())

    # Basic statistics
    print(f"Total of {dat['pts']} patients, {dat['eyes']} eyes, and {dat['hvfs']} HVFs")
    # Expected output: Total of 3871 patients, 7428 eyes, and 28943 HVFs

    print(f"Age of first HVF of the right eye for patient 647: {dat['data']['647']['R'][0]['age']}")
    # Expected output: Age of first HVF of the right eye for patient 647: 52.79671457905544


    # print(np.array(dat['data']['647']['R'][0]['td']))
    # Expected output:
    # [[100.   100.   100.    -3.23  -5.88  -6.43  -4.72 100.   100.  ]
    #  [100.   100.    -4.77  -4.51  -6.26  -5.42  -4.09  -4.29 100.  ]
    #  [100.    -8.91  -4.39  -4.18  -5.29  -5.85  -4.3   -3.97  -5.51]
    #  [ -9.04  -4.48  -3.82  -4.45  -3.83  -4.12  -3.21  21.    -4.07]
    #  [ -9.04  -6.13  -4.78  -2.51  -3.16  -3.94  -5.34   0.    -4.91]
    #  [100.    -4.35  -3.12  -1.36  -4.67  -2.36  -3.99  -6.09  -5.  ]
    #  [100.   100.    -5.13  -3.57  -3.65  -4.73  -3.24  -4.27 100.  ]
    #  [100.   100.   100.    -2.46  -4.54  -5.23  -3.82 100.   100.  ]]

    # generate VF image in original 8x9 dimension. Note how low-resolution this is
    vf_unscaled = np.array(dat['data']['2']['R'][0]['hvf'])

    ##drawwww
    # padding_mask = vf_unscaled == 100
    # vf_scaled = 255*(vf_unscaled - 0) / (50 - 0)
    # vf_scaled[padding_mask] = 255 # convert padded regions to white
    # im = Image.fromarray(vf_scaled).convert('RGB')
    # if args.show_save == 'show':
    #     im.show()
    # else:
    #     im.save(os.path.join(args.save_dir,'low_res_vf.png'))
    #
    #
    # # to increase resolution, just repeat the array along both axes
    # factor = 20
    # vf_resized = np.repeat(vf_scaled, factor, axis=1)
    # vf_resized = np.repeat(vf_resized, factor, axis=0)
    # im = Image.fromarray(vf_resized).convert('RGB')
    # if args.show_save == 'show':
    #     im.show()
    # else:
    #     im.save(os.path.join(args.save_dir,'high_res_vf.png'))
    #
    # # call the function that does the above
    # if args.show_save == 'show':
    #     show_vf(dat['data']['2']['R'][0]['hvf'])
    # else:
    #     save_vf(dat['data']['2']['R'][0]['hvf'], os.path.join(args.save_dir,'high_res_vf_func.png'))