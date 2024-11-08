import sys
sys.path.append(r'/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion')
import json
import numpy as np

# After loading the data
with open('../src/uwhvf/alldata.json') as f:
    dat = json.loads(f.read())

# Initialize a counter for td_seq
td_seq_count = 0

for patient_id, patient_data in dat['data'].items():
    for eye in ['R', 'L']:
        if eye in patient_data:
            for record in patient_data[eye]:
                if 'td_seq' in record:
                    # Count the td_seq
                    td_seq_count += 1

print(f"Total number of td_seq data entries: {td_seq_count}")
