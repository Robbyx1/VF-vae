import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
import glo_var

# Define the static mask globally
static_mask = torch.tensor([
    [0., 0., 0., 1., 1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 0., 0., 1., 1., 1., 1., 0., 0.]
], dtype=torch.float32)
static_mask = F.pad(static_mask, (2, 1, 2, 2), value=0.0)

class HVFDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

        self.sequences = []
        for patient_id, patient_data in data['data'].items():
            for eye_key in ['R', 'L']:
                if eye_key in patient_data:
                    eye_data = patient_data[eye_key]
                    for record in eye_data:
                        if 'td_seq' in record:
                            hvf_data = torch.tensor(record['td_seq'], dtype=torch.float32)
                            # hvf_data = torch.cat((hvf_data[:26], hvf_data[27:35], hvf_data[36:]))
                            hvf_data = torch.cat((hvf_data[:25], hvf_data[26:34], hvf_data[35:]))
                            self.sequences.append(hvf_data)



        # Convert to tensor
        all_sequences = torch.stack(self.sequences)
        glo_var.global_min = all_sequences.min(dim=0)[0]  # Min for each of the 52 points
        glo_var.global_max = all_sequences.max(dim=0)[0]
        # print(f"Global min calculated: {glo_var.global_min}")
        # print(f"Global max calculated: {glo_var.global_max}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_tensor = self.sequences[idx]
        sequence_tensor = sequence_tensor.unsqueeze(0)
        min_max_normalized_data = (sequence_tensor - glo_var.global_min) / (glo_var.global_max - glo_var.global_min)

        return min_max_normalized_data

if __name__ == "__main__":
    dataset = HVFDataset('../src/uwhvf/alldata.json')
    print(f"Loaded {len(dataset)} sequences.")

    if len(dataset) > 0:
        for i in range(3):
            data = dataset[i]
            print(f"Sequence {i + 1} (normalized):", data)

        # Print global min and max values for validation
        print("Global min calculated:", glo_var.global_min)
        print("Global max calculated:", glo_var.global_max)

