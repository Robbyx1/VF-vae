import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F

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
        self.global_min_value = all_sequences.min()
        self.global_max_value = all_sequences.max()
        print(f"Single Global Min Value: {self.global_min_value}")
        print(f"Single Global Max Value: {self.global_max_value}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_tensor = self.sequences[idx]
        # sequence_tensor = sequence_tensor.unsqueeze(0)

        return sequence_tensor

    def get_all_sequences(self):
        return torch.stack(self.sequences)

if __name__ == "__main__":
    dataset = HVFDataset('../src/uwhvf/alldata.json')
    print(f"Loaded {len(dataset)} sequences.")

    if len(dataset) > 0:
        for i in range(3):
            data = dataset[i]
            print(f"Sequence {i + 1} (normalized):", data)

        print("Single global min value:", dataset.global_min_value)
        print("Single global max value:", dataset.global_max_value)

