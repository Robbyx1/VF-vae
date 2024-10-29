import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from plot import plot_single_hvf
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

global_min = None
global_max = None

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
                        if 'td' in record:
                            hvf_data = torch.tensor(record['td'], dtype=torch.float32)
                            hvf_data = F.pad(hvf_data, (2, 1, 2, 2), value=100.0)
                            self.sequences.append(hvf_data)


        # valid_data = torch.stack([seq * static_mask for seq in self.sequences])
        # self.std[self.std == 0] = 1  # Prevent division by zero
        # Convert to tensor
        all_sequences = torch.stack(self.sequences)  # Shape: [num_sequences, 12, 12]

        # Apply mask to all sequences
        valid_data = all_sequences * static_mask

        # Calculate point-wise global min and max for all sequences (masked points)
        glo_var.global_min = torch.where(static_mask == 1, valid_data.min(dim=0)[0], torch.tensor(100.0))
        glo_var.global_max = torch.where(static_mask == 1, valid_data.max(dim=0)[0], torch.tensor(100.0))
        # print(f"Global min calculated: {glo_var.global_min}")
        # print(f"Global max calculated: {glo_var.global_max}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # global global_min, global_max  # Access the global min/max
        # if global_min is None or global_max is None:
        #     raise ValueError("Global min/max values have not been initialized.")
        sequence_tensor = self.sequences[idx]
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Changes shape from [8, 9] to [1, 8, 9]
        mask_3d = static_mask.unsqueeze(0)
        min_max_normalized_data = torch.where(
            mask_3d == 1,
            (sequence_tensor - glo_var.global_min) / (glo_var.global_max - glo_var.global_min),
            sequence_tensor  # Leave masked values unchanged
        )
        # normalized_data = ((sequence_tensor - self.mean) / self.std) * mask_3d
        #
        # valid_values = normalized_data[mask_3d == 1]  # Extract only valid values
        #
        # min_val = valid_values.min()
        # max_val = valid_values.max()
        # min_max_normalized_data = torch.where(
        #     mask_3d == 1,
        #     (normalized_data - min_val) / (max_val - min_val),
        #     normalized_data  # Leave masked values unchanged
        # )
        # return sequence_tensor,normalized_data, self.mean, self.std, self.valid_count
        # return sequence_tensor
        return min_max_normalized_data
        # return normalized_data
        # return sequence_tensor

    # def get_global_min_max(self):
    #     """
    #     Method to retrieve the global min and max for normalization in other classes.
    #     """
    #     return self.global_min, self.global_max

# testing block:
if __name__ == "__main__":
    dataset = HVFDataset('../src/uwhvf/alldata.json')
    print(f"Loaded {len(dataset)} sequences.")
    # global_min, global_max = dataset.get_global_min_max()
    if len(dataset) > 0:
        data = dataset[0]  # Access the data
        plot_single_hvf(data ,"single_res")
        print("First loaded sequence example:", data)
        # print("111111test",glo_var.global_max,glo_var.global_min)
        # print(f"Global min calculated: {glo_var.global_min}")
        # print(f"Global max calculated: {glo_var.global_max}")

