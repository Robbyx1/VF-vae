import torch
from torch.utils.data import Dataset
import json
from plot import plot_single_hvf

# Define the static mask globally
static_mask = torch.tensor([
    [0., 0., 0., 1., 1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 0., 0., 1., 1., 1., 1., 0., 0.]
], dtype=torch.float32)

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
                        if 'hvf' in record:
                            hvf_data = torch.tensor(record['hvf'], dtype=torch.float32)
                            self.sequences.append(hvf_data)

        # self.mean = torch.stack(self.sequences).mean(dim=0)
        # self.std = torch.stack(self.sequences).std(dim=0)
        # self.std[self.std == 0] = 1  # Prevent division by zero
        valid_data = torch.stack([seq * static_mask for seq in self.sequences])
        self.valid_count = static_mask * len(self.sequences)
        # self.mean = valid_data.sum(dim=0) / static_mask.sum()
        # self.mean = valid_data.sum(dim=0) / self.valid_count
        self.mean = torch.where(self.valid_count > 0, valid_data.sum(dim=0) / self.valid_count, torch.tensor(0.0))

        # self.std = ((valid_data - self.mean) ** 2).sum(dim=0) / static_mask.sum()
        # self.std = ((valid_data - self.mean) ** 2).sum(dim=0) / self.valid_count
        self.std = torch.where(self.valid_count > 0, ((valid_data - self.mean) ** 2).sum(dim=0) / self.valid_count,
                               torch.tensor(0.0))
        self.std = torch.sqrt(self.std)
        self.std[self.std == 0] = 1  # Prevent division by zero

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_tensor = self.sequences[idx]
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Changes shape from [8, 9] to [1, 8, 9]
        normalized_data = ((sequence_tensor - self.mean) / self.std) * static_mask
        # return sequence_tensor,normalized_data, self.mean, self.std, self.valid_count
        return normalized_data
        # return sequence_tensor

# testing block:
if __name__ == "__main__":
    dataset = HVFDataset('../src/uwhvf/alldata.json')
    print(f"Loaded {len(dataset)} sequences.")
    if len(dataset) > 0:
        data = dataset[0]  # Access the data
        # plot_single_hvf(data,dataset.mean, dataset.std ,"single_res")
        print("First loaded sequence example:", data)

