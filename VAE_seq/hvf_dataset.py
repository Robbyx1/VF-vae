import torch
from torch.utils.data import Dataset
import json

class SequentialHVFDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

        self.input_data = []
        self.target_data = []

        # Assume data is ordered chronologically for each patient and eye
        for patient_id, patient_data in data['data'].items():
            for eye_key in ['R', 'L']:
                if eye_key in patient_data:
                    eye_data = patient_data[eye_key]
                    for i in range(len(eye_data) - 2):  # Ensure there's enough data for a sequence
                        current = eye_data[i]['td_seq']
                        next_seq = eye_data[i + 1]['td_seq']
                        next_target = eye_data[i + 2]['td_seq']
                        self.input_data.append(current + next_seq)  # Concatenate current and next for input
                        self.target_data.append(next_target)  # Use the following observation as target

        self.input_data = torch.tensor(self.input_data, dtype=torch.float32)
        self.target_data = torch.tensor(self.target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]





# In the main testing block:
if __name__ == "__main__":
   dataset = SequentialHVFDataset('../src/uwhvf/alldata.json')
   print(f"Total entries in dataset: {len(dataset)}")

   # Print the first 5 entries to check their format and correctness
   for i in range(min(5, len(dataset))):  # Make sure not to exceed the dataset size
       inputs, targets = dataset[i]
       print(f"Sample {i}:")
       print("Input Data (current + next HVF data):", inputs)
       print("Target Data (subsequent HVF data):", targets)
       print()  # Print a blank line for better readability between samples


