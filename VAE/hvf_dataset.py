   # dataset = HVFDataset('../src/uwhvf/alldata.json')
import torch
from torch.utils.data import Dataset
import json

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
                           # record['td_seq'][25] = 0  # Adjust for 0-indexed list
                           # record['td_seq'][34] = 0
                           self.sequences.append(record['td_seq'])
                           # print(
                           #     f"Loading {patient_id}-{eye_key}: {record['td_seq'][:5]}...")  # Prints first 5 items

       all_data = torch.tensor(self.sequences, dtype=torch.float32)
       self.mean = all_data.mean(dim=0)
       self.std = all_data.std(dim=0)
       self.std[self.std == 0] = 1  # Prevent division by zero
       self.min = all_data.min()
       self.max = all_data.max()

   def __len__(self):
       return len(self.sequences)

   def __getitem__(self, idx):
       sequence_tensor = torch.tensor(self.sequences[idx], dtype=torch.float32)

       # # Normalize the data[0,1]
       # normalized_data = (torch.tensor(self.sequences[idx], dtype=torch.float32) - self.min) / (self.max - self.min)
       # # normalized_data = (sequence_tensor - self.min) / (self.max - self.min)
       # return normalized_data


       normalized_data = (torch.tensor(self.sequences[idx], dtype=torch.float32) - self.mean) / self.std
       return normalized_data

       # return torch.tensor(self.sequences[idx], dtype=torch.float32)




# In the main testing block:
if __name__ == "__main__":
   dataset = HVFDataset('../src/uwhvf/alldata.json')
   print(f"Loaded {len(dataset)} sequences.")
   if len(dataset) > 0:
       print("First loaded sequence example:", dataset[0])


