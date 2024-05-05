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
                           self.sequences.append(record['td_seq'])
                           # print(
                           #     f"Loading {patient_id}-{eye_key}: {record['td_seq'][:5]}...")  # Prints first 5 items

   def __len__(self):
       return len(self.sequences)

   def __getitem__(self, idx):
       return torch.tensor(self.sequences[idx], dtype=torch.float32)
       # return self.sequences[idx]



# In the main testing block:
if __name__ == "__main__":
   dataset = HVFDataset('../src/uwhvf/alldata.json')
   print(f"Loaded {len(dataset)} sequences.")
   if len(dataset) > 0:
       print("First loaded sequence example:", dataset[0])


