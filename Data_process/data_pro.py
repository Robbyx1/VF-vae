import json
import csv
import torch
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

class testDataset:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def get_td_by_patient_age(self, patient_id, age, eye):
        target_age = round(age, 2)  # Round the target age to two decimal places
        for patient_data in self.data['data'].get(str(patient_id), {}).get(eye, []):
            recorded_age = round(float(patient_data.get('age', 0)), 2)
            if recorded_age == target_age:
                return torch.tensor(patient_data['td'], dtype=torch.float32)
        return None



def add_archetypal_and_decomposed_values_to_json(json_file, archetype_csv_file, decomposed_csv_file, output_file):
    with open(json_file, 'r') as file:
        alldata = json.load(file)

    # Load Archetypal Results CSV data
    with open(archetype_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_id = row['PatientID']
            eye = row['Eye']
            age = round(float(row['Age']), 2)
            highest_decomposed_value = row['HighestDecomposedValue']
            archetype_type = row['Type']

            for visit in alldata['data'].get(str(patient_id), {}).get(eye, []):
                recorded_age = round(float(visit.get('age', 0)), 2)
                if recorded_age == age:
                    # Add the archetypal result fields to the visit
                    visit['HighestDecomposedValue'] = highest_decomposed_value
                    visit['Type'] = archetype_type
                    break

    with open(decomposed_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_id = row['PatientID']
            eye = row['Eye']
            age = round(float(row['Age']), 2)
            decomposed_values = row['DecomposedValues']  # Assuming it's a string that needs to be converted to a list


            # decomposed_values = [float(val) for val in decomposed_values.split(',')]
            decomposed_values = decomposed_values.strip('c()')
            # print(f"Processing DecomposedValues for Patient {patient_id}, Eye {eye}, Age {age}: {decomposed_values}")

            for visit in alldata['data'].get(str(patient_id), {}).get(eye, []):
                recorded_age = round(float(visit.get('age', 0)), 2)
                # print(f"Before adding DecomposedValues: {visit}")
                if recorded_age == age:
                    # print(f"Matching entry found in JSON for Patient {patient_id}, Age {age}")
                    visit['DecomposedValues'] = decomposed_values
                    # print(f"DecomposedValues added for Patient {patient_id}, Age {age}: {visit['DecomposedValues']}")
                    # print(f"After adding DecomposedValues: {visit}")
                    td_seq = visit.get('td_seq', [])
                    if td_seq:
                        td_seq_tensor = torch.tensor(td_seq, dtype=torch.float32)  # Convert to tensor
                        md_value = torch.mean(td_seq_tensor).item()  # Calculate mean
                        visit['MD'] = md_value  # Add the MD value to the visit
                    # print(f"After adding DecomposedValues and MD: {visit}")
                    break

    with open(output_file, 'w') as outfile:
        json.dump(alldata, outfile)
        # json.dump(alldata, outfile, indent=4)

    print(f"Updated JSON file saved to {output_file}")


def main():
    json_file = '../src/uwhvf/alldata.json'
    # csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_max.csv'
    archetype_csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_max.csv'
    decomposed_csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed.csv'
    # output_file = './alldata_with_archetypes.json'
    output_file = './alldata_pro.json'

    # add_archetypal_results_to_json(json_file, csv_file, output_file)
    add_archetypal_and_decomposed_values_to_json(json_file, archetype_csv_file, decomposed_csv_file, output_file)

if __name__ == "__main__":
    main()
