import json
import csv
import torch
import torch.nn.functional as F


class testDataset:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def get_td_by_patient_age(self, patient_id, age, eye):
        target_age = round(age, 6)  # Round the target age to two decimal places
        for patient_data in self.data['data'].get(str(patient_id), {}).get(eye, []):
            recorded_age = round(float(patient_data.get('age', 0)), 6)
            if recorded_age == target_age:
                return torch.tensor(patient_data['td'], dtype=torch.float32)
        return None



def add_archetypal_and_decomposed_values_to_json(json_file, decomposed_csv_file, output_file):
    with open(json_file, 'r') as file:
        alldata = json.load(file)

    with open(decomposed_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_id = row['PatientID']
            eye = row['Eye']
            age = round(float(row['Age']), 6)
            highest_decomposed_value = row['HighestDecomposedValue']
            archetype_type = row['Type']
            # decomposed_values = row['DecomposedValues']
            decomposed_values = [
                float(row[f'Archetype_{i}']) for i in range(1, 18)
            ]
            print(f"PatientID: {patient_id}, Eye: {eye}, Age: {age}")
            print(f"Decomposed Values: {decomposed_values}")

            # print(f"Processing DecomposedValues for Patient {patient_id}, Eye {eye}, Age {age}: {decomposed_values}")

            for visit in alldata['data'].get(str(patient_id), {}).get(eye, []):
                recorded_age = round(float(visit.get('age', 0)), 6)
                # print(f"Before adding DecomposedValues: {visit}")
                if recorded_age == age:
                    # print(f"Matching entry found in JSON for Patient {patient_id}, Age {age}")
                    visit['DecomposedValues'] = decomposed_values
                    # print(f"DecomposedValues added for Patient {patient_id}, Age {age}: {visit['DecomposedValues']}")
                    # print(f"After adding DecomposedValues: {visit}")
                    visit['HighestDecomposedValue'] = highest_decomposed_value
                    visit['Type'] = archetype_type
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
    # archetype_csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_max.csv'
    decomposed_csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_all.csv'
    # output_file = './alldata_with_archetypes.json'
    output_file = './alldata_26.json'

    # add_archetypal_results_to_json(json_file, csv_file, output_file)
    add_archetypal_and_decomposed_values_to_json(json_file, decomposed_csv_file, output_file)

if __name__ == "__main__":
    main()
