import torch
import torch.nn.functional as F
import csv
import json
from vae_model import VAE
import glo_var  # For global min/max variables
from hvf_dataset import HVFDataset as glo
from plot import plot_interpolated_hvf_arch
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
            data = json.load(file)

        self.data = data

    def get_td_by_patient_age(self, patient_id, age, eye):
        target_age = round(age, 2)  # Round the target age to two decimal places
        for patient_data in self.data['data'].get(str(patient_id), {}).get(eye, []):
            recorded_age = round(float(patient_data.get('age', 0)), 2)
            if recorded_age == target_age:
                return torch.tensor(patient_data['td'], dtype=torch.float32)
        return None


def extract_patient_eye_data(file_path):
    patient_eye_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_id = row['PatientID']
            eye = row['Eye']
            age = float(row['Age'])

            patient_eye_key = (patient_id, eye)

            if patient_eye_key not in patient_eye_dict:
                patient_eye_dict[patient_eye_key] = []

            patient_eye_dict[patient_eye_key].append({
                'PatientID': patient_id,
                'Eye': eye,
                'Age': age,
            })

    # Filter patients' eyes with more than 10 visits and sort visits by age
    filtered_patient_eye_dict = {
        patient_eye_key: sorted(visits, key=lambda x: x['Age'])
        for patient_eye_key, visits in patient_eye_dict.items()
        if len(visits) > 15
    }

    return filtered_patient_eye_dict

# Method to normalize the td_data based on global min/max
def normalize_to_0_1(td_data):
    valid_min = glo_var.global_min[(static_mask == 1) & (glo_var.global_min != 100.0)]
    valid_max = glo_var.global_max[(static_mask == 1) & (glo_var.global_max != 100.0)]

    min_val = valid_min.min()
    max_val = valid_max.max()

    normalized_data = (td_data - min_val) / (max_val - min_val)

    # Restore the original padding values of 100
    normalized_data = torch.where((static_mask == 1) & (td_data != 100.0), normalized_data, td_data)

    return normalized_data


def encode_dictionary_to_latent_space(patient_eye_data_dict, model_file):
    model = load_model(model_file)
    encoded_dict = {}  # Dictionary to store latent vectors for each patient and eye

    # Iterate over each patient and eye in the dictionary
    for (patient_id, eye), visits in patient_eye_data_dict.items():
        latent_vectors = []  # List to store latent vectors for each visit

        for visit in visits:
            td_data = visit['td_data']

            # Normalize td_data to the range [0, 1]
            normalized_td = normalize_to_0_1(td_data)

            # model expects a shape like [1, 1, 12, 12] for a single sample
            normalized_td = normalized_td.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Feed the normalized data into the model and extract the latent space
            with torch.no_grad():  # No need to track gradients for evaluation
                latent_space = model.encode(normalized_td)  # Adjust this based on your model structure

            # Append the latent space vector for this visit
            latent_vectors.append(latent_space)

        # Store the latent vectors for the current patient and eye in the dictionary
        encoded_dict[(patient_id, eye)] = latent_vectors

    return encoded_dict

# Load the VAE model
def load_model(model_file):
    latent_dim = 10  # Assuming latent dimension size is 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

def decode_latent_vectors(model, latent_vector_list):
    decoded_results = []

    for latent_vector in latent_vector_list:
        with torch.no_grad():
            # Decode the latent vector back to data space
            decoded = model.decode(latent_vector.unsqueeze(0))  # Add batch dimension
            decoded_results.append(decoded.squeeze().cpu().numpy())  # Store the decoded result

    return decoded_results



def main():
    hvf_dataset = testDataset('../src/uwhvf/alldata.json')
    # print(f"Global min1: {glo_var.global_min}, Global max1: {glo_var.global_max}")
    glo_cal = glo('../src/uwhvf/alldata.json')
    # print(f"Global min2: {glo_var.global_min}, Global max2: {glo_var.global_max}")
    model_file = 'model_fold_4.pth'

    decomposed_data_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_max.csv'
    patient_eye_data_dict = extract_patient_eye_data(decomposed_data_file)

    # unique_patients = set()
    #
    # for (patient_id, eye), visits in patient_eye_data_dict.items():
    #     unique_patients.add(patient_id)  # Track unique patient IDs
    #     print(f"Patient ID: {patient_id}, Eye: {eye}")
    #     for visit in visits:
    #         print(
    #             f"  Age: {visit['Age']}")
    #     print("\n")
    # print(f"Total number of different patients: {len(unique_patients)}")

    td_data_dict = {}

    for patient_eye_key, visits in patient_eye_data_dict.items():
        td_data_list = []
        patient_id, eye = patient_eye_key

        for visit in visits:
            age = visit['Age']

            # Retrieve the td_data for each visit from the HVF dataset
            td_data = hvf_dataset.get_td_by_patient_age(patient_id, age, eye)

            if td_data is not None:
                td_data = F.pad(td_data, (2, 1, 2, 2), value=100.0)
                td_data_list.append({
                    'td_data': td_data,
                    'Age': age
                })
            else:
                print(f"TD data not found for Patient {patient_id}, Age {age}, Eye {eye}.")

        # Store the list of TD data for each patient/eye in the dictionary
        td_data_dict[patient_eye_key] = td_data_list

    encoded_dict = encode_dictionary_to_latent_space(td_data_dict, model_file)
    # for (patient_id, eye), latent_vectors in encoded_dict.items():
    #     print(f"Patient ID: {patient_id}, Eye: {eye}, Number of Visits: {len(latent_vectors)}")
    #
    #     # Print each latent vector for the patient's visits
    #     for i, latent_vector in enumerate(latent_vectors):
    #         print(f"  Visit {i + 1} Latent Vector: {latent_vector}")

    first_key = list(encoded_dict.keys())[10]
    patient_id, eye = first_key
    latent_vectors = encoded_dict[first_key]

    print(f"Selected Patient ID: {patient_id}, Eye: {eye}")
    print(f"Number of latent vectors for this patient-eye pair: {len(latent_vectors)}")
    # for i, latent_vector in enumerate(latent_vectors):
    #     print(f"Latent vector for Visit {i + 1}: {latent_vector}")
    model = load_model(model_file)
    decode_res = decode_latent_vectors(model,latent_vectors)

    plot_interpolated_hvf_arch(decode_res, results_dir='patient_visit_results',
                               file_name='patient_progression.png')
if __name__ == "__main__":
    main()
