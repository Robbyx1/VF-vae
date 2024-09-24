import csv
import json
import torch
import torch.nn.functional as F
from plot import plot_hvf, plot_hvf_arch, visualize_latent_vectors,visualize_latent_distribution, visualize_latent_distribution_violin,\
    visualize_latent_vectors_across_dimensions,visualize_sorted_latent_vectors,visualize_original_latent_vectors, \
    visualize_highlighted_latent_vectors,visualize_highlighted_and_faded_latent_vectors, visualize_interpolation, plot_interpolated_hvf_arch
from vae_model import VAE
from operator import itemgetter
import numpy as np
from scipy.spatial import distance

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


class HVFDataset:
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


def extract_decomposed_data(file_path):
    extracted_rows = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            type = row['Type']
            # print(f"1Processing DecomposedValues: {type}")
            if int(type) == 8:
                extracted_rows.append(row)

    extracted_rows.sort(key=lambda row: row['Type'], reverse=True)

    return extracted_rows

def extract_top_10_per_type(file_path):
    # Dictionary to store rows for each type
    type_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Get the type for each row
            type_val = int(row['Type'])
            if type_val not in type_dict:
                type_dict[type_val] = []
            type_dict[type_val].append(row)

    for type_val, rows in type_dict.items():
        rows.sort(key=lambda row: float(row['HighestDecomposedValue']), reverse=True)
        type_dict[type_val] = rows[:10]

    return type_dict


# def normalize_to_0_1(td_data):
#     min_val = td_data*static_mask.min()
#     max_val = td_data*static_mask.max()
#     return (td_data - min_val) / (max_val - min_val)

def normalize_to_0_1(td_data):
    # Apply the mask to get only valid (non-masked) elements and exclude values equal to 100
    valid_data = td_data[(static_mask == 1) & (td_data != 100.0)]
    min_val = valid_data.min()
    max_val = valid_data.max()

    normalized_data = (td_data - min_val) / (max_val - min_val)
    normalized_data = torch.where((static_mask == 1) & (td_data != 100.0), normalized_data, td_data)
    return normalized_data

def load_model(model_file):
    latent_dim = 10
    # model_file = 'model_fold_4.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_file))
    return model

def encode_samples_to_latent_space(samples, model_file):
    model = load_model(model_file)
    latent_vectors = []

    for sample in samples:
        td_data = sample['td_data']
        # Normalize td_data to the range [0, 1]
        normalized_td = normalize_to_0_1(td_data)
        # model expects a shape like [1, 1, 12, 12] for a single sample
        normalized_td = normalized_td.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        print(f"Normalized td_data for PatientID {sample['PatientID']}, Age {sample['Age']}:")
        print(f"Shape: {normalized_td.shape}")
        print(normalized_td)
        # Feed the normalized data into the model and extract the latent space
        with torch.no_grad():  # No need to track gradients for evaluation
            latent_space = model.encode(normalized_td)  # Adjust this based on your model structure

        # Append the latent space vector to the list
        latent_vectors.append(latent_space)

        print(f"Latent space for PatientID {sample['PatientID']}, Age {sample['Age']} calculated.")

    return latent_vectors

def encode_top_5_per_type(td_data_dict, model_file):
    encoded_dict = {}  # Dictionary to store latent vectors for each type

    for type_val, td_data_list in td_data_dict.items():
        top_5_data = td_data_list[:5]
        # sampled_data = [item['td_data'] for item in top_5_data]
        latent_vectors = encode_samples_to_latent_space(top_5_data, model_file)
        encoded_dict[type_val] = latent_vectors

    return encoded_dict


def interpolate_latent_vectors(model, z_type_8, z_type_13, steps=10):
    """
    Perform linear interpolation between two latent vectors.

    Args:
        model: The VAE model with an encoder and decoder.
        z_type_8: The latent vector for type 8.
        z_type_13: The latent vector for type 13.
        steps: Number of interpolation steps.

    Returns:
        A list of decoded reconstructions from the interpolated latent space.
    """
    interpolated_results = []

    # Linearly interpolate between z_type_8 and z_type_13
    for alpha in np.linspace(0, 1, steps):
        z_interpolated = (1 - alpha) * z_type_8 + alpha * z_type_13

        # Decode the interpolated latent vector back to data space
        with torch.no_grad():
            decoded = model.decode(z_interpolated.unsqueeze(0))  # Add batch dimension
            interpolated_results.append(decoded.squeeze().cpu().numpy())

    return interpolated_results


def find_extreme_latent_vectors(encoded_latent_vectors):
    # Flatten all latent vectors into a single array
    all_latent_vectors = []
    for type_val, latent_vectors in encoded_latent_vectors.items():
        all_latent_vectors.extend(latent_vectors)  # Collect all latent vectors from all types

    # Convert latent vectors to a numpy array
    latent_array = np.array([lv.squeeze().numpy() for lv in all_latent_vectors])  # Shape: [num_vectors, latent_dim]

    # Calculate pairwise distances between all latent vectors
    max_distance = -1
    max_pair = (None, None)

    for i in range(len(latent_array)):
        for j in range(i + 1, len(latent_array)):
            dist = distance.euclidean(latent_array[i], latent_array[j])  # Euclidean distance
            if dist > max_distance:
                max_distance = dist
                max_pair = (latent_array[i], latent_array[j])

    # Return the two most different latent vectors
    vec1 = torch.tensor(max_pair[0])
    vec2 = torch.tensor(max_pair[1])
    return vec1, vec2

def main():
    hvf_dataset = HVFDataset('../src/uwhvf/alldata.json')
    decomposed_data_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_max.csv'

    extracted_rows = extract_decomposed_data(decomposed_data_file)
    top_10_per_type = extract_top_10_per_type(decomposed_data_file)

    for type_val, top_rows in top_10_per_type.items():
        print(f"Type {type_val}:")
        for row in top_rows:
            print(row)
    print(f"Found {len(extracted_rows)} entries where the 8th archetype is the largest.")

    td_data_dict = {}

    for type_val, extracted_rows in top_10_per_type.items():
        td_data_list = []  # List to store td_data for the current type

        for row in extracted_rows:
            patient_id = row['PatientID']
            year = int(row['Year'])
            eye = row['Eye']
            age = float(row['Age'])
            highest_decomposed_value = float(row['HighestDecomposedValue'])

            td_data = hvf_dataset.get_td_by_patient_age(patient_id, age, eye)

            if td_data is not None:
                td_data = F.pad(td_data, (2, 1, 2, 2), value=100.0) * static_mask
                td_data_list.append({
                    'td_data': td_data,
                    'PatientID': patient_id,
                    'Age': age,
                    'HighestDecomposedValue': highest_decomposed_value,
                })
                # print(f"Patient {patient_id}, Year {year}, Age {age}: TD data shape {td_data.shape}")
            else:
                print(f"TD data not found for Patient {patient_id}, Year {year}, Eye {eye}.")

        # Store the list of td_data for this type in the dictionary
        td_data_dict[type_val] = td_data_list

    # Print the number of TD data entries stored for each type
    for type_val, td_list in td_data_dict.items():
        print(f"Stored {len(td_list)} TD data entries for Type {type_val}.")

    # td_data_list = []
    # for row in extracted_rows:
    #     patient_id = row['PatientID']
    #     year = int(row['Year'])
    #     eye = row['Eye']
    #     age = float(row['Age'])
    #     highest_decomposed_value = float(row['HighestDecomposedValue'])
    #
    #     td_data = hvf_dataset.get_td_by_patient_age(patient_id, age, eye)
    #
    #     if td_data is not None:
    #         td_data = F.pad(td_data, (2, 1, 2, 2), value=100.0) * static_mask
    #         td_data_list.append({
    #             'td_data': td_data,
    #             'PatientID': patient_id,
    #             'Age': age,
    #             'HighestDecomposedValue': highest_decomposed_value,
    #         })
    #         # print(f"Patient {patient_id}, Year {year}, Age {age}: TD data shape {td_data.shape}")
    #     else:
    #         print(f"TD data not found for Patient {patient_id}, Year {year}, Eye {eye}.")
    # print(f"Stored {len(td_data_list)} TD data entries in the list.")

    # # Evenly sample 10 entries from the list
    # td_data_list.sort(key=lambda x: x['HighestDecomposedValue'], reverse=True)
    # if len(td_data_list) >= 10:
    #     step = len(td_data_list) // 10
    #     sampled_data = [td_data_list[i] for i in range(0, len(td_data_list), step)][:10]
    # else:
    #     sampled_data = td_data_list
    #
    # for i, sample in enumerate(sampled_data):
    #     print(f"Sample {i+1}: PatientID: {sample['PatientID']}, Age: {sample['Age']}, "
    #           f"HighestDecomposedValue: {sample['HighestDecomposedValue']}")
    #
    # results_dir = './arch_8'
    # plot_hvf_arch(sampled_data, results_dir)
    # model_file = 'model_fold_4.pth'
    # latent_vectors = encode_samples_to_latent_space(sampled_data, model_file)
    # print("All latent vectors:")
    # for i, latent_vector in enumerate(latent_vectors):
    #     print(f"Latent vector {i + 1}:")
    #     print(latent_vector)
    # visualize_latent_vectors(latent_vectors)
    # visualize_latent_distribution(latent_vectors)
    # visualize_latent_distribution_violin(latent_vectors)
    model_file = 'model_fold_4.pth'
    # latent_vectors = encode_samples_to_latent_space(sampled_data, model_file)
    encoded_latent_vectors = encode_top_5_per_type(td_data_dict, model_file)
    for type_val, latent_vectors in encoded_latent_vectors.items():
        print(f"Encoded {len(latent_vectors)} latent vectors for Type {type_val}.")
    # visualize_latent_vectors_per_type(encoded_latent_vectors)
    # visualize_latent_vectors_across_dimensions(encoded_latent_vectors)
    # visualize_sorted_latent_vectors(encoded_latent_vectors)
    # visualize_original_latent_vectors(encoded_latent_vectors)
    # visualize_highlighted_latent_vectors(encoded_latent_vectors)
    # visualize_highlighted_and_faded_latent_vectors(encoded_latent_vectors)
    z_type_8 = encoded_latent_vectors[3][0]  # 0.991
    z_type_13 = encoded_latent_vectors[5][0] # 0.977
    vec1, vec2 = find_extreme_latent_vectors(encoded_latent_vectors)
    model = load_model(model_file)
    # interpolated_results = interpolate_latent_vectors(model, z_type_8, z_type_13, steps=10)
    interpolated_results = interpolate_latent_vectors(model, vec1, vec2, steps=10)
    # visualize_interpolation(interpolated_results)
    plot_interpolated_hvf_arch(interpolated_results,results_dir='interpolation_results', file_name='latent_space_interpolation.png')
if __name__ == "__main__":
    main()
