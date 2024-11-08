import json
import matplotlib.pyplot as plt
import numpy as np

# Helper function to transform td_seq into an 8x9 image matrix
def transform_to_image(data):
    matrix = np.full((8, 9), 100.0)
    indices = [
        (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6)
    ]
    # Fill the specified indices with data from the input array
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    masked_matrix = np.where(matrix == 100, np.nan, matrix)
    return masked_matrix

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def find_patients_with_many_visits(data, min_visits=15):
    patients_with_many_visits = {}

    for patient_id, patient_data in data['data'].items():
        for eye, visits in patient_data.items():
            if isinstance(visits, list) and len(visits) > min_visits:
                if patient_id not in patients_with_many_visits:
                    patients_with_many_visits[patient_id] = {}
                patients_with_many_visits[patient_id][eye] = visits

    return patients_with_many_visits


def plot_td_values_for_patient(patients_with_many_visits):
    for patient_id, eyes_data in patients_with_many_visits.items():
        for eye, visits in eyes_data.items():
            ages = []
            td_images = []

            for visit in visits[:15]:
                age = visit.get('age', None)
                td_values = visit.get('td_seq', None)

                if age is not None and td_values is not None:
                    ages.append(age)
                    td_images.append(transform_to_image(td_values))  # Transform td_seq to image

            if ages and td_images:
                num_visits = len(ages)
                rows, cols = 3, 5

                fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
                axes = axes.flatten()

                # Plot each visit's TD image with corresponding age
                for i, (age, td_image) in enumerate(zip(ages, td_images)):
                    axes[i].imshow(td_image, cmap='gray', interpolation='none')
                    axes[i].axis('off')
                    axes[i].set_title(f'Age {age:.2f}')

                if num_visits < rows * cols:
                    for i in range(num_visits, rows * cols):
                        axes[i].axis('off')

                plt.suptitle(f'TD Value Progression for Patient {patient_id}, Eye {eye}')
                plt.tight_layout()
                plt.show()

# Main function to find patients with more than 15 visits and plot TD values
def main():
    json_file = './alldata_pro.json'  # Path to your JSON file
    data = load_json_data(json_file)
    patients_with_many_visits = find_patients_with_many_visits(data)
    plot_td_values_for_patient(patients_with_many_visits)

if __name__ == "__main__":
    main()
