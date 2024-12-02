import json
import matplotlib.pyplot as plt
from collections import Counter
import os


# Function to classify MD based on Hodapp-Parrish-Anderson classification
def classify_md(md_value):
    if md_value < -12:
        return "severe"
    elif -12 <= md_value <= -6:
        return "moderate"
    else:
        return "mild"


# Load the dataset
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Process the dataset and classify MD values
def process_data(data):
    classification_counts = {"mild": [], "moderate": [], "severe": []}
    lost_cases_count = 0
    for patient_id, patient_data in data['data'].items():
        for key, visits in patient_data.items():
            if key in ['L', 'R']:  # Only process the visits for "L" and "R" keys
                if isinstance(visits, list):
                    for visit in visits:
                        if isinstance(visit, dict):
                            md_value = visit.get('MD', None)
                            archetype_type = visit.get('Type', None)
                            age = visit.get('age', None)

                            # Check if both MD and Type are available
                            if md_value is not None and archetype_type is not None:
                                stage = classify_md(md_value)
                                classification_counts[stage].append(archetype_type)
                            else:
                                lost_cases_count += 1
                                # print(f"Missing data in visit: {visit}")
                                if md_value is None:
                                    print(f" {patient_id} - {age} - 'MD' value is missing")
                                if archetype_type is None:
                                    print(f"  {patient_id} - {age} - 'Type' value is missing")
                        else:
                            print(f"Unexpected visit format: {visit} (Type: {type(visit)})")
                            lost_cases_count += 1


    return classification_counts, lost_cases_count


# Plot the archetype distribution for each stage
def plot_archetype_distribution(classification_counts, total_cases, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for stage, archetypes in classification_counts.items():
        archetype_counter = Counter(archetypes)

        # Convert the archetype keys to integers for sorting, then back to strings for plotting
        archetypes_list = sorted(archetype_counter.keys(), key=lambda x: int(x))  # Sort numerically
        counts = [archetype_counter[archetype] for archetype in archetypes_list]  # Corresponding counts

        # Get the total number of cases for this classification
        total_classification_cases = len(archetypes)

        plt.figure(figsize=(8, 6))
        plt.bar(archetypes_list, counts, color='skyblue')

        # Update the title to include the total number of cases
        plt.title(f'Archetype Distribution for {stage.capitalize()} Stage (Total: {total_classification_cases})')
        plt.xlabel('Archetype Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'archetype_distribution_{stage}.png')
        plt.savefig(output_path)
        plt.close()


    print(f"Plots saved in directory: {output_dir}")
    print(f"Total cases processed: {total_cases}")


def main():
    # json_file = './alldata_pro.json'
    json_file = './alldata_26.json'
    data = load_json_data(json_file)

    classification_counts, lost_cases_count = process_data(data)

    total_cases = sum(len(v) for v in classification_counts.values())
    print(f"Total lost cases due to unexpected format: {lost_cases_count}")
    output_dir = "archetype_distribution"
    plot_archetype_distribution(classification_counts, total_cases, output_dir)


if __name__ == "__main__":
    main()