import pandas as pd

# Define the path to your CSV file
# csv_file_path = '/VF-vae/data/patient_data_decomposed.csv'
csv_file_path = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-vae/data/patient_data_decomposed.csv'
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)
print(df.columns)
# print(df.dtypes)
print(df.head())
print("11111")
# Now using the correct column names as they appear in the DataFrame
patient_id = 647
# age = 52.7967145790554  # Your input age
age = 52.7967146
tolerance = 0.1  # Tolerance for matching the age

# Query the dataframe with the correct column names
# result = df[(df['PatientID'] == patient_id) and (df['Age'] == age)]
result = df[(df['PatientID'] == patient_id)]
print(result.head())

# Check if any results were found and print the relevant data
if not result.empty:
    print(result['DecomposedValues'])
else:
    print(f'No entry found for PatientID {patient_id} with Age {age}')

