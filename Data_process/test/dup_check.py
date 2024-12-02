import pandas as pd

def assess_r_library_csv(file_path):
    """
    Function to assess the R library CSV for duplicate rows based on PatientID, Age, and Eye.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Debug: Check basic information about the dataset
    print("Basic Information:")
    print("Total rows:", len(df))
    print("Unique PatientID count:", df['PatientID'].nunique())
    print("Unique Age count:", df['Age'].nunique())
    print("Unique Eye count:", df['Eye'].nunique())

    # Check for duplicates based on PatientID, Age, and Eye
    print("\nChecking for duplicate (PatientID, Age, Eye) combinations...")
    duplicate_check = df.duplicated(subset=['PatientID', 'Age', 'Eye'], keep=False)

    if duplicate_check.any():
        print(f"Found {duplicate_check.sum()} duplicate rows based on (PatientID, Age, Eye):")

        # Filter rows with duplicate keys
        duplicate_rows = df[duplicate_check]

        # Identify the first 10 unique duplicate keys
        duplicate_keys = duplicate_rows[['PatientID', 'Age', 'Eye']].drop_duplicates().head(10)

        print("\nFirst 10 duplicate keys:")
        print(duplicate_keys)

        print("\nRows corresponding to the first 10 duplicate keys:")
        for _, key_row in duplicate_keys.iterrows():
            patient_id, age, eye = key_row['PatientID'], key_row['Age'], key_row['Eye']
            rows = duplicate_rows[
                (duplicate_rows['PatientID'] == patient_id) &
                (duplicate_rows['Age'] == age) &
                (duplicate_rows['Eye'] == eye)
            ]
            print(f"\nDuplicate Rows for (PatientID={patient_id}, Age={age}, Eye={eye}):")
            print(rows)
    else:
        print("No duplicates found based on (PatientID, Age, Eye). All combinations are unique.")

    # Summary statistics
    print("\nSummary:")
    print(f"Total rows: {len(df)}")
    print(f"Number of unique (PatientID, Age, Eye) combinations: {df.drop_duplicates(subset=['PatientID', 'Age', 'Eye']).shape[0]}")

if __name__ == "__main__":
    # Replace with the path to your R library CSV file
    r_csv_path = "/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_all.csv"

    # Assess the R library CSV
    assess_r_library_csv(r_csv_path)



