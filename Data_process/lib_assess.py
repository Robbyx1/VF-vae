import pandas as pd

def load_and_prepare_csv(file_path):
    # Load CSV
    # df = pd.read_csv(file_path)
    # df['Key'] = df['PatientID'].astype(str) + "_" + df['Age'].astype(str)
    # Load CSV
    df = pd.read_csv(file_path)

    # Construct Key using a tuple of PatientID and Age
    df['Key'] = list(zip(df['PatientID'], df['Age'], df['Eye']))

    # Debug: Check unique PatientID and Age before creating Key
    print("Unique PatientID count:", df['PatientID'].nunique())
    print("Unique Age count:", df['Age'].nunique())
    print("Total rows:", len(df))

    # Debug: Check the first few keys
    print("Sample Keys:")
    print(df['Key'].head())

    # Check for duplicate keys and summarize
    print("Checking for duplicate keys in the dataset...")
    if not df['Key'].is_unique:
        # Identify duplicated keys
        duplicated_keys = df['Key'][df.duplicated('Key', keep=False)]
        unique_duplicated_keys = duplicated_keys.value_counts()

        print("\nDuplicate keys found:")
        print(unique_duplicated_keys)  # Print each duplicated key and the count
        print("\nRows with duplicate keys:")
        print(df[df['Key'].isin(unique_duplicated_keys.index)])
    else:
        print("All keys are unique.")
    return df

def compare_csv_files(python_csv, r_csv, output_csv="type_changed_rows.csv", merged_output_csv="merged_results.csv"):
    python_df = load_and_prepare_csv(python_csv)
    r_df = load_and_prepare_csv(r_csv)

    # Merge the two dataframes on the composite key
    merged_df = pd.merge(python_df, r_df, on='Key', suffixes=('_python', '_r'))

    # Add columns for mean deviation and std deviation (absolute differences)
    merged_df['row_mean_deviation'] = merged_df[
        [f'Archetype_{i}_python' for i in range(1, 18)]
    ].subtract(
        merged_df[[f'Archetype_{i}_r' for i in range(1, 18)]].values, axis=1
    ).abs().mean(axis=1)

    merged_df['row_std_deviation'] = merged_df[
        [f'Archetype_{i}_python' for i in range(1, 18)]
    ].subtract(
        merged_df[[f'Archetype_{i}_r' for i in range(1, 18)]].values, axis=1
    ).abs().std(axis=1)

    # Check if type changed
    merged_df['type_changed'] = merged_df['Type_python'] != merged_df['Type_r']

    archetype_r_columns = [f'Archetype_{i}_r' for i in range(1, 18)]
    merged_df['top_two_diff_r'] = merged_df[archetype_r_columns].apply(
        lambda row: abs(row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1]), axis=1
    )

    archetype_diff_columns = [
        f'diff_archetype_{i}' for i in range(1, 18)
    ]
    merged_df[archetype_diff_columns] = merged_df[
        [f'Archetype_{i}_python' for i in range(1, 18)]
    ].subtract(
        merged_df[[f'Archetype_{i}_r' for i in range(1, 18)]].values, axis=1
    ).abs()

    # Compute the mean absolute difference for each archetype across all rows
    mean_diff_per_archetype = merged_df[archetype_diff_columns].mean(axis=0)

    # Print the mean absolute difference for each archetype
    print("\nMean Absolute Difference for Each Archetype (Python vs R):")
    for archetype, mean_diff in zip(archetype_diff_columns, mean_diff_per_archetype):
        print(f"{archetype}: {mean_diff:.4f}")


    selected_columns = (
        ['PatientID_python', 'Age_python', 'Eye_python', 'row_mean_deviation', 'row_std_deviation','Type_python', 'Type_r', 'type_changed', 'top_two_diff_r'] +
        [f'Archetype_{i}_python' for i in range(1, 18)] +
        [f'Archetype_{i}_r' for i in range(1, 18)]
    )
    cleaned_merged_df = merged_df[selected_columns]

    # Save the merged dataframe with selected columns
    cleaned_merged_df.to_csv(merged_output_csv, index=False)
    print(f"Cleaned merged results saved to {merged_output_csv}")

    # Extract rows where the type changed
    type_changed_rows = merged_df[merged_df['type_changed']]

    # Select columns to save in the output CSV for type changes
    output_columns = (
        ['PatientID_python', 'Age_python', 'Type_python', 'Type_r', 'row_mean_deviation', 'row_std_deviation', 'type_changed', 'top_two_diff_r'] +
        [f'Archetype_{i}_python' for i in range(1, 18)] +
        [f'Archetype_{i}_r' for i in range(1, 18)]
    )
    type_changed_rows_to_save = type_changed_rows[output_columns]
    type_changed_rows_to_save.to_csv(output_csv, index=False)
    print(f"Type-changed rows with archetype values saved to {output_csv}")

    # Print summary statistics
    overall_mean_of_row_deviations = cleaned_merged_df['row_mean_deviation'].mean()
    overall_std_of_row_deviations = cleaned_merged_df['row_std_deviation'].std()
    rate_of_type_change = cleaned_merged_df['type_changed'].mean()
    num_rows_with_type_change = cleaned_merged_df['type_changed'].sum()

    print("Comparison Results:")
    print(f"Overall Mean of Row-Wise Deviations: {overall_mean_of_row_deviations:.4f}")
    print(f"Overall Std of Row-Wise Deviations: {overall_std_of_row_deviations:.4f}")
    print(f"Rate of Type Change: {rate_of_type_change:.2%}")
    print(f"Number of Rows with Type Change: {num_rows_with_type_change}")

    return {
        'overall_mean_of_row_deviations': overall_mean_of_row_deviations,
        'overall_std_of_row_deviations': overall_std_of_row_deviations,
        'rate_of_type_change': rate_of_type_change,
        'num_rows_with_type_change': num_rows_with_type_change,
    }


if __name__ == "__main__":
    python_csv_path = "/Users/xingrobert/Documents/2024/glaucoma progression/VF-vae/Data_process/patient_decomposed_all_python.csv"
    r_csv_path = "/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_all.csv"

    summary = compare_csv_files(python_csv_path, r_csv_path)
    print("Summary of Comparison:")
    # print(summary)
    for key, value in summary.items():
        print(f"{key}: {value}")