import pandas as pd


csv_file = '/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed.csv'
df = pd.read_csv(csv_file)


df['DecomposedValues'] = df['DecomposedValues'].str.strip('c()')
df['DecomposedValues'] = df['DecomposedValues'].apply(lambda x: [float(i) for i in x.split(',')])

# Save the cleaned CSV file
df.to_csv('./cleaned_patient_data_decomposed.csv', index=False)
