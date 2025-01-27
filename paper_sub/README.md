# Glaucoma Severity Classification: ARVO Abstract

This repository contains Python scripts used for generating the classification histogram for glaucoma severity stages, as described in the ARVO abstract.

## Overview

This project uses Humphrey Visual Field (HVF) data to classify glaucoma progression into stages (mild, moderate, severe) using the Hodapp-Parrish-Anderson classification. The key output is a histogram showing the distribution of archetypes within each stage.

## Data Preparation

The data is derived from **UWHVF json file**. Here's what was done to the data:
1. Extracted archetypes (`Type`) from `patient_data_decomposed.csv` generated via previous R code.
2. Combined archetypes with the original HVF dataset stored in a JSON file.
3. Added a calculated **MD value** (mean deviation), which represents the severity of visual field loss.


## Key Script: `visual.py`

### Purpose
This Python script generates a histogram of archetype distributions for each glaucoma stage.

### Steps Performed in the Script for key feature:
1. **Data Loading**: 
   - Loads 'alldata_26' JSON dataset.
2. **Classification**:
   - Categorizes patients based on MD values.
3. **Visualization**:
   - Generates a histogram for each stage.
   - Labels the plot with the total number of cases for each classification.
4. **Customization**:
   - The archetypes are sorted in ascending order.
   - Total cases and lost cases are annotated on the histogram(no lost).
   
### follow up
For any cases in the histogram that needs to be analysed, 'plot_detailed_histogram' is used for secondary highest archetype.
Heatmap function is an exploration that we didn't include in the abstract at the end.