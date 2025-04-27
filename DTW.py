import os
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from glob import glob
import numpy as np

# Set move number
move_number = 3  # Change this manually to 2 or 3 if needed

# Paths (quotes handle spaces automatically)
initial_folder = f'Raw Data/Move {move_number}'
output_folder = f'Aligned Data/Move {move_number}'
os.makedirs(output_folder, exist_ok=True)

# Read all CSVs
csv_files = sorted(glob(os.path.join(initial_folder, '*.csv')))
dataframes = [pd.read_csv(f) for f in csv_files]

# Pick the first CSV in the folder as the reference (golden standard)
reference_file = csv_files[0]
reference_df = pd.read_csv(reference_file)
reference = reference_df[['Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle']].to_numpy()

# Check for NaN/Inf in the reference data and remove them
reference = reference[~np.isnan(reference).any(axis=1)]
reference = reference[~np.isinf(reference).any(axis=1)]

print(f"Reference file selected: {os.path.basename(reference_file)}")

# Align all other attempts
for i, df in enumerate(dataframes):
    attempt_full = df  # Keep the full DataFrame (including all columns)
    attempt_angles = attempt_full[['Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle']].to_numpy()

    # Clean attempt data (remove rows with NaN or Inf)
    attempt_angles = attempt_angles[~np.isnan(attempt_angles).any(axis=1)]
    attempt_angles = attempt_angles[~np.isinf(attempt_angles).any(axis=1)]

    # Perform DTW to align the angles
    distance, path = fastdtw(attempt_angles, reference, dist=euclidean)

    # Handle dynamic lengths — ensure the aligned data matches the reference's length
    aligned_angles = [attempt_angles[j] for _, j in path if j < len(attempt_angles)]
    aligned_com_velocity = [attempt_full.iloc[j]['CoM Velocity'] for _, j in path if j < len(attempt_full)]
    aligned_move_successful = [attempt_full.iloc[j]['Move Successful'] for _, j in path if j < len(attempt_full)]

    # Ensure the aligned sequences have the same length (based on the reference)
    aligned_angles = np.array(aligned_angles)
    aligned_com_velocity = np.array(aligned_com_velocity)
    aligned_move_successful = np.array(aligned_move_successful)

    # If any sequence is shorter than the reference, pad it (optional based on your needs)
    while len(aligned_angles) < len(reference):
        aligned_angles = np.vstack([aligned_angles, aligned_angles[-1]])
        aligned_com_velocity = np.append(aligned_com_velocity, aligned_com_velocity[-1])
        aligned_move_successful = np.append(aligned_move_successful, aligned_move_successful[-1])

    # Create the final aligned DataFrame
    aligned_df = pd.DataFrame({
        'Frame': range(len(aligned_angles)),  # You can adjust frame numbering if needed
        'Left Arm Angle': aligned_angles[:, 0],
        'Right Arm Angle': aligned_angles[:, 1],
        'Left Leg Angle': aligned_angles[:, 2],
        'Right Leg Angle': aligned_angles[:, 3],
        'CoM Velocity': aligned_com_velocity,
        'Move Successful': aligned_move_successful
    })

    # Save the aligned data to a new CSV file
    aligned_filename = os.path.join(output_folder, f'aligned_{os.path.basename(csv_files[i])}')
    aligned_df.to_csv(aligned_filename, index=False)

    print(f'Aligned file saved: {aligned_filename}')
