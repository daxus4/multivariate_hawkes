from pathlib import Path

import pandas as pd

PATH = "data\\coe_dataframes\\simulation_dataframes\\multivariate_hawkes"


def get_recursive_files(path):
    """
    Recursively retrieves all files in a directory.

    Parameters:
        path (str): Directory path.

    Returns:
        list: List of file paths.
    """
    path = Path(path)
    files = []
    for p in path.iterdir():
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(get_recursive_files(p))
    return files


def filter_timestamps(df, time_threshold=1, column: str = "real"):
    """
    Filters rows in a DataFrame based on the timestamp selection rule.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Timestamp' column.
        time_threshold (float): Minimum time difference for filtering (default 0.5 seconds).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Ensure 'Timestamp' column exists
    if column not in df.columns:
        raise ValueError("DataFrame must contain a 'Timestamp' column")

    # List to store indices of rows to keep
    keep_indices = []

    # Always keep the first row
    keep_indices.append(df.index[0])
    last_kept = df.iloc[0][column]
    current_index = 0  # Start from the first row

    while True:
        candidate_index = None
        # Find the first row with timestamp > last_kept + time_threshold
        for idx in range(current_index + 1, len(df)):
            if df.iloc[idx][column] > last_kept + time_threshold:
                candidate_index = idx
                break

        # If no candidate is found, break out of the loop
        if candidate_index is None:
            break

        # Determine the row following the candidate
        next_index = candidate_index + 1
        # If next_index is out of bounds, stop the loop
        if next_index >= len(df):
            break

        # Keep the row following the candidate
        keep_indices.append(next_index)
        last_kept = df.iloc[next_index][column]
        current_index = next_index  # Update current index

    # Return the new filtered DataFrame
    return df.loc[keep_indices].reset_index(drop=True)


if __name__ == "__main__":
    files = get_recursive_files(PATH)
    for file in files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, sep="\t")
        df = filter_timestamps(df, column="real")

        # Save the filtered DataFrame to PATH + filtered + all the subdirectories
        file = str(file).replace(
            "simulation_dataframes", "simulation_dataframes_filtered"
        )

        # Create the directories if they don't exist
        Path(file).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file, index=False, sep="\t")
        print(f"Processed file: {file}")
