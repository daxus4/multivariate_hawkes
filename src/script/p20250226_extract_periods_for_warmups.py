import pandas as pd


def extract_time_windows(D, T, X):
    """
    For each event type in T and each occurrence of such an event in D, extract
    a subset of D containing events within X seconds before its occurrence.

    Parameters:
    - D (pd.DataFrame): DataFrame with "Time" and "Event Type" columns, sorted by Time.
    - T (list): List of event types to process.
    - X (float): Time window size (seconds).

    Returns:
    - List of extracted DataFrames.
    """
    results = []

    for event_type in T:
        event_rows = D[D["Event Type"] == event_type]  # Get rows with event type in T

        for _, event in event_rows.iterrows():
            start_time = event["Time"] - X
            time_window_df = D[(D["Time"] >= start_time) & (D["Time"] <= event["Time"])]
            results.append(time_window_df)

    return results


# Example usage
data = [(1, "A"), (2, "B"), (4.1, "A"), (4.3, "B"), (5, "A")]
D = pd.DataFrame(data, columns=["Time", "Event Type"])
T = ["A"]
X = 1.0  # 1 second before

windows = extract_time_windows(D, T, X)

# Display extracted dataframes
for i, df in enumerate(windows):
    print(f"DataFrame {i+1}:\n{df}\n")
