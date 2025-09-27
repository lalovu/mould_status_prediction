import pandas as pd
from . import config as cf

def count_issues(df):
    # Count
    n_missing = df[cf.DATETIME_COL].isna().sum()
    n_dupes = df.duplicated(subset=[cf.DATETIME_COL]).sum()

    # Previews
    missing_preview = df[df[cf.DATETIME_COL].isna()].head(10)
    dupes_preview = df[df.duplicated(subset=[cf.DATETIME_COL], keep=False)] \
                    .sort_values(cf.DATETIME_COL) \
                    .head(20)

    # report
    print(f"Total records loaded: {len(df)}")
    print(f"Number of missing timestamps: {n_missing}")
    print(f"Number of duplicate timestamps: {n_dupes}")
    print("\nPreview of missing timestamps:")
    print(missing_preview.to_string(index=False))
    print("\nPreview of duplicate timestamps:")
    print(dupes_preview.to_string(index=False))

def reindex_report(df, raw_df):
    print(f"Reindexed data from {df[cf.DATETIME_COL].min()} to {df[cf.DATETIME_COL].max()}")
    print(f"Total records after reindexing: {len(df)}")
    print(f"Preview {df.head(20)}")
    print("-" * 40)


def interpolate_report(original_df, filled_df):
    total_points = len(original_df)
    total_missing = original_df[cf.SENSOR_COLUMNS].isna().sum().sum()
    total_filled = filled_df[cf.SENSOR_COLUMNS].isna().sum().sum()
    total_interpolated = total_missing - total_filled

    filled_df.to_csv('csv_checklist/filled_df.csv', index=False)

    print(filled_df.head(20).to_string(index=False))
    print(f"Total data points: {total_points * len(cf.SENSOR_COLUMNS)}")
    print(f"Total missing data points before interpolation: {total_missing}")
    print(f"Total data points filled by interpolation: {total_interpolated}")
    print("-" * 40)

def filter_report(original_df, filtered_df):
    filtered_df.to_csv('csv_checklist/filtered_output.csv', index=False)

    # Report
    print(f"Total records after filtering: {len(filtered_df)}")
    print(f"Preview of filtered data:")
    print(filtered_df.head(20).to_string(index=False))
    print("-" * 40)

'''
def plot_data(original_df, filtered_df, interpolated_df, sensor_col):
    plt.figure(figsize=(15, 6))
    plt.plot(original_df[cf.DATETIME_COL], original_df[sensor_col], label='Original Data', color='blue', alpha=0.5)
    plt.plot(filtered_df[cf.DATETIME_COL], filtered_df[sensor_col], label='After Hampel Filter', color='orange', alpha=0.7)
    plt.plot(interpolated_df[cf.DATETIME_COL], interpolated_df[sensor_col], label='After Interpolation', color='green', alpha=0.9)
    
    plt.xlabel('Timestamp')
    plt.ylabel(sensor_col)
    plt.title(f'{sensor_col} Data Processing Steps')
    plt.legend()
    plt.grid(True)
    plt.show()
'''