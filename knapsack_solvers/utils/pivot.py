import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('../../table3.csv')

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Create a pivot table
pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result')
# Display the pivot table
print("\nPivot Table:")
print(pivot_df)

# Step 3: Save the pivot table to a new CSV file
pivot_df.to_csv('pivot_table_per_result.csv')
print("\nPivot table saved to pivot_table.csv")

# Step 2: Create a pivot table
pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Time')
# Display the pivot table
print("\nPivot Table:")
print(pivot_df)

# Step 3: Save the pivot table to a new CSV file
pivot_df.to_csv('pivot_table_per_time.csv')
print("\nPivot table saved to pivot_table.csv")
# Fill missing 'Time' values with a default value (e.g., 1 to avoid division by zero)
df['Time'] = df['Time'].fillna(1)
# Fill missing 'Result' values with 0 or any other default value as needed
df['Result'] = df['Result'].fillna(0)
print(df)
df['Result_per_Time'] = df['Result'] / df['Time']

pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result_per_Time')

print(pivot_df)

# Step 6: Save the pivot table to a new CSV file
pivot_df.to_csv('pivot_table_result_per_time.csv')
print("\nPivot table saved to pivot_table.csv")
