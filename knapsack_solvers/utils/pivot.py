import pandas as pd

df = pd.read_csv('../../table3.csv')

print("Original DataFrame:")
print(df)

pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result')
print("\nPivot Table:")
print(pivot_df)

pivot_df.to_csv('pivot_table_per_result.csv')
print("\nPivot table saved to pivot_table.csv")

pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Time')
# Display the pivot table
print("\nPivot Table:")
print(pivot_df)

pivot_df.to_csv('pivot_table_per_time.csv')
print("\nPivot table saved to pivot_table.csv")
df['Time'] = df['Time'].fillna(1)
df['Result'] = df['Result'].fillna(0)
print(df)
df['Result_per_Time'] = df['Result'] / df['Time']

pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result_per_Time')

print(pivot_df)

pivot_df.to_csv('pivot_table_result_per_time.csv')
print("\nPivot table saved to pivot_table.csv")
