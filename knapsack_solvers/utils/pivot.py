from datetime import datetime

import pandas as pd
#
df = pd.read_csv('../../table_20250331_131558.csv')
#
# print("Original DataFrame:")
# print(df)
#
# pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result')
# print("\nPivot Table:")
# print(pivot_df)
#
# pivot_df.to_csv('pivot_table_per_result1.csv')
# print("\nPivot table saved to pivot_table.csv")
#
# pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Time')
# # Display the pivot table
# print("\nPivot Table:")
# print(pivot_df)
#
# pivot_df.to_csv('pivot_table_per_time1.csv')
# print("\nPivot table saved to pivot_table.csv")
# df['Time'] = df['Time'].fillna(1)
# df['Result'] = df['Result'].fillna(0)
# print(df)
# df['Result_per_Time'] = df['Result'] / df['Time']
#
# pivot_df = df.pivot_table(index='Algorithm', columns='Sample', values='Result_per_Time')
#
# print(pivot_df)
#
# pivot_df.to_csv('pivot_table_result_per_time1.csv')
# print("\nPivot table saved to pivot_table1.csv")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
benchmark_filename = f"table_{timestamp}.csv"
pivot_result_filename = f"pivot_result_{timestamp}.csv"
pivot_time_filename = f"pivot_time_{timestamp}.csv"
pivot_ratio_filename = f"pivot_result_per_time_{timestamp}.csv"

df.to_csv(benchmark_filename, index=False)

pivot_result = df.pivot_table(index='Algorithm', columns='Sample', values='Result')
pivot_result.to_csv(pivot_result_filename)

pivot_time = df.pivot_table(index='Algorithm', columns='Sample', values='Time')
pivot_time.to_csv(pivot_time_filename)

df['Time'] = df['Time'].fillna(1)
df['Result'] = df['Result'].fillna(0)
df['Result_per_Time'] = df['Result'] / df['Time']
pivot_ratio = df.pivot_table(index='Algorithm', columns='Sample', values='Result_per_Time')
pivot_ratio.to_csv(pivot_ratio_filename)