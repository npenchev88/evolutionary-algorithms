import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, scale_x_log10, scale_y_log10, geom_line, geom_text, theme

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('../../table3.csv')

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Combine the 'Algorithm' and 'Sample' into a single label
df['Algorithm_Sample'] = df['Algorithm'] + '_' + df['Sample'].astype(str)

# Step 3: Handle missing or non-numeric values by converting 'Result' and 'Time' to numeric and dropping NaNs
df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df = df.dropna(subset=['Result', 'Time'])

# Display the DataFrame after processing
print("\nDataFrame after processing:")
print(df)

# Step 4: Create the scatter plot
p = (
    ggplot(df, aes(x='Result', y='Time', color='Algorithm_Sample', group='Algorithm')) +
    geom_point(size=5) +
    geom_line() +
    geom_text(aes(label='Algorithm_Sample'), nudge_x=0.3,nudge_y=-0.1, size=7, va="top", ha="right") +
    scale_x_log10() +  # Logarithmic scale for the x-axis
    scale_y_log10() +  # Logarithmic scale for the x-axis
    labs(
        title='Scatter Plot of Result vs Time by Algorithm and Sample',
        x='Result',
        y='Time'
    ) +
    theme(legend_position='none')
)

# Display the plot
p.show()