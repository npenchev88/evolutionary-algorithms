import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_text,
    labs, scale_x_log10, scale_y_log10,
    theme, theme_minimal, element_text,
    guides, guide_legend
)


df = pd.read_csv('../../table_20250331_131558.csv')
df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df = df.dropna(subset=['Result', 'Time'])


p = (
    ggplot(df, aes(x='Result', y='Time', color='Algorithm', group='Algorithm')) +
    geom_point(size=5) +
    geom_line(size=2.5) +
    geom_text(aes(label='Sample'), nudge_x=0.05, nudge_y=0.05, size=28) +
    scale_x_log10() +
    scale_y_log10() +
    labs(
        title='Portfolio Optimization: Total Sharpe Ratio vs Execution Time',
        x='Total Sharpe Ratio (Log Scale)',
        y='Execution Time (Log Scale)'
    ) +
    theme_minimal() +
    theme(
        legend_position='none',
        plot_title=element_text(size=34, weight='bold'),
        axis_title_x=element_text(size=30, weight='bold'),
        axis_title_y=element_text(size=30, weight='bold'),
        axis_text=element_text(size=24, weight='bold')
    )
)


p.show()
