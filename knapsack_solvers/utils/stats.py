import pandas as pd, numpy as np

df = pd.read_csv("../../table_20250720_144033.csv")
summary = (df.groupby(["Algorithm","Sample"])
             .agg(Result_mean=("Result","mean"),
                  Result_min =("Result","min"),
                  Result_max =("Result","max"),
                  Time_mean  =("Time"  ,"mean"),
                  Time_min   =("Time"  ,"min"),
                  Time_max   =("Time"  ,"max"))
             .reset_index())


summary["Result_CI"] = (1.96*df.groupby(["Algorithm","Sample"])
                                      ["Result"].std().reset_index(drop=True)
                                     /np.sqrt(30))
summary["Time_CI"]   = (1.96*df.groupby(["Algorithm","Sample"])
                                      ["Time"].std().reset_index(drop=True)
                                     /np.sqrt(30))
summary.to_csv("summary_table.csv", index=False)

df = pd.read_csv("summary_table.csv")

mean_tbl   = df.pivot_table(index="Algorithm",
                            columns="Sample",
                            values="Result_mean")
time_tbl   = df.pivot_table(index="Algorithm",
                            columns="Sample",
                            values="Time_mean")
ratio_tbl  = mean_tbl / time_tbl

# round for nicer LaTeX:
mean_tbl  = mean_tbl.round(1)
time_tbl  = time_tbl.round(2)
ratio_tbl = ratio_tbl.round(1)

mean_tbl.to_csv("sharpe_mean.csv")
time_tbl.to_csv("time_mean.csv")
ratio_tbl.to_csv("efficiency.csv")
