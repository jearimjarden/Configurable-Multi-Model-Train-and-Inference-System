from pathlib import Path
import pandas as pd

csv_path = Path("data/train/train_data.csv")
df = pd.read_csv(csv_path)
df_new = df.drop(["Loan_Status"], axis=1)
df_csv = df_new.to_csv(index=False)

with open(Path("data/test/new_data.csv"), "w") as f:
    f.write(df_csv)
