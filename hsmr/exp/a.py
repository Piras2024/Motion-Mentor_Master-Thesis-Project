import pandas as pd

file = "exp/01_ik.mot"

# find where header ends
with open(file) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "endheader" in line:
        header_end = i
        break

# now read dataframe
df = pd.read_csv(
    file,
    sep="\t",
    skiprows=header_end + 1
)

print(df.shape)
print(df.head())