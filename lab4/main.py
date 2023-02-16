import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

temp = pd.read_csv("data.csv")
df = temp[(temp.Date <= 201801) & (temp.Date >= 195301)]
df['Value'] = (df['Value'] - 32) * 5 / 9
df.columns = ["Data", "Temperatura"]
print(df.describe())
linear_regression = stats.linregress(x=df.Data, y=df.Temperatura)
print(linear_regression)
print("\n difference: ", abs(df.Temperatura[[123]] - (linear_regression.slope * 2021 - linear_regression.intercept)))
x = df["Data"]
plt.plot(df["Data"] / 100, df["Temperatura"], 'o')
plt.plot(df["Data"] / 100, linear_regression.intercept + linear_regression.slope * df["Data"], 'r')
plt.show()
