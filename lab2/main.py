import pandas as pd
import re


def format_price(price):
    if isinstance(price, str):
        return abs(int('0' + re.sub("[^\d]", "", price)))
    return abs(price)


df = pd.read_excel("data.xlsx")

# format
df['wartosc'] = df['wartosc'].map(format_price)

# czyszczenie
df = df.dropna()
df = df[df.wartosc > 0]
df = df[(df.kwartal > 0) & (df.kwartal < 5)]

# poprawnosc
df2 = df["nazwa_zmiennej"].str.contains("\D+1 m2\D+")

print(df2)
