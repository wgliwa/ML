import matplotlib.pyplot as plt
import pandas as pd

titanic = pd.read_csv('TitanicSurvival.csv')
pd.set_option('display.precision', 2)
titanic.columns = ['Nazwisko i imię', 'Ocalały?', 'Płeć', 'Wiek', 'Klasa']
titanic_f = titanic[(titanic['Płeć'] == 'female') & (titanic["Klasa"] == "1st")].sort_values(by=['Wiek']).dropna()
print("lowest age passenger: ", titanic[titanic.Wiek == titanic.Wiek.min()].to_string(index=False, header=False))
print("highest age passenger: ", titanic[titanic.Wiek == titanic.Wiek.max()].to_string(index=False, header=False))
print("mean age of passengers: ", titanic.Wiek.mean())
print("female 1st class survivors: ", titanic_f["Ocalały?"].value_counts()['yes'])
print("lowest age female 1st class: ", titanic_f.head(1).to_string(index=False, header=False))
print("highest age female 1st class: ", titanic_f.tail(1).to_string(index=False, header=False))
print("statistics of surviors:\n", titanic.loc[titanic['Ocalały?'] == 'yes'].describe())
histogram = titanic.hist()
plt.show()
