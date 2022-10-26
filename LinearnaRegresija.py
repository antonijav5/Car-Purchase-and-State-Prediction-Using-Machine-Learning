import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Obican model Linearne regresije
from sklearn.linear_model import LinearRegression
import LinearRegressionGradientDescent as lrgd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
data = pd.read_csv('car_purchase.csv')
# Ispis prvih 5 redova DataFrame-a
print("Prvih pet redova:")
print(data.head())
print("\n")
print("Poslednjih pet redova:")
print(data.tail())
print("\n")
print("Informacije o tabeli")
print(data.info())
print("\n")
print("Generisanje opisne statistike:")
print(data.describe(percentiles=[0.2, 0.5, 0.8], include=['object', 'float', 'int']))
print("\n")
x = data.loc[:, ['customer_id']]
y = data['max_purchase_amount']
# Skaliramo vrednosti labela y
y = y / 8000
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Id', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the id')
plt.legend()
plt.tight_layout()
plt.show()

data = data.replace({'gender': {'M': 1, 'F': 2}})
x = data.loc[:, ['gender']]

y = data['max_purchase_amount']
# Skaliramo vrednosti labela

y = y / 1000
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Gender', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the gender')
plt.legend()
plt.tight_layout()
plt.show()
data = data.replace({'gender': {1: 'M', 2: 'F'}})
x = data.loc[:, ['age']]
y = data['max_purchase_amount']
# Skaliramo vrednosti labela
y = y / 1000
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Age', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the age')
plt.legend()
plt.tight_layout()
plt.show()
x = data.loc[:, ['annual_salary']]
y = data['max_purchase_amount']
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Annual salary', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the annual salary')
plt.legend()
plt.tight_layout()
plt.show()
x = data.loc[:, ['credit_card_debt']]
y = data['max_purchase_amount']
# Skaliramo vrednosti labela
y = y / 10
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Credit card debt', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the credit card debt')
plt.legend()
plt.tight_layout()
plt.show()

x = data.loc[:, ['net_worth']]
# Skalirano x 10 puta
x = x / 10
y = data['max_purchase_amount']
plt.figure('Maximum purchase amount')
plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='people')
plt.xlabel('Net worth', fontsize=13)
plt.ylabel('Price in $', fontsize=13)
plt.title('Maximum purchase amount for the net worth')
plt.legend()
plt.tight_layout()
plt.show()

# atributi koji ucestvuju u treniranju modela su svi osim customer_id i gender

# transformacija tabele se vrsi nad podacima Gender, tako sto se M imati vr 0, a F 1 ali
# nema nekog efekta

data['age'] = data['age'].apply(lambda l: l * 1000)
x = data[['age', 'annual_salary', 'credit_card_debt', 'net_worth']]

y = data['max_purchase_amount']
x = x / 1000
y = y / 1000
spots = 200
variables = pd.DataFrame(
    data=np.linspace(0, data[['age', 'annual_salary', 'credit_card_debt', 'net_worth']].max(), num=spots))

# Kreiranje i obucavanje modela
lrgdm = lrgd.LinearRegressionGradientDescent()
lrgdm.fit(x, y)

# learning_rates=np.array([[0.17],[0.0000475]])
learning_rates = np.array([[0.45], [0.000475], [0.00003], [0.000002], [0.0000001]])
res_coeff, mse_history = lrgdm.perform_gradient_descent(learning_rates, 1000)

# Kreiranje i obucavanje sklearn.LinearRegression modela
lr_model = LinearRegression()
lr_model.fit(x, y)

# Testiranje predikcije oba modela nad jednim uzorkom
# price = c1 * net_worth + c0
example_estate_sqm = 42, 63000, 11600, 530961
example_estate = pd.DataFrame(data=[example_estate_sqm])
lrgdm.set_coefficients(res_coeff)
print(f'LRGD purchase price for age {example_estate_sqm[0]}'
      f', annual salary {example_estate_sqm[1]}, credit_card_debt {example_estate_sqm[2]} and '
      f'net worth {example_estate_sqm[3]} is '
      f'{lrgdm.predict(example_estate)[0]:.2f}$')
print(f'LRGD c0: {lrgdm.coeff.flatten()[0]:.2f}, '
      f'c1: {lrgdm.coeff.flatten()[1]:.2f} '
      f'c2: {lrgdm.coeff.flatten()[2]:.2f} '
      f'c3: {lrgdm.coeff.flatten()[3]:.2f} '
      f'c4: {lrgdm.coeff.flatten()[4]:.2f} '
      )
print(f'LR purchase price for age {example_estate_sqm[0]}'
      f', annual salary {example_estate_sqm[1]}, credit_card_debt {example_estate_sqm[2]} and '
      f'net worth {example_estate_sqm[3]} is '
      f'{lr_model.predict(example_estate)[0]:.2f}$ ')
print(f'LR c0: {lr_model.intercept_:.2f}, '
      f'c1: {lr_model.coef_[0]:.2f} '
      f'c2: {lr_model.coef_[1]:.2f} '
      f'c3: {lr_model.coef_[2]:.2f} '
      f'c4: {lr_model.coef_[3]:.2f} '
      )

# Stampanje mse za oba modela
# vrednosti su prvobitno male iz razloga sto podaci deljeni sa 1000 zbog preciznosti modela.
# pomnozeno je sa 1000 da bi se dobila prava greska
realCost1 = lrgdm.cost() * 1000
lrgdm.set_coefficients(res_coeff)
print(f'LRGD MSE: {realCost1:.2f}')
c = np.concatenate((np.array([lr_model.intercept_]), lr_model.coef_))
lrgdm.set_coefficients(c)
realCost2 = lrgdm.cost() * 1000
print(f'LR MSE: {realCost2:.2f}')
# Restauracija koeficijenata
lrgdm.set_coefficients(res_coeff)

# Nije moguca vizuelizacija MSE, 6D je.

# Racunanje score-a za oba modela
data_test = pd.read_csv('car_purchase.csv')
x = data_test[['age', 'annual_salary', 'credit_card_debt', 'net_worth']]
y = data_test['max_purchase_amount']
# Zapamte se koeficijenti LR modela,
# da bi se postavili LRGD koeficijenti i izracunao LR score.
lr_coef_ = lr_model.coef_
lr_int_ = lr_model.intercept_
lr_model.coef_ = lrgdm.coeff.flatten()[1:]
lr_model.intercept_ = lrgdm.coeff.flatten()[0]
print(f'LRGD score: {lr_model.score(x, y):.2f}')
# Restauriraju se koeficijenti LR modela
lr_model.coef_ = lr_coef_
lr_model.intercept_ = lr_int_
print(f'LR score: {lr_model.score(x, y):.2f}')
