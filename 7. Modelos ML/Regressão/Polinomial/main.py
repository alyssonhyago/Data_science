import matplotlib.pyplot as plt
import numpy as np

#Regressão polinomial

m = 100
x = 6 * np.random.rand(m,1) - 3

y = 0.5 * x ** 2 + x +2 + 6 * np.random.randn(m,1)

plt.scatter(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)

x_poly = poly_features.fit_transform(x)
print(x_poly)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

# Variaveis da Regressão

lin_reg.intercept_

lin_reg.coef_

plt.scatter(x,y)
plt.scatter(x, lin_reg.predict(x_poly))