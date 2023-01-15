#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv")


#Splitting the dataset in to I.V(x) and D.V(y)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Our main goal is to predict if this employee is bluffing  by building machine learning model that is polynomial regression model


#fit the lin_reg object to x & y. Now our simple linear regression is ready 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)


#we crate an 2nd object for same LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


#lets starts the plotting by true observation 
plt.scatter(x, y, color = 'red')
#we are going to plot for actual value of x & y
plt.plot(x, lin_reg.predict(x), color = 'blue')
#Now plot for the prediction line where x coordinate are predicting points & for y-cordinates predicted value which is lin_reg.predict(x)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
#in y-coordinate we have to replace with lin_reg2 which we create for poly regression model
#x_poly is not defined cuz we already defined in above plot, so insted of x_poly we will define complete fit_trasnform code 
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) # slr

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
#This code show me that predicted salary of 6.5 level using poly reg model
#That means employee is True and we solved this by using polynomial regression model


lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))
lin_reg_2.predict(poly_reg.fit_transform([[8.5]]))
lin_reg_2.predict(poly_reg.fit_transform([[11.5]]))
