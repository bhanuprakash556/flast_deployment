import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
data = pd.read_csv('Cost_of_living_index.csv')
data=data.fillna(data.mean)
data.drop("City",axis=1,inplace=True)
x = data[['Cost of Living Plus Rent Index','Groceries Index','Restaurant Price Index','Local Purchasing Power Index']]
y = data[['Cost of Living Index']]
lr = LinearRegression()
lr.fit(x,y)
pickle.dump(lr,open('new_model1.pk','wb'))
