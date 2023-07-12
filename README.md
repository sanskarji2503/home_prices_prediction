# home_prices_prediction
import pandas as pd
from sklearn import linear_model
df =pd.read_csv("D:\\ML\\homeprices_OneHot.csv")
df
dum=pd.get_dummies(df.town)
df_dum=pd.concat([df,dum],axis='columns')
df_dum.drop(['town','Vijay Nagar'],axis='columns',inplace=True)
df_dum
new_df=df_dum.drop('price',axis='columns')
new_df
model=linear_model.LinearRegression()
model
model.fit(new_df,df_dum.price)

model.predict([[3400,0,0]])
