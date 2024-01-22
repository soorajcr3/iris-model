import pandas as pd
import pickle
df = pd.read_excel('/iris .xls')
x = df.drop(['Classification'],axis=1)
y = df['Classification']
from sklearn import preprocessing
standardisation = preprocessing.StandardScaler()
x = standardisation.fit_transform(x)
x = pd.DataFrame(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=.2)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
pickle.dump(regressor,open('ir_model.pkl','wb'))