import pandas as pd
df=pd.read_csv('diabetes.csv')
#print(df.head())
#print(df.isnull().values)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)

from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(x_train)

normalized_x_train=scaler.transform(x_train)
normalized_x_test=scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(4)
knn.fit(normalized_x_train,y_train)


import pickle
pickle.dump(knn,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))





