###import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data set
data=pd.read_csv("C:\imagecon\Dataset\logistic_regression_uncleaned_dataset.csv")
data
#data inspection
data.info()
data.shape
data.head()
data.tail()
data.describe()

#data cleaning
data.isnull().sum()

#fill na() for salary
data["EstimatedSalary"]=data["EstimatedSalary"].fillna(data["EstimatedSalary"].mean())
#fill na() for age
data["Age"]=data["Age"].fillna(data["Age"].median())

#Eda

import seaborn as sns
#histogram
sns.histplot(data=data,x="Age",color="red",bins=5,weights=2)
plt.title("Age Distribution")
plt.show()
#scatter plot
plt.scatter(data["Age"],
            data["EstimatedSalary"],
            color="red")
plt.show()
#line plot
sns.lineplot(x="Age",
             y="EstimatedSalary",
             data=data,color="black",size=20)
plt.show()

#splitting dataset

x=data.iloc[:,0:2].values
y=data.iloc[:,-1].values

#train test split
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.70,random_state=20)

#pre processing (standard scalar)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x)
x_test=sc.transform(x)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

#predection
y_pred=classifier.predict(x_test)
y_pred
#score
score=classifier.score(x_train,y_train)
score=classifier.score(x_test,y_test)
score

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

plt.figure(dpi=500)
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from matplotlib.colors import ListedColormap
plt.figure(dpi=300)
# Test set
X_set, y_set = x_test, y_test
# Create mesh grid (larger step size = less memory)
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-1,
              stop=X_set[:, 0].max()+1,
              step=0.1),

    np.arange(start=X_set[:, 1].min()-1,
              stop=X_set[:, 1].max()+1,
              step=0.1)
)

# Predict grid points
Z = classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T
)

Z = Z.reshape(X1.shape)

# Decision boundary
plt.contourf(
    X1, X2, Z,
    alpha=0.5,
    cmap=ListedColormap(("red", "green"))
)

# Plot test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=[ListedColormap(("red", "green")).colors[i]],
        label=j
    )

plt.title("Logistic Regression (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()