# Iris-Flower-Classification
This project is a simple machine learning classification problem using the famous Iris flower dataset. The goal of this project is to build a model that can classify the species of iris flowers based on their petal and sepal dimensions.

# Prerequisites
<h3>To run this project, you will need the following:<br></h3>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

<h3>let’s have a look at the first five rows of this dataset:</h3>
data.head()<br>

<h3>let's check for null values</h3>
data.isnull().sum()
![result]()

<h3>let's check for the size of the dataset</h3>
data.shape<br>
(150, 5)<br>

<h3>let’s have a look at the descriptive statistics of this dataset:</h3>
data.describe()<br>

<h3>The target labels of this dataset are present in the species column, let’s have a quick look at the target labels:</h3>
data['species'].unique()<br>
array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)<br>

# Iris Classification Model
Now let’s train a machine learning model for the task of classifying iris species. Here, I will first split the data into training and test sets, and then I will use the KNN classification algorithm to train the iris classification model<br>

x=data.drop('species',axis=1)<br>
y=data['species']<br>
from sklearn.model_selection import train_test_split<br>
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)<br>
from sklearn.neighbors import KNeighborsClassifier<br>
knn= KNeighborsClassifier(n_neighbors=1)<br>
knn.fit(x_train,y_train)<br>
x_new=np.array([[5, 2, 1, 10]])<br>
predictions=knn.predict(x_new)<br>
print("Prediction: {}".format(predictions))<br><br>

Prediction: ['Iris-virginica']

