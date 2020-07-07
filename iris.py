# Load Libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# DIMENSIONS OF DATASET
# shape
print(dataset.shape)
#sneakpeak
print(dataset.head(20))
#statistical summary
print(dataset.describe())
# how many rows per iris type
print(dataset.groupby('class').size())

# DATA VISUALIZATION
# univariate plots - box plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# univariate plots - histogram
dataset.hist()
pyplot.show()
# multivariate plot - scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Algorithms
array = dataset.values
X = array[:,0:4] # measurements
y= array[:,4] # iris name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    # shuffled everything and created a validation data set

# test different algorithms
models = []
models.append(('Logistic Regression (LR)', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors (KNN)', KNeighborsClassifier()))
models.append(('Classification and Regression Trees (CART)', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes (NB)', GaussianNB()))
models.append(('Support Vector Machiens (SVM)', SVC(gamma='auto')))
# evaluate each model using stratified 10-fold validation
results=[]
names=[]
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare algorithms
pyplot.boxplot(results, labels=['LR','LDA','KNN','CART','NB','SVM'])
pyplot.title('Algorithm Comparison')
pyplot.show()

# BASED ON THE RESULTS IT'S CLEAR THAT SVM IS THE MOST ACCURATE

# MAKE PREDICTIONS
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate Predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))