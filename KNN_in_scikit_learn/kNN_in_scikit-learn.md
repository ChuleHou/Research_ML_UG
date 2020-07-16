# kNN in scikit-learn

## kNN Basics
<div align=center><img width="800" height="400" src="https://github.com/ChuleHou/Research_ML_UG/blob/master/iMages/time-tumorsize.png"/></div>

In the above picture, we can see the two-dimensional data relationship between tumor size and time. The red dots are positive tumors and the blue dots are negative tumors. The green dots are new case. Determine the type of the case by judging the distance between the green dot and the blue or red dot.

### Euclidean Distance
<div align=center><img width="400" height="150" src="https://github.com/ChuleHou/Research_ML_UG/blob/master/iMages/Euclidean_Distance1.png"/></div>
<div align=center><img width="300" height="100" src="https://github.com/ChuleHou/Research_ML_UG/blob/master/iMages/Euclidean_Distance2.png"/></div>

In mathematics, the Euclidean distance or Euclidean metric is the "ordinary" straight-line distance between two points in Euclidean space.

#### raw data

|X_train|y_train|
| ------ | ------ |
|3.393533211, 2.331273381|0|
|3.110073483, 1.781539638|0|
|1.343808831, 3.368360954|0|
|3.582294042, 4.679179110|0|
|2.280362439, 2.866990263|0|
|7.423436942, 4.696522875|1|
|5.745051997, 3.533989803|1|
|9.172168622, 2.511101045|1|
|7.792783481, 3.424088941|1|
|7.939820817, 0.791637231|1|

#### test data

|X|y|
| ------ | ------ |
|8.093607318, 3.365731514|?|

#### Calculate the Eulidean Distance

```sh
from math import sqrt
distances = []
for x_train in X_train:
    d = sqrt(np.sum((x_train - x)**2))
    distances.append(d)
```

#### Sorting distance

```sh
argsort(distances)
```
#### Find the most common y value

```sh
k = 6
```
  - Find the first six values closest to the test value.
  - Use counter to count the value of y.
  - Output the most common y value.
* [Challenge Solution](https://github.com/ChuleHou/Research_ML_UG/blob/master/KNN_in_scikit_learn/KNN_Basics.ipynb)

## kNN in scikit-learn
The K nearest neighbor algorithm is very special and can be considered as an algorithm without a model
In order to be unified with other algorithms, the training data set as the model itself.

#### Use encapsulated KNN algorithm
```sh
%run Research_ML_UG/KNN_in_scikit_learn/KNN_function/KNN.py
```
#### Use KNN in scikit-learn
```sh
from sklearn.neighbors import KNeighborsClassifier
```
* [Challenge Solution](https://github.com/ChuleHou/Research_ML_UG/blob/master/KNN_in_scikit_learn/kNN_in_scikit-learn.ipynb)

## Test algorithm

### data split

<div align=center><img width="400" height="200" src="https://github.com/ChuleHou/Research_ML_UG/blob/master/iMages/Train-Test-Data-Split.png"/></div>

#### Use the iris dataset in sklearn
```sh
from sklearn import datasets
iris = datasets.load_iris()
iris.keys()
```

#### Shuffle the indexex
```sh
shuffled_indexes = np.random.permutation(len(X))
```

#### use the ratio to split the train and test data
```sh
test_ratio = 0.2
test_size = int(len(X) * test_ratio)
test_indexes = shuffled_indexes[:test_size]
train_indexes = shuffled_indexes[test_size:]
```

#### Use encapsulated KNN algorithm
* [Challenge Solution](https://github.com/ChuleHou/Research_ML_UG/blob/master/KNN_in_scikit_learn/Test_the_algorithm.ipynb)


