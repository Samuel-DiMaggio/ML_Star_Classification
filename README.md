# Multiclass Classification Using Logistic Regression, Support Vector Machine, and Random Forest Algorithms

## Purpose:
The purpose for this project is to use machine learning techniques, such as logistical regression, support vector machine, and random forest algorithms to predict multiple classes for a set of data. For this project, in particular, will focus on a dataset set of 240 stars to predict classes of star based on features provided by the dataset and evaluate which technique works best for these predictions. 

## Key Takeaways from this Project
1. During data clean-up identified spelling, grammar, and non-uniformed related errors in the categorical column for Star Color. Reduced 19 color categories to reflect 7 categories. 
2. Reviewing and comparing each algorithm, the random forest algo. performed the best with an accuracy, precision, and f1-score values of 1.0 (or 100%).
3. The logistical regression algo. performed the second best with an accuracy of ~0.958, precision of ~0.966, and f1-score values > 0.85.
4. The SVC had the worst out of the three, however still performed well. SVC performed with an accuracy of 0.916, a precision value of 0.933 and f1-score values > 0.86.

## Data Source And Method of Collection
The dataset was collected from kaggle.com, under title "Star dataset to Predict Star Types" from Deepraj Baidya (The link is provided below). Deepraj describe the purpose for this dataset as, "to prove that the star follow a certain graph in th celestial space, called the Hertzsprung-Russell Diagram or otherwise known as the HR-Diagram". According to this link, Deepraj indicated that this dataset took around 3 weeks to collect data on 240 stars and missing data was manually calculated from the following astrophysics equations:

1. Stefan-Boltzmann's law of Black body radiation (To find the luminosity of a star)
2. Wienn's Displacement law (for finding surface temperature of a star using wavelength)
3. Absolute magnitude relation
4. Radius of a star using parallax.

#### Dataset: https://www.kaggle.com/deepu1109/star-dataset

#### Other Helpful links for about the equations used:

* Stefan-Boltzmann's Law: http://hyperphysics.phy-astr.gsu.edu/hbase/thermo/stefan.html
* Wienn's Displacement Law: http://hyperphysics.phy-astr.gsu.edu/hbase/wien.html
* Absolute Magnitude Relation: https://en.wikipedia.org/wiki/Absolute_magnitude
* Radius of a star using parallax:https://www.youtube.com/watch?v=FnFkb5Dw5-A  

## The dataset contains the following column descriptions:
1. Absolute Temperature (in K)
2. Relative Luminosity (L/Lo)
3. Relative Radius (R/Ro)
4. Absolute Magnitude (Mv)
5. Star Color 
6. Spectral Class (O,B,A,F,G,K,,M)
7. Star Type 

#### Note:
* Lo = 3.828 x 10^26 Watts (Avg Luminosity of Sun)
* Ro = 6.9551 x 10^8 m (Avg Radius of Sun)

#### Star Type describes the following 6 types of stars:

* Brown Dwarf &rarr; Star Type = 0
* Red Dwarf &rarr; Star Type = 1
* White Dwarf &rarr; Star Type = 2
* Main Sequence &rarr; Star Type = 3
* Supergiant &rarr; Star Type = 4
* Hypergiant &rarr; Star Type = 5

## Load and Clean the Data
First, we load the data into memory using pandas.
```python
import pandas as pd
#loading the Stars dataset
stars = pd.read_csv('Stars.csv', low_memory=False)
stars.head()
```
![image](https://user-images.githubusercontent.com/47721595/149651427-21619102-e9a4-4f88-8c95-ac337bd5550a.png)

Reviewing the data types and whether the data has any missing information.
```python
stars.dtypes
```
![image](https://user-images.githubusercontent.com/47721595/149651462-535cd406-ded7-4fa6-904b-dbfcd124c0e6.png)

```python
stars.isnull().sum(axis=0)
```
![image](https://user-images.githubusercontent.com/47721595/149651476-6617c2fc-71ad-4144-8f89-d7c245789464.png)

There is no missing values in all columns. But there are several features are categorical variables. Let's look into it.

## Convert Categorical Features to Numerical Features
We notice that the following features/columns are not numerical variables that include object and int64.

* Star type
* Star Color
* Spectral Class
Star type is our label/target, we normally would need to convert it to numerical variable, however in this case it has already been completed. The remaining two will need to be reviewed for errors and have dummy variables created for their respected categories.

```python
stars['Star type'].value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651623-bc1c99ea-6549-4f09-aec6-574e41cc004b.png)

```python
stars['Star type'] = stars['Star type'].astype('category')
stars.dtypes
```
![image](https://user-images.githubusercontent.com/47721595/149651634-00a296fd-8384-413d-8fe5-cd2137611db6.png)

```python
Unique = ['Star color', 'Spectral Class']
stars[Unique].describe(include='all').loc['unique', :]
```
![image](https://user-images.githubusercontent.com/47721595/149651648-95561d47-9e66-4572-a163-4b8db7d0b6e6.png)

According to this above information, there are 19 different colors and 7 different spectral classes. Let's delve more into these catergories and see if there are spelling, grammar, or additional errors to address first. If there are any errors detected, we can rename these variables to match or create a equvilent/relatable catergory.

```python
stars['Star color'].value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651685-44374383-d34a-4b49-8f74-956bd923f851.png)

So, it looks like there were some spelling/grammar errors and some ambiguity for some colors. Below we will address this issue by creating some new catergorical identifiers.

```python
# Assigning new color identifier for bluish colors
stars['Star color'].replace({'Blue-white':'Bluish',
                             'Blue White':'Bluish',
                             'Blue ':'Bluish',
                             'Blue white':'Bluish',
                             'Blue-White':'Bluish'}, inplace=True)

# Assigning new color identifier for Whitish colors
stars['Star color'].replace({'yellow-white':'Off-White',
                             'Yellowish White':'Off-White',
                             'white':'White',
                            'Whitish':'Off-White'}, inplace=True)

# Assigning new color identifier for Orange, Yellow, and Red colors
stars['Star color'].replace({'Pale yellow orange':'Orange',
                             'White-Yellow':'Yellowish',
                            'yellowish':'Yellowish',
                            'Orange-Red':'Red'}, inplace=True)
```
```python
stars['Star color'].value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651715-43466afc-6007-4b92-9737-208971f193a2.png)

```python
stars['Spectral Class'].value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651729-c5c394d5-eba0-4e5f-9bdb-03465eb46f79.png)

Color and Spectral Class are categorical variables that have only finite many cases. Let's convert them to numerical variables. We use get_dummies function by specifying drop_first = True to reduce the redundant features.

```python
Star_Features = ['Star color','Spectral Class']
factors = pd.get_dummies(stars[Star_Features],drop_first=True)
factors.head()
```
![image](https://user-images.githubusercontent.com/47721595/149651751-6f181e8d-f76b-475e-82cb-32c2dc0c43b1.png)

```python
stars = stars.drop(Star_Features,axis=1)
stars = pd.concat([stars,factors],axis=1)
stars.head()
```
![image](https://user-images.githubusercontent.com/47721595/149651769-a1185535-9f4c-4536-a21d-1a3985b58a6d.png)

## Numerically Summarize the Data
Let's numerically summarize continuous variables in the dataset.
```python
import numpy as np
numerics =['Temperature (K)','Luminosity(L/Lo)', 'Radius(R/Ro)', 
           'Absolute magnitude(Mv)', 'Star color_Bluish',
           'Star color_Off-White','Star color_Orange','Star color_Red',
           'Star color_White', 'Star color_Yellowish', 'Spectral Class_B',
          'Spectral Class_F', 'Spectral Class_G', 'Spectral Class_K', 
          'Spectral Class_M', 'Spectral Class_O']
#summarize it
np.round(stars[numerics].describe(), decimals=4)
```
![image](https://user-images.githubusercontent.com/47721595/149651799-3d08b7e1-260e-46cf-8625-2b103892fe00.png)

Let's look at the correlation between all these numerical features.
```python
corr=stars.corr()
corr.style.background_gradient(cmap='PuBu')
```
![image](https://user-images.githubusercontent.com/47721595/149651826-ab305d72-c52c-4771-821e-28e2a183f70f.png)

## Graphically Summarize the Data
Let's summarize the numerical features graphically.
```python
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sns.pairplot(stars[['Temperature (K)','Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']], 
             diag_kind='kde')
```
![image](https://user-images.githubusercontent.com/47721595/149651858-597bdb24-7dcc-47d5-bb48-46008cd08bf8.png)

```python
sns.boxplot(x='Star type',y='Temperature (K)',data=stars)
```
![image](https://user-images.githubusercontent.com/47721595/149651869-a100e4ee-1910-476c-85b5-94d6b9886ef8.png)

```python
sns.boxplot(x='Star type',y='Luminosity(L/Lo)',data=stars)
```
![image](https://user-images.githubusercontent.com/47721595/149651876-df566fd1-4ca1-4cbb-91ff-1540be1fc577.png)

```python
sns.boxplot(x='Star type',y='Radius(R/Ro)',data=stars)
```
![image](https://user-images.githubusercontent.com/47721595/149651883-9f91a8d1-0d75-438d-8a72-ac7102d35810.png)

```python
sns.boxplot(x='Star type',y='Absolute magnitude(Mv)',data=stars)
```
![image](https://user-images.githubusercontent.com/47721595/149651893-bdeb4bbe-9472-43dd-a7b2-32853b37c15b.png)

## Split the Data Into Training and Test Data Set
Before I split the data into a 80/20 split, I need to reduce the size of the data to reduce the time it will take to train later on. Afterwards I will need first to combine all features into X and select the label column as y.

```python
X = stars.drop('Star type',axis=1)
y = stars['Star type']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = 0, stratify=y)
print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
```
![image](https://user-images.githubusercontent.com/47721595/149651932-b8db3c94-53d4-4637-a72f-efed1231d02e.png)

```python
y_test.value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651946-ad0dc7f2-5321-490d-8e90-f3c00d121489.png)

```python
y_train.value_counts()
```
![image](https://user-images.githubusercontent.com/47721595/149651954-d49aae0b-ff76-46c2-a020-f75a272ecd6e.png)

## Train and evaluate a multiclass classifier
Now that we have a set of training features and corresponding training labels, we can fit a multiclass classification algorithm to the data to create a model. Most scikit-learn classification algorithms inherently support multiclass classification. We'll try a logistic regression algorithm.

```python
from sklearn.linear_model import LogisticRegression
reg = 0.1
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', 
                                 multi_class='auto', max_iter=10000).fit(X_train, y_train)
print (multi_model)
```
![image](https://user-images.githubusercontent.com/47721595/149651978-1b2a55af-2452-404d-84fe-78139e38b1f3.png)

```python
predictions = multi_model.predict(X_test)
print('Predicted labels: ', predictions[:15])
print('Actual labels   : ' ,y_test[:15])
```
![image](https://user-images.githubusercontent.com/47721595/149651989-34c53cfe-a73d-40ac-b40a-44bc880e930d.png)

```python
from sklearn. metrics import classification_report
print(classification_report(y_test, predictions))
```
![image](https://user-images.githubusercontent.com/47721595/149652003-2e343757-d391-4577-a18e-eb663249024d.png)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Overall Accuracy:",accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, predictions, average='macro'))
```
![image](https://user-images.githubusercontent.com/47721595/149652015-33802ba9-a6fa-4dc6-a6bf-259c445c2896.png)

```python
from sklearn.metrics import confusion_matrix

mcm = confusion_matrix(y_test, predictions)
print(mcm)
```
![image](https://user-images.githubusercontent.com/47721595/149652027-2e1247e2-ae46-4c14-9eae-dd14a55f9f1b.png)

```python
classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant', 'Hypergiant']
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Stars")
plt.ylabel("Actual Stars")
plt.show()
```
![image](https://user-images.githubusercontent.com/47721595/149652040-94fc9331-3c6b-4120-b4be-6a5b70bda739.png)

The darker squares in the confusion matrix plot indicate high numbers of cases, and you can hopefully see a diagonal line of darker squares indicating cases where the predicted and actual label are the same.

In the case of a multiclass classification model, a single ROC curve showing true positive rate vs false positive rate is not possible. However, you can use the rates for each class in a One vs Rest (OVR) comparison to create a ROC chart for each class.

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

prob = multi_model.predict_proba(X_test)

fpr = {}
tpr = {}
thresh ={}
for i in range(len(classes)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, prob[:,i], pos_label=i)
    
# Plot the ROC chart
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=classes[0] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=classes[1] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=classes[2] + ' vs Rest')
plt.plot(fpr[0], tpr[0], linestyle='--',color='red', label=classes[3] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='yellow', label=classes[4] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='black', label=classes[5] + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
```
![image](https://user-images.githubusercontent.com/47721595/149652081-6381a054-21cd-431c-937e-fa813b76aa64.png)

To quantify the ROC performance, you can calculate an aggregate area under the curve score that is averaged across all of the OVR curves.

```python
auc = roc_auc_score(y_test,prob, multi_class='ovr')
print('Average AUC:', auc)
```
![image](https://user-images.githubusercontent.com/47721595/149652115-41c682a9-2086-4754-9718-0a9986f43544.png)

## Preprocess data in a pipeline
Using a pipeline to apply preprocessing steps to the data before fitting it to an algorithm to train a model. Let's see if we can improve the predictor by scaling the numeric features in a transformation steps before training. We'll also try a different algorithm (a support vector machine), just to show that we can!
```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Define preprocessing for numeric columns (scale them)
feature_columns = [0,1,2,3,4,5]
feature_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('preprocess', feature_transformer, feature_columns)])

# Create training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', SVC(probability=True))])


# fit the pipeline to train a linear regression model on the training set
multi_model = pipeline.fit(X_train, y_train)
print (multi_model)
```
![image](https://user-images.githubusercontent.com/47721595/149652127-da0827a4-263a-4f57-9b9e-46076669a17c.png)

Now we can evaluate the new model.
```python
predictions = multi_model.predict(X_test)
prob = multi_model.predict_proba(X_test)

print("Overall Accuracy:",accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, predictions, average='macro'))
print('Average AUC:', roc_auc_score(y_test, prob, multi_class='ovr'))
print(classification_report(y_test, predictions))

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()
```
![image](https://user-images.githubusercontent.com/47721595/149652140-338e0940-4a4b-4332-b445-8608e6ec132c.png)

Now, let's give a Random Forest Algorithm a chance.
```python
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', RandomForestClassifier(n_estimators=100))])
model = pipeline.fit(X_train, (y_train))
print (model)
```
![image](https://user-images.githubusercontent.com/47721595/149652280-f3e5700c-3d00-4acd-bc57-0321e3711ae6.png)

```python
predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)
cm = confusion_matrix(y_test, predictions)

print("Overall Accuracy:",accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, predictions, average='macro'))
print('Average AUC:', roc_auc_score(y_test, prob, multi_class='ovr'))
print(classification_report(y_test, predictions))

plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()
```
![image](https://user-images.githubusercontent.com/47721595/149652292-31946ea0-2407-446c-abfb-8df19fe93cc9.png)

## Conclusion:
Reviewing and comparing each algorithm, the random forest algo. performed the best with an accuracy, precision, and f1-score values of 1.0 (or 100%). While the logistical regression algo. performed the second best with an accuracy of ~0.958, precision of ~0.966, and f1-score values > 0.85. The SVC had the worst out of the three, however still performed well. SVC performed with an accuracy of 0.916, a precision value of 0.933 and f1-score values > 0.86. These values, of course were trained on a small dataset and may or not be effective at predicting larger datasets. This of course can be resolved by reviewing the cleaning data steps, identifying potential errors, rerunning the algorithms, and identifying which algo. worked the best.
