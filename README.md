<H3>Name: Lokesh M</H3>
<H3>Reg no:212223230114</H3>
<H3>EX. NO.1</H3>
<H3>Date: 10/03/2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.DataFrame(sns.load_dataset("planets"))
df

df.isnull()

df.fillna(df["mass"].mean().round(1),inplace=True)

df.drop("method",axis=1,inplace=True)
df.head()

scaler = StandardScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
df1.head()

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
print(X)
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train)
print(len(X_test))
print(y_train)
print(len(y_test))
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/ed6840b1-6700-4a78-b214-89800906fe1b)

![image](https://github.com/user-attachments/assets/e0495b09-d61d-48a7-8915-92a17e42e6ae)

![image](https://github.com/user-attachments/assets/a4150407-d65f-47af-b21b-21a102adfe30)

![image](https://github.com/user-attachments/assets/ed8e8416-5d2e-440b-aa8e-1482fa7bf824)

![image](https://github.com/user-attachments/assets/0a440025-75f5-4d17-b59a-d6c4035b7a87)

![image](https://github.com/user-attachments/assets/eb121745-0998-43e0-9769-532da4cf573a)

![image](https://github.com/user-attachments/assets/c4b1c56d-57e2-4bfc-939c-72ab7a78495f)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


