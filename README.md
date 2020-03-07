# EmployeeChurnPredictor-
#Calculate the probability an employee will leave a firm in the coming year using a decision tree. The predictor also indicates the factor with the highest influence allowing the user to try and mitigate to retain the employee. In python. 


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

 

 

#Import data

 

df = pd.read_excel("dumby_hr_data.xlsx") 

 

#Examine data

df.info()

 

 

#Clean Data

#Remove Redundant Varaibles

df = df.drop(['Managed Geography ID', 'Managed Geography Description','Job Function', 'Job Family', 'Officer Title (Emp Only)', 'Country','Business Unit', 'Source System', 'GOC OU', 'GOC OU Description', 'GOC BU', 'GOC BU Description', 'GOC LV', 'GOC LV Description','HR Responsible (Emp Only)'], axis = 1)

 

#Machine Learning Algorithm only takes in numerical data,therefore non numerical variables must be converted to numerical based on whether it is ordinal (has instrict order, like c-level) or nominal (no instrict order, like City). If a variable is ordinal then the data must be encoded (assigned a integers in increasing value). If a variable is nominal then it must be given dummies variable (tunred into columns and assigned a zero or a one).

 

df['clevel'] = df['clevel'].astype('category')

df.clevel = df.clevel.cat.reorder_categories(['c10','c11','c12','c13','c14'])

df.clevel = df.clevel.cat.codes

job_title = pd.get_dummies(df['Job Title'])

df = df.drop('Job Title', axis = 1)

df = df.join(job_title)

managed_segment_description = pd.get_dummies(df['Managed Segment Description'])

df = df.join(managed_segment_description)

df = df.drop('Managed Segment Description', axis = 1)

city = pd.get_dummies(df['City'])

df = df.drop('City', axis = 1)

df = df.join(city)

direct_staff = pd.get_dummies(df['Direct Staff Indicator'])

df = df.join(direct_staff)

df = df.drop('Direct Staff Indicator', axis = 1)

funding = pd.get_dummies(df['Funding'])

df = df.join(funding)

df = df.drop('Funding', axis = 1)

df.info()

import matplotlib.pyplot as plt

import seaborn as sns

corr_matrix = df.corr()

sns.heatmap(corr_matrix)

plt.show()

df['Left'] = df['Left'].astype('category')

df.Left = df.Left.cat.reorder_categories(['No','Yes'])

df.Left = df.Left.cat.codes

target = df.Left

features = df.drop('Left', axis = 1)

from sklearn.model_selection import train_test_split, target_train, target_test, features_train
features_test = train_test_split(target,features,test_size=0.25,random_state=42)

from sklearn.tree import DecisionTreeClassifier

 

# Initialize and call model by specifying the random_state parameter

model = DecisionTreeClassifier(random_state=42)

 

# Apply a decision tree model to fit features to the target

model.fit(features_train, target_train)

 

# Apply a decision tree model to fit features to the target in the training set

model.fit(features_train,target_train)


# Check the accuracy score of the prediction for the training set

model.score(features_train,target_train)*100

 
# Check the accuracy score of the prediction for the test set

model.score(features_test,target_test)*100

 
# Initialize and call model by specifying the random_state parameter

model_depth_5 = DecisionTreeClassifier(max_depth = 5, random_state=42)

 
# Apply a decision tree model to fit features to the target

model_depth_5.fit(features_train, target_train)


# Check the accuracy score of the prediction for the test set

model_depth_5.score(features_test,target_test)*100
