# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('placement-dataset.csv')
sns.histplot(df['iq'], kde=True)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# %%
df.head()

# %%
df.info()

# %%
df.shape

# %%
#steps

#preprocessing + EDA + FEATURE Selection
#extract input and output columns
#scale the values
#train test split
#train the model
#evaluate the model/model selection
#deploy the model


# %%
df = df.iloc[:,1:]
df.head()

# %%
plt.scatter(df['cgpa'],df['iq'], c = df['placement'])

# %%
#cgpa,iq is input column
#placement is output column
x = df.iloc[:,0:2]
y = df.iloc[:,2:]


# %%
x

# %%
y

# %%
#train test split
import sklearn
print(sklearn.__version__)

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

# %%
x_train

# %%
x_test

# %%
y_test

# %%
y_train

# %%
from sklearn.preprocessing import StandardScaler


# %%
scaler = StandardScaler()


# %%
x_train = scaler.fit_transform(x_train)
x_train

# %%
x_test = scaler.fit_transform(x_test)


# %%
x_test

# %%
from sklearn.linear_model import LogisticRegression


# %%
clf = LogisticRegression()

# %%
y_train

# %%
#model training
clf.fit(x_train,y_train)

# %%
type(clf)

# %% [markdown]
# 

# %%
clf.predict(x_test)

# %%
y_test

# %%



