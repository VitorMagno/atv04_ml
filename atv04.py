# %%
import pandas as pd
from sklearn import mixture
import seaborn as sns
import numpy as np
from scipy import stats
# %%
df = pd.read_csv('./csv')

#%%
df

#%%
df.drop(columns=['Unnamed: 0'], inplace=True)

#%%
df

#%%
cols = ['Age', 'C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

#%%
model = mixture.BayesianGaussianMixture(n_components=6, random_state=12)

# %%
model.weight_concentration_prior_ = 'dirichlet_distribution'

#%%
model.fit(df)

# %%
labels = model.predict(df)

# %%
len(np.unique(labels))

#%%
y = df.copy()

#%%
y['label'] = labels

#%%
y.groupby(by='label')['Age'].agg(['count', 'mean', 'std', 'min', 'max'])

#%%
y.groupby(by='label')['C'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['S'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['ST'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['T'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['IT'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['I'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['IN'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['N'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
y.groupby(by='label')['SN'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
# sns.pairplot(data=y, hue='label', kind='scatter')

# %%
from sklearn.cluster import KMeans

# %%
model = KMeans(n_clusters=8, random_state=12, n_init="auto")
model.fit(df)
model.labels_
#%%
model_y = df.copy()
model_y['label'] = model.predict(df)
#%%
model_y.groupby(by='label')['Age'].agg(['count', 'mean', 'std', 'min', 'max'])

#%%
model_y.groupby(by='label')['C'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['S'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['ST'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['T'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['IT'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['I'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['IN'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['N'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
model_y.groupby(by='label')['SN'].agg(['count', 'mean', 'std', 'min', 'max'])

# %%
sns.pairplot(data=model_y, hue='label', kind='kde')