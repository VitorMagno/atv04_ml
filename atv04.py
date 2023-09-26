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
# remocao de outliers #
# for col in cols:
#     print(f"Old Shape: {df.shape}")
#     z = np.abs(stats.zscore(df[col]))
#     limsuperior = z<3
#     df = df[limsuperior]
#     print("New Shape: ", df.shape)

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
y.groupby(by='label')['Age'].describe()
# %%
y.groupby(by='label')['C'].describe()
# %%
y.groupby(by='label')['S'].describe()
# %%
y.groupby(by='label')['ST'].describe()
# %%
y.groupby(by='label')['T'].describe()
# %%
y.groupby(by='label')['IT'].describe()
# %%
y.groupby(by='label')['I'].describe()
# %%
y.groupby(by='label')['IN'].describe()
# %%
y.groupby(by='label')['N'].describe()
# %%
y.groupby(by='label')['SN'].describe()
# %%
sns.pairplot(data=y, hue='label', kind='scatter')
# %%
df.to_csv('./csv')