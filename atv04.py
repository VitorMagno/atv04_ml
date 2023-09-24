# %%
import pandas as pd
from sklearn import mixture
import seaborn as sns
import numpy as np

# %%
df = pd.read_csv('./csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
#%%
model = mixture.BayesianGaussianMixture(n_components=10)
# %%
model.weight_concentration_prior_ = 'dirichlet_distribution'
#%%
model.fit(df)
# %%
labels = model.predict(df)
#%%
y = df.copy()
#%%
y['label'] = labels
# %%
y
# %%
sns.pairplot(data=y, hue='label', kind='kde', height = 20)
# %%