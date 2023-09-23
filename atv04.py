#%%
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
# %%
df = pd.read_excel('./RTVue_20221110_MLClass.xlsx')

# %%
df.head()

# %%
df.isna().sum()

# %%
df_oldtype = df.dropna(axis=0, inplace=False)
df_oldtype

# %%
df_oldtype.dtypes

# %%
df_oldtype.drop(columns=['Index', 'pID'], inplace=True)

#%%
filtro_genero = df_oldtype['Gender'] == 'F'
filtro_eye = df_oldtype['Eye'] == 'OS'

#%%
df_oldtype.loc[filtro_genero, 'Gender'] = 0
df_oldtype.loc[filtro_genero == False, 'Gender'] = 1

df_oldtype.loc[filtro_eye, 'Eye'] = 0
df_oldtype.loc[filtro_eye == False, 'Eye'] = 1

#%%
df_newtype = df_oldtype.convert_dtypes()

# %%
bandwidth_oldtypes = estimate_bandwidth(df_oldtype, quantile=0.2)
bandwidth_newtypes = estimate_bandwidth(df_newtype, quantile=0.2)