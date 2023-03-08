import pandas as pd

FILL_VALUES = {
    'alb': 3.5,
    'pafi': 333.3,
    'bili': 1.01,
    'crea': 1.01,
    'bun': 6.51,
    'wblc': 9.,
    'urine': 2502.
}

TO_DROP = [
    'aps',
    'sps',
    'surv2m',
    'surv6m',
    'prg2m',
    'prg6m',
    'dnr',
    'dnrday',
    'sfdm2',
    'hospdead'
]

df = (
    pd.read_csv('datasets/support2.csv')
    .drop(TO_DROP, axis=1)
    .fillna(value=FILL_VALUES)
)

# get dummies for categorical vars
df = pd.get_dummies(df, dummy_na=True)

# fill remaining values to the median

df = df.fillna(df.median())

# standardize numeric columns

numrc_cols = df.dtypes == 'float64'
df.loc[:, numrc_cols] = (
    (df.loc[:, numrc_cols] - df.loc[:, numrc_cols].mean())
    / df.loc[:, numrc_cols].std()
)

df.to_csv('datasets/support2_preprocessed.csv', index=False)