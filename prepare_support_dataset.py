import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

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

# shuffle

df = df.sample(frac=1., random_state=2023)

df.to_csv('datasets/support2_preprocessed.csv', index=False)

test_idx = len(df) * 3 // 5

df[:test_idx].to_csv('datasets/support2_train.csv', index=False)
df[test_idx:].to_csv('datasets/support2_test.csv', index=False)

s = df['death'].values
t = df['d.time'].values

x = df.drop(['death', 'd.time'], axis=1).values

s_train = s[:test_idx]
s_test = s[test_idx:]

t_train = t[:test_idx]
t_test = t[test_idx:]

x_train = x[:test_idx]
x_test = x[test_idx:]

pred_risk = CoxPHSurvivalAnalysis(alpha=.1).fit(
    x_train,
    Surv().from_arrays(s_train == 1, t_train)
).predict(x_test)

pd.DataFrame({
    'death': s_train,
    'time': t_train
}).to_csv('datasets/support2_train_outcomes.csv', index=False)

pd.DataFrame({
    'death': s_test,
    'time': t_test,
    'pred_risk': pred_risk
}).to_csv('datasets/support2_test_predictions.csv', index=False)
