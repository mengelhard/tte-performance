import numpy as np
import pandas as pd


def main():

    pass



def kaplan_meier(s_true, t_true):

    t, d, n = hazard_components(s_true, t_true)

    m = np.cumprod(1 - d / n)
    v = (m ** 2) * np.cumsum(d / (n * (n - d)))

    return t, m, v


def nelson_aalen(s_true, t_true):

    t, d, n  = hazard_components(s_true, t_true)

    m = np.cumsum(d / n)
    v = np.cumsum((n - d) * d / ((n - 1) * (n ** 2)))
        
    return t, m, v


def hazard_components(s_true, t_true):

    df = (
        pd.DataFrame({'event': s_true, 'time': t_true})
        .groupby('time')
        .agg(['count', 'sum'])
    )

    t = df.index.values
    d = df[('event', 'sum')].values
    c = df[('event', 'count')].values
    n = np.sum(c) - np.cumsum(c) + c

    return t, d, n


def concordance_index(s_test, t_test, pred_risk):
    
    valid_pairs = (
        (t_test[:, np.newaxis] < t_test[np.newaxis, :]) &
        s_test[:, np.newaxis]
    )
    
    correctly_ranked_pairs = valid_pairs & (pred_risk[:, np.newaxis] > pred_risk[np.newaxis, :])
    
    return np.sum(correctly_ranked_pairs) / np.sum(valid_pairs)


def ipcw_concordance_index(s_train, t_train, s_test, t_test, pred_risk):

    t, m, _ kaplan_meier(1 - s_train, t_train)
    m = 1 - m

    s = pd.Series(data=m, index=t)
    pc = s.reindex(s.index.union(t_test)).interpolate()[t_test].values

    df.reindex(df.index.union(np.linspace(.11,.25,8)))
    
    valid_pairs = (
        (t_test[:, np.newaxis] < t_test[np.newaxis, :]) &
        s_test[:, np.newaxis]
    )
    
    correctly_ranked_pairs = valid_pairs & (pred_risk[:, np.newaxis] > pred_risk[np.newaxis, :])
    
    return np.sum(correctly_ranked_pairs) / np.sum(valid_pairs)


def xCI(s_test, t_test, group1_bool, group2_bool, pred_risk):
    
    valid_pairs = (
        (t_test[:, np.newaxis] < t_test[np.newaxis, :]) &
        s_test[:, np.newaxis] &
        group1_bool[:, np.newaxis] &
        group2_bool[np.newaxis, :]
    )
    
    correctly_ranked_pairs = valid_pairs & (pred_risk[:, np.newaxis] > pred_risk[np.newaxis, :])
    
    return np.sum(correctly_ranked_pairs) / np.sum(valid_pairs)


if __name__ == '__main__':
    main()
