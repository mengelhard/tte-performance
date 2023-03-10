import numpy as np
import pandas as pd


def main():

    df = pd.read_csv('datasets/support2_preprocessed.csv')

    s = df['death'].values
    t = df['d.time'].values

    x = df.drop(['death', 'd.time'], axis=1).values

    test_idx = len(df) * 4 // 5

    s_train = s[:test_idx]
    s_test = s[test_idx:]

    t_train = t[:test_idx]
    t_test = t[test_idx:]

    x_train = x[:test_idx]
    x_test = x[test_idx:]

    from statsmodels.duration.hazard_regression import PHReg

    mdl = PHReg(t_train, x_train, s_train).fit_regularized(alpha=.1)
    pred_risk = mdl.predict(x_test).predicted_values

    print('CI = %.3f' % concordance_index(s_test, t_test, pred_risk))
    print('IPCW CI = %.3f' % ipcw_concordance_index(s_test, t_test, pred_risk, s_train, t_train))


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


def kaplan_meier(s_true, t_true, tol=1e-6):

    t, d, n = hazard_components(s_true, t_true)

    m = np.cumprod(1 - d / n)
    v = (m ** 2) * np.cumsum(d / (tol + n * (n - d)))

    return t, m, v


def nelson_aalen(s_true, t_true):

    t, d, n  = hazard_components(s_true, t_true)

    m = np.cumsum(d / n)
    v = np.cumsum((n - d) * d / ((n - 1) * (n ** 2)))
        
    return t, m, v


def interpolate(x, y, new_x):
    
    s = pd.Series(data=y, index=x)
    new_y = s.reindex(s.index.union(new_x).unique()).interpolate()[new_x].values
    
    return new_y


def censoring_km(s_train, t_train, t_test):

    t, m, _ = kaplan_meier(1 - s_train, t_train)
    #m = 1 - m

    return interpolate(t, m, t_test)


def valid_pairs(s, t, g1=None, g2=None):

    g1 = g1 or np.ones_like(s)
    g2 = g2 or np.ones_like(s)

    return (
        (t[:, np.newaxis] < t[np.newaxis, :]) &
        s[:, np.newaxis] &
        g1[:, np.newaxis] &
        g2[np.newaxis, :]
    )


def concordance_index(s_test, t_test, pred_risk, return_num_valid=False):
    
    valid = valid_pairs(s_test, t_test)
    correctly_ranked = valid & (pred_risk[:, np.newaxis] > pred_risk[np.newaxis, :])
    
    if return_num_valid:
        return np.sum(correctly_ranked) / np.sum(valid), np.sum(valid)
    else:
        return np.sum(correctly_ranked) / np.sum(valid)


def ipcw_concordance_index(s_test, t_test, pred_risk, s_train, t_train, return_num_valid=False):

    max_event_time = t_train[s_train == 1].max()

    t_test_v = t_test[t_test < max_event_time]
    s_test_v = s_test[t_test < max_event_time]
    pred_risk_v = pred_risk[t_test < max_event_time]

    ckm = censoring_km(s_train, t_train, t_test_v)[:, np.newaxis]
    
    valid = valid_pairs(s_test_v, t_test_v)    
    correctly_ranked = valid & (pred_risk_v[:, np.newaxis] > pred_risk_v[np.newaxis, :])

    if return_num_valid:
        return np.sum((ckm ** -2) * correctly_ranked) / np.sum((ckm ** -2) * valid), np.sum((ckm ** -2) * valid)
    else:
        return np.sum((ckm ** -2) * correctly_ranked) / np.sum((ckm ** -2) * valid)


def xCI(s_test, t_test, pred_risk, g1_bool, g2_bool, return_num_valid=False):

    valid = valid_pairs(s_test, t_test, g1_bool, g2_bool)    
    correctly_ranked = valid & (pred_risk[:, np.newaxis] > pred_risk[np.newaxis, :])

    if return_num_valid:
        return np.sum(correctly_ranked) / np.sum(valid), np.sum(valid)
    else:
        return np.sum(correctly_ranked) / np.sum(valid)


def ipcw_xCI(s_test, t_test, pred_risk, g1_bool, g2_bool, s_train, t_train, return_num_valid=False):

    max_event_time = t_train[s_train == 1].max()

    t_test_v = t_test[t_test < max_event_time]
    s_test_v = s_test[t_test < max_event_time]
    pred_risk_v = pred_risk[t_test < max_event_time]

    ckm = censoring_km(s_train, t_train, t_test_v)[:, np.newaxis]
    
    valid = valid_pairs(s_test_v, t_test_v, g1_bool, g2_bool)
    correctly_ranked = valid & (pred_risk_v[:, np.newaxis] > pred_risk_v[np.newaxis, :])
    
    if return_num_valid:
        return np.sum((ckm ** -2) * correctly_ranked) / np.sum((ckm ** -2) * valid), np.sum((ckm ** -2) * valid)
    else:
        return np.sum((ckm ** -2) * correctly_ranked) / np.sum((ckm ** -2) * valid)


def xxCI(s_test, t_test, pred_risk, g1_bool, g2_bool):

    m1, n1 = xCI(s_test, t_test, pred_risk, g1_bool, g2_bool, return_num_valid=True)
    m2, n2 = xCI(s_test, t_test, pred_risk, g2_bool, g1_bool, return_num_valid=True)
    
    return (m1 * n1 + m2 * n2) / (n1 + n2)


def ipcw_xxCI(s_test, t_test, pred_risk, g1_bool, g2_bool, s_train, t_train):

    m1, n1 = ipcw_xCI(s_test, t_test, pred_risk, g1_bool, g2_bool, s_train, t_train, return_num_valid=True)
    m2, n2 = ipcw_xCI(s_test, t_test, pred_risk, g2_bool, g1_bool, s_train, t_train, return_num_valid=True)
    
    return (m1 * n1 + m2 * n2) / (n1 + n2)


if __name__ == '__main__':
    main()
