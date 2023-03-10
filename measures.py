import numpy as np
import pandas as pd


def main():

    df_train = pd.read_csv('datasets/support2_train_outcomes.csv')
    df_test = pd.read_csv('datasets/support2_test_predictions.csv')

    s_train = df_train['death'].values
    s_test = df_test['death'].values

    t_train = df_train['time'].values
    t_test = df_test['time'].values

    pred_risk = df_test['pred_risk'].values

    print('CI = %.3f' % xCI(s_test, t_test, pred_risk))
    print('IPCW CI = %.3f' % xCI(s_test, t_test, pred_risk, ipcw=True, s_train=s_train, t_train=t_train))


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


def kaplan_meier(s_true, t_true, t_new=None):

    t, d, n = hazard_components(s_true, t_true)

    m = np.cumprod(1 - np.divide(d, n, out=np.zeros(len(d)), where=n > 0))
    v = (m ** 2) * np.cumsum(np.divide(d, n * (n - d), out=np.zeros(len(d)), where=n * (n - d) > 0))

    if t_new is not None:
        return interpolate(t, m, t_new)
    else:
        return t, m, v


def nelson_aalen(s_true, t_true, t_new=None):

    t, d, n  = hazard_components(s_true, t_true)

    m = np.cumsum(np.divide(d, n, out=np.zeros(len(d)), where=n > 0))
    v = np.cumsum(np.divide((n - d) * d, (n - 1) * (n ** 2), out=np.zeros(len(d)), where=(n - 1) > 0))
        
    if t_new is not None:
        return interpolate(t, m, t_new)
    else:
        return t, m, v


def interpolate(x, y, new_x):
    
    s = pd.Series(data=y, index=x)
    new_y = s.reindex(s.index.union(new_x).unique()).interpolate()[new_x].values
    
    return new_y


def xAUCt(s_test, t_test, pred_risk, times, g1_bool=None, g2_bool=None):

    # NOTE: enter groups g1_bool and g2_bool for xAUC_t; omit for AUC_t

    g1 = g1_bool if g1_bool is not None else np.ones_like(s_test)
    g2 = g2_bool if g2_bool is not None else np.ones_like(s_test)

    # pred_risk can be 1d (static) or 2d (time-varying)
    if len(pred_risk.shape) == 1:
        pred_risk = pred_risk[:, np.newaxis]

    # positives: s_test = 1 & t_test =< t
    pos = (t_test[:, np.newaxis] <= times[np.newaxis, :]) & s_test[:, np.newaxis] & g1[:, np.newaxis]
    
    # negatives: t_test > t
    neg = (t_test[:, np.newaxis] > times[np.newaxis, :]) & g2[:, np.newaxis]

    valid = pos[:, np.newaxis, :] & neg[np.newaxis, :, :]
    correctly_ranked = valid & (pred_risk[:, np.newaxis, :] > pred_risk[np.newaxis, :, :])

    return np.sum(correctly_ranked, axis=(0, 1)) / np.sum(valid, axis=(0, 1))


def xROCt(s_test, t_test, pred_risk, time, g1_bool=None, g2_bool=None):

    # NOTE: enter groups g1_bool and g2_bool for xROC_t; omit for ROC_t

    g1 = g1_bool if g1_bool is not None else np.ones_like(s_test)
    g2 = g2_bool if g2_bool is not None else np.ones_like(s_test)

    threshold = np.append(np.sort(pred_risk), np.infty)

    # positives: s_test = 1 & t_test =< t
    pos = (t_test < time) & s_test & g1
    
    # negatives: t_test > t
    neg = (t_test > time) & g2

    # prediction
    pred = pred_risk[:, np.newaxis] > threshold[np.newaxis, :]

    tpr = np.sum(pred & pos[:, np.newaxis], axis=0) / np.sum(pos)
    fpr = np.sum(pred & neg[:, np.newaxis], axis=0) / np.sum(neg)

    return tpr, fpr, threshold


def xCI(s_test, t_test, pred_risk,
        ipcw=False, s_train=None, t_train=None,
        g1_bool=None, g2_bool=None,
        return_num_valid=False, tied_tol=1e-8):

    # NOTE: enter groups g1_bool and g2_bool for xROC_t; omit for ROC_t

    g1 = g1_bool if g1_bool is not None else np.ones_like(s_test)
    g2 = g2_bool if g2_bool is not None else np.ones_like(s_test)

    if ipcw:
        
        assert (s_train is not None) and (t_train is not None), 's_train and t_train are required when ipcw=True'

        before_last_cens = t_test < t_train[s_train == 0].max()

        s_test, t_test, pred_risk, g1, g2 = (
            arr[before_last_cens] for arr in (s_test, t_test, pred_risk, g1, g2)
        )

        w = (1 / kaplan_meier(1 - s_train, t_train, t_test) ** 2)[:, np.newaxis]

    else:

        w = 1.

    valid = (
        (t_test[:, np.newaxis] < t_test[np.newaxis, :]) &
        s_test[:, np.newaxis] &
        g1[:, np.newaxis] &
        g2[np.newaxis, :]
    )

    correctly_ranked = valid & (pred_risk[:, np.newaxis] > (pred_risk[np.newaxis, :] + tied_tol))
    tied = valid & (np.abs(pred_risk[:, np.newaxis] - pred_risk[np.newaxis, :]) <= tied_tol)

    num_valid = np.sum(w * valid)
    ci = np.sum(w * (correctly_ranked + 0.5 * tied)) / num_valid

    return (ci, num_valid) if return_num_valid else ci


def xxCI(s_test, t_test, pred_risk, g1_bool, g2_bool, ipcw=False, s_train=None, t_train=None):

    m1, n1 = xCI(
        s_test, t_test, pred_risk,
        ipcw=ipcw, s_train=s_train, t_train=t_train,
        g1_bool=g1_bool, g2_bool=g2_bool,
        return_num_valid=True
    )
    
    m2, n2 = xCI(
        s_test, t_test, pred_risk,
        ipcw=ipcw, s_train=s_train, t_train=t_train,
        g1_bool=g2_bool, g2_bool=g1_bool,
        return_num_valid=True
    )
    
    return (m1 * n1 + m2 * n2) / (n1 + n2)


if __name__ == '__main__':
    main()
