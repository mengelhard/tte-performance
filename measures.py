import numpy as np
import pandas as pd


# TODO add bootstrapping to AUC
# TODO add average precision at t
# TODO add xPRt


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


def interpolate(x, y, new_x, method='pad'):
    
    # s = pd.Series(data=y, index=x)
    # new_y = (s
    #     .reindex(s.index.union(new_x).unique())
    #     .interpolate(method=method)[new_x]
    #     .values
    # )
    
    # return new_y

    return np.interp(new_x, x, y)


def xAUCt(s_test, t_test, pred_risk, times, g1_bool=None, g2_bool=None):

    # NOTE: enter groups g1_bool and g2_bool for xAUC_t; omit for AUC_t

    # pred_risk can be 1d (static) or 2d (time-varying)
    if len(pred_risk.shape) == 1:
        pred_risk = pred_risk[:, np.newaxis]

    # positives: s_test = 1 & t_test =< t
    pos = (t_test[:, np.newaxis] <= times[np.newaxis, :]) & s_test[:, np.newaxis]

    if g1_bool is not None:
        pos = pos & g1_bool[:, np.newaxis]
    
    # negatives: t_test > t
    neg = (t_test[:, np.newaxis] > times[np.newaxis, :])

    if g2_bool is not None:
        neg = neg & g2_bool[:, np.newaxis]

    valid = pos[:, np.newaxis, :] & neg[np.newaxis, :, :]
    correctly_ranked = valid & (pred_risk[:, np.newaxis, :] > pred_risk[np.newaxis, :, :])

    return np.sum(correctly_ranked, axis=(0, 1)) / np.sum(valid, axis=(0, 1))


def xROCt(s_test, t_test, pred_risk, time, g1_bool=None, g2_bool=None):

    # NOTE: enter groups g1_bool and g2_bool for xROC_t; omit for ROC_t

    threshold = np.append(np.sort(pred_risk), np.infty)

    # positives: s_test = 1 & t_test =< t
    pos = (t_test < time) & s_test

    if g1_bool is not None:
        pos = pos & g1_bool
    
    # negatives: t_test > t
    neg = (t_test > time)

    if g2_bool is not None:
        neg = neg & g2_bool

    # prediction
    pred = pred_risk[:, np.newaxis] > threshold[np.newaxis, :]

    tpr = np.sum(pred & pos[:, np.newaxis], axis=0) / np.sum(pos)
    fpr = np.sum(pred & neg[:, np.newaxis], axis=0) / np.sum(neg)

    return tpr, fpr, threshold


def ipc_weights(s_train, t_train, s_test, t_test, tau=None):

    if tau == 'auto':
        mask = t_test < t_train[s_train == 1].max()
        #mask = t_test < t_train[s_train == 0].max()
    
    elif tau is not None:
        mask = t_test < tau

    else:
        mask = np.ones_like(t_test, dtype=bool)

    pc = kaplan_meier(1 - s_train, t_train, t_test)
    pc[s_test == 0] = 1.

    w = 1. / pc
    w[~mask] = 0.

    return w


def xCI(s_test, t_test, pred_risk,
        weights=None,
        g1_bool=None, g2_bool=None,
        return_num_valid=False,
        tied_tol=1e-8):

    w = weights if weights is not None else np.ones_like(s_test)

    mask1 = (s_test == 1)

    if g1_bool is not None:
        mask1 = mask1 & g1_bool

    w = w[mask1, np.newaxis]

    mask2 = np.ones_like(s_test, dtype=bool)

    if g2_bool is not None:
        mask2 = mask2 & g2_bool

    valid = t_test[mask1, np.newaxis] < t_test[np.newaxis, mask2]

    risk_diff = pred_risk[mask1, np.newaxis] - pred_risk[np.newaxis, mask2]

    correctly_ranked = valid & (risk_diff > tied_tol)
    tied = valid & (np.abs(risk_diff) <= tied_tol)

    num_valid = np.sum((w ** 2) * valid)
    ci = np.sum((w ** 2) * (correctly_ranked + 0.5 * tied)) / num_valid

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
