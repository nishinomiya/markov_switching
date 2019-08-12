import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import  pi
from decimal import *

def markov_switch(ts, p11, p22, regime1, regime2, err_std_dev, ergodic_probab=0.5):

    dec = lambda x: Decimal(str(x))
    eta = lambda y, regime, esd:  dec(1.0) / (dec(2.0) * dec(pi) * dec(esd)**dec(2.0)).sqrt() * \
                  dec(-((dec(y) - dec(regime))**dec(2.0) / (dec(2.0) * dec(esd)**dec(2.0)))).exp()

    tp1000, tp0001 = dec(p11).exp() / (dec(1.0)+ dec(p11).exp()), dec(p22).exp()/(dec(1.0) + dec(p22).exp())
    tp0100, tp0010 = dec(1.0) - dec(tp1000), dec(1.0) - dec(tp0001)
    xittpo1, xittpo2 = np.zeros_like(ts), np.zeros_like(ts)

    eta1, eta2 = eta(dec(ts[0]), dec(regime1), dec(err_std_dev)), eta(dec(ts[0]), dec(regime2), dec(err_std_dev))
    bt = dec(ergodic_probab) * dec(eta1) + (dec(1.0) - dec(ergodic_probab)) * dec(eta2)
    xitt1,  xitt2 = (dec(ergodic_probab) * dec(eta1)) / dec(bt), ((dec(1.0) - dec(ergodic_probab)) * dec(eta2)) / dec(bt)
    xittpo1[0] = dec(xitt1) * dec(tp1000) + dec(xitt2) * (dec(1.0) - dec(tp0001))
    xittpo2[0] = dec(xitt2) * dec(tp0001) + dec(xitt1) * (dec(1.0) - dec(tp1000))

    for i in range(1, ts.shape[0]):
        eta1, eta2 = eta(dec(ts[i]), dec(regime1), dec(err_std_dev)), eta(dec(ts[i]), dec(regime2), dec(err_std_dev))
        bt = dec(xittpo1[i - 1]) * dec(eta1) + dec(xittpo2[i - 1]) * dec(eta2)
        xitt1, xitt2 = (dec(xittpo1[i - 1]) * dec(eta1)) / dec(bt), (dec(xittpo2[i-1]) * dec(eta2)) / dec(bt)
        xittpo1[i] = dec(xitt1) * dec(tp1000) + dec(xitt2) * (dec(1.0) - dec(tp0001))
        xittpo2[i] = dec(xitt2) * dec(tp0001) + dec(xitt1) * (dec(1.0) - dec(tp1000))
    return xittpo1

def normalization(x):
    x -= np.nanmin(x)
    return x / np.nanmax(x)

def log_returns(ts):
    assert 0 not in ts, 'divide by zero'
    log_return_data = np.log(ts[1:] / ts[:-1])
    return np.r_[[0.0], log_return_data]

dn = np.random.choice([-1,1], size=500)
initial_price = 100
gw = np.cumprod(np.exp(dn * 0.01)) * initial_price
grwalk = pd.Series(gw)
grwalk.plot()

lr = log_returns(grwalk.values)
lr = normalization(lr)
f = np.percentile(lr, 90)
s = np.percentile(lr, 10)
errstddev = np.mean(lr)

nprice = normalization(grwalk.values)
xittpo = markov_switch(lr, 8, 8, f, s, errstddev*1.5)
plt.figure(figsize=(16, 4), dpi=100)
plt.plot(xittpo)
plt.plot(nprice)
plt.show()
