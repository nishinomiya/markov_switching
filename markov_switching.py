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
    xittpo1[-1], xittpo2[-1] =  ergodic_probab, ergodic_probab

    for i in range(ts.shape[0]):
        eta1, eta2 = eta(dec(ts[i]), dec(regime1), dec(err_std_dev)), eta(dec(ts[i]), dec(regime2), dec(err_std_dev))
        bt = dec(xittpo1[i - 1]) * dec(eta1) + dec(xittpo2[i - 1]) * dec(eta2)
        xitt1, xitt2 = (dec(xittpo1[i - 1]) * dec(eta1)) / dec(bt), (dec(xittpo2[i-1]) * dec(eta2)) / dec(bt)
        xittpo1[i] = dec(xitt1) * dec(tp1000) + dec(xitt2) * (dec(1.0) - dec(tp0001))
        xittpo2[i] = dec(xitt2) * dec(tp0001) + dec(xitt1) * (dec(1.0) - dec(tp1000))
    return xittpo1

def normalization(x):
    x -= np.nanmin(x)
    return x / np.nanmax(x)

rc = np.random.choice([-1,1], size=1000)
initial_price = 100
grw = np.cumprod(np.exp(rc * 0.01)) * initial_price
grwalk = pd.Series(grw)
log_returns = np.log(grwalk) - np.log(grwalk.shift(1))
log_returns[0] = 0

log_returns = normalization(log_returns)
f = np.percentile(log_returns, 90)
s = np.percentile(log_returns, 10)
xittpo = markov_switch(log_returns, 8, 8, f, s, np.mean(lr)*1.5)

nprice = normalization(grw)
plt.figure(figsize=(16, 4), dpi=100)
plt.plot(xittpo)
plt.plot(nprice)
plt.show()
