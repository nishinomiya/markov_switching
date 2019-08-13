import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import  pi
from decimal import *

def markov_switch(ts, p11, p22, regime1, regime2, err_std_dev, ergodic_probab=0.5):
    dec = lambda x: Decimal(str(x))
    eta = lambda y, regime, esd:  dec(1.0) / (dec(2.0) * dec(pi) * dec(esd)**dec(2.0)).sqrt() * \
                  dec(-((dec(y) - dec(regime))**dec(2.0) / (dec(2.0) * dec(esd)**dec(2.0)))).exp()

    tp1000 = dec(p11).exp() / (dec(1.0) + dec(p11).exp())
    tp0001 = dec(p22).exp() / (dec(1.0) + dec(p22).exp())
#    tp0100 = dec(1.0) - dec(tp1000) #spare for expansion
#    tp0010 = dec(1.0) - dec(tp0001) #spare for expansion
    xprobab1, xprobab2 = np.zeros_like(ts), np.zeros_like(ts)
    xprobab2[-1], xprobab2[-1] =  ergodic_probab, ergodic_probab

    for i in range(ts.shape[0]):
        eta1 = eta(ts[i], regime1, err_std_dev)
        eta2 = eta(ts[i], regime2, err_std_dev)
        bt = dec(xprobab1[i - 1]) * dec(eta1) + dec(xprobab2[i - 1]) * dec(eta2)
        xit1 = dec(xprobab1[i - 1]) * dec(eta1) / dec(bt)
        xit2 = dec(xprobab2[i - 1]) * dec(eta2) / dec(bt)
        xprobab1[i] = dec(xit1) * dec(tp1000) + dec(xit2) * (dec(1.0) - dec(tp0001))
        xprobab2[i] = dec(xit2) * dec(tp0001) + dec(xit1) * (dec(1.0) - dec(tp1000))
    return xprobab1

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
xprobab = markov_switch(log_returns, 8, 8, f, s, np.mean(log_returns)*1.5)

nprice = normalization(grw)
plt.figure(figsize=(16, 4), dpi=100)
plt.plot(xprobab)
plt.plot(nprice)
plt.show()
