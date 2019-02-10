import numpy as np
from math import log, sqrt, exp, fabs, pi
from scipy.stats import stats

''' ==============================================================
compute a Markov Switching 
'''
def moving_average(prices, n, type='simple'):
  prices = np.asarray(prices)
  if type=='simple':
    weights = np.ones(n)
  else:
    weights = np.exp(np.linspace(-1., 0., n))
  weights /= weights.sum()
  ma =  np.convolve(prices, weights, mode='full')[:len(prices)]
  ma[:n] = ma[n]
  return ma

"""
markov_regression
"""
def markov_switch(ts, p11, p22, regime1, regime2, err_std_dev, ergodic_probab=0.5):
  eta = lambda x, regime, esd: 1 / sqrt(2 * pi * pow(esd, 2)) * \
                             exp( -(pow((x - regime), 2) / (2 * pow(esd, 2))))
  tp1000, tp0001 = exp(p11) / (1 + exp(p11)), exp(p22) / (1 + exp(p22))
  tp0100, tp0010 = 1 - tp1000, 1 - tp0001
  xittpo1 = np.zeros_like(ts)
  xittpo2 = np.zeros_like(ts)
  for i in range(ts.shape[0]):
    eta1 = eta(ts[i], regime1, err_std_dev)
    eta2 = eta(ts[i], regime2, err_std_dev)
    if i == 0:
      bt = ergodic_probab * eta1 + (1- ergodic_probab) * eta2 #ln = log(bt)
      if bt == 0: bt=0.0001
      xitt1 = (ergodic_probab * eta1) / bt
      xitt2 = ((1 - ergodic_probab) * eta2) / bt
    else:
      bt = xittpo1[i-1] * eta1 + xittpo2[i-1] * eta2
      xitt1 = (xittpo1[i-1] * eta1) / bt
      xitt2 = (xittpo2[i-1] * eta2) / bt
    xittpo1[i] = xitt1 * tp1000 + xitt2 * (1 - tp0001)
    xittpo2[i] = xitt2 * tp0001 + xitt1 * (1 - tp1000)
  return xittpo1

