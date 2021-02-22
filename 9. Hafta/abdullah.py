# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:35:43 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import numpy as np

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 100)
print("Mean Difference:",abs(mu - np.mean(s)))
print("STD Difference:",abs(sigma - np.std(s, ddof=1)))
"""
The standard deviation is the square root of the average of the squared deviations from the mean, i.e., 
std = sqrt(mean(x)), where x = abs(a - a.mean())**2.

The average squared deviation is typically calculated as x.sum() / N, where N = len(x). 
If, however, ddof is specified, the divisor N - ddof is used instead. In standard statistical practice, 
ddof=1 provides an unbiased estimator of the variance of the infinite population. 
ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables. 
The standard deviation computed in this function is the square root of the estimated variance, 
so even with ddof=1, it will not be an unbiased estimate of the standard deviation per se.

Reference: https://numpy.org/doc/stable/reference/generated/numpy.std.html
"""