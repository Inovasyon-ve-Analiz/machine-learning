from numpy import random

def get_mean(numbers):
    total = 0
    for n in numbers:
        total += n
    mean = total/len(numbers)
    return mean

def get_deviation(numbers):
    result = 0
    mean = get_mean(numbers)
    for n in numbers:
        result += (mean-n)**2
    result=(result/len(numbers))**0.5
    return result

means = [0,1,5]
deviations = [1,5,10]
for m in means:
    for d in deviations:
        numbers = random.normal(loc=m, scale=d, size = 100)
        mean = get_mean(numbers)
        deviation = get_deviation(numbers)
        print(str(m)+" "+str(round(mean,3)))
        print(str(d)+" "+str(round(deviation,3)))
        print("")