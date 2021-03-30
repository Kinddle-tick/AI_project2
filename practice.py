# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

male=[173,172,179,167,169,176,173,170,180,167,171,167,184,183,177,169,188,183,171,172,
      165,164,174,174,178,181,175,174,171,168]
female=[162,148,151,179,161,153,166,158,164,152,170,163,164,160,146,162,163,162,158,159]


def rect(x,data,h):
      rtn = []
      for i in x:
            tmp=(data-i)/h
            rtn.append(np.sum(np.abs(tmp)<0.5)/len(data)/h**1)
      return rtn
def norm(x,data,h):
      rtn = []

      for i in x:
            tmp = (data - i) / h
            rtn.append(np.sum(stats.norm.pdf(tmp)) / len(data) / h ** 1)
      return rtn

def quick(h):
      y_rect_male = rect(x, male, h)
      y_rect_female = rect(x, female, h)
      y_norm_male = norm(x, male, h)
      y_norm_female = norm(x, female, h)
      return y_rect_male,y_rect_female,y_norm_male,y_norm_female


x = np.linspace(min(male+female),max(male+female),100)


h=7

y_rect_male = rect(x,male,h)
y_rect_female = rect(x,female,h)
y_norm_male = norm(x,male,h)
y_norm_female = norm(x,female,h)

male_E=np.average(male)
male_var = np.sqrt(np.var(male))
female_E = np.average(female)
female_var = np.sqrt(np.var(female))
y_male = stats.norm.pdf(x,male_E,male_var)
y_female = stats.norm.pdf(x,female_E,female_var)

# plt.hist(np.array(male)/h/len(male),color="black",alpha = 0.5)
# plt.hist(np.array(female)/h/len(female),color="green", alpha = 0.5)

plt.plot(x,y_norm_male,c="blue",linestyle=":",label='norm_male')
plt.plot(x,y_norm_female,c="red",linestyle=":",label='norm_female')
plt.plot(x,y_rect_male,c="cyan",linestyle="-.",label='rect_male')
plt.plot(x,y_rect_female,c="pink",linestyle="-.",label='rect_female')
plt.plot(x,y_male,c="black",linestyle="-",label='male')
plt.plot(x,y_female,c="m",linestyle="-",label='female')
plt.legend(loc='upper right')
# plt.plot(x,y1*150,c="red")
#
# plt.xlabel("height/cm")
# plt.ylabel("P*150 ")
plt.show()

plt.cla()
for i in np.linspace(1,15,30*5):
      y_rect_male,y_rect_female,y_norm_male,y_norm_female = quick(i)
      plt.plot(x, y_norm_male, c=(i/32,1,i/16), linestyle="-", label=f'male_{i}')
      plt.plot(x, y_norm_female, c=(0,i/32,i/16), linestyle="-", label=f'female_{i}')
# plt.legend(loc='lower right')
plt.plot(x,y_male,c="green",linestyle="-",label='male')
plt.plot(x,y_female,c="m",linestyle="-",label='female')
plt.show()

plt.cla()
for i in np.linspace(1,15,30*5):
      y_rect_male,y_rect_female,y_norm_male,y_norm_female = quick(i)
      plt.plot(x, y_rect_male, c=(i/32,1,i/16), linestyle="-", label=f'male_{i}')
      plt.plot(x, y_rect_female, c=(0,i/32,i/16), linestyle="-", label=f'female_{i}')
# plt.legend(loc='lower right')
plt.plot(x,y_male,c="green",linestyle="-",label='male')
plt.plot(x,y_female,c="m",linestyle="-",label='female')
plt.show()





