from space_elevator import *
from math import pi
import numpy as np
from matplotlib import pyplot as plt

t = UniformStressTether()
# pts = [lc.L*i/100 for i in range(101)]
# plt.plot(pts, [lc.rho(x) for x in pts])
# plt.show()
#
# print(f'{lc.M:.4}')
# plt.plot(pts, [lc.P(x) for x in pts])
# plt.show()
sp = t.spectrum()
for i in range(0, 21, 5):
    print(f't_eq[{i}] = {sp.period_eq(i,"h")}, t_mer[{i}] = {sp.period_mer(i,"h")}')
    sp.show_mode(i)

#print(f'{2*pi/np.array(find_eigenvals(lc, 10))/3600}')

