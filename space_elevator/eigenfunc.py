from scipy.integrate import *
from scipy.optimize import root_scalar, RootResults
from scipy.misc import derivative
from math import exp, sin, cos, atan, pi
from .planet import Earth
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt


class InvalidConfigException(Exception):
    def __init__(self, msg):
        super().__init__(f'Invalid space elevator configuration: {msg}')


class SETether(ABC):
    """ Base class for space lift configurations """
    def __init__(self, L, planet=Earth):
        self.L = L
        self.planet = planet
        self.R = planet.radius
        self.ell = self.R + L
        self.M = self.counterweight()
        if self.M < 0:
            raise InvalidConfigException('counterweight < 0')
        if self.P(0)<0 or self.P(self.L)<0:
            raise InvalidConfigException('tether tension < 0')
        self.Mt = quad(lambda s: self.rho(s), 0, self.L)[0]
        self._p = None
        self.Z = quad(lambda s: (self.rho(s)*self.P(s))**0.5, 0, self.L)[0]

    def _P0(self, s) -> float:
        if self._p is None:
            p0 = self.planet.F(self.ell)*self.M
            self._p = solve_ivp(lambda s, _: self.rho(self.L-s)*self.planet.F(self.ell-s), (0, self.L), y0=[p0],
                                atol=p0*1e-8, rtol=1e-8, dense_output=True).sol

            if p0 < 0 or self._p(self.L) <= 0:
                # print('Warning: Tension is negative')
                raise InvalidConfigException("Tension is negative")

        return self._p(self.L - s)

    @abstractmethod
    def counterweight(self) -> float:
        """ Counterweight at the end of tether (kg) """
        pass

    @abstractmethod
    def rho(self, s) -> float:
        """ Linear density of tehter at given point (kg/m) """
        pass

    def _log_rho_der(self, s) -> float:
        """ Derivative of linear density of tehter at given point (kg/m^2) """
        return derivative(self.rho, s)/self.rho(s)

    def P(self, s) -> float:
        """ Tension of tehter at given point (kg*m/s^2) """
        return self._P0(s)

    def phiL(self, s_lamb, whole_func=False, atol=1e-12, rtol=1e-7):
        def f(s, phi):
            r = s+self.R
            dlog = 1/r + 0.25*(self._log_rho_der(s) - self.rho(s)*self.planet.F(r)/self.P(s))
            return s_lamb*(self.rho(s)/self.P(s))**0.5 + dlog * sin(2*phi)

        res = solve_ivp(f, t_span=(0, self.L), y0=[0], atol=atol, rtol=rtol, dense_output=whole_func)

        if whole_func:
            return res.sol
        return float(res.y[0, -1])

    def calc_mode(self, s_lamb, atol=1e-12, rtol=1e-7):
        # phi = self.phiL(s_lamb, whole_func=True, atol=atol, rtol=rtol)
        def f(s, y):
            q, phi = y
            r = s+self.R
            dlog = 2/r + 0.5*(self._log_rho_der(s) - self.rho(s)*self.planet.F(r)/self.P(s))
            return [-dlog*cos(phi)**2, s_lamb*(self.rho(s)/self.P(s))**0.5 + 0.5*dlog * sin(2*phi)]

        r = solve_ivp(f, t_span=(0, self.L), y0=[0, 0], atol=atol, rtol=rtol, dense_output=True).sol
        def v0(s):
            lr, phi = r(s)
            return exp(lr)*sin(phi)*(self.R + s)
        wt = quad(lambda s: v0(s)**2*self.rho(s), 0, self.L, limit=10000)[0] + v0(self.L)**2*self.M
        coef = ((self.M+self.Mt)/wt)**0.5
        return lambda s: v0(s)*coef


    def eL(self, s_lamb):
        return pi/2 if s_lamb <= 0 else atan((self.P(self.L)*self.rho(self.L))**0.5/(self.M * s_lamb))

    def eq(self, n, **kwargs):
        return lambda s_lamb: self.phiL(s_lamb, **kwargs) - self.eL(s_lamb) - pi*n

    def spectrum(self, n=0, **kwargs):
        return TetherSpectrum(self, n, **kwargs)


class TetherSpectrum:
    def __init__(self, tc: SETether, n, **kwargs):
        self.tether = tc
        self._params = kwargs
        self._params.setdefault('verbose', 0)
        self.ev = find_eigenvals(tc, n, **kwargs) if n else []
        self.ef = {}

    def __len__(self):
        return len(self.ev)

    def __getitem__(self, n):
        if n >= len(self.ev):
            self.ev += find_eigenvals(self.tether, max(len(self)*2,n+1), start=len(self), **self._params)
        return self.ev[n]

    def eigen_mode(self, i):
        if i not in self.ef:
            self.ef[i] = self.tether.calc_mode(self[i])
        return self.ef[i]

    def period_eq(self, i, units='s'):
        return 2*pi/self[i]/{'s': 1, 'min': 60, 'h': 3600, 'd': 3600*24}[units]

    def period_mer(self, i, units='s'):
        return 2*pi/(self.tether.planet.omega**2+self[i]**2)**0.5/{'s': 1, 'min': 60, 'h': 3600, 'd': 3600*24}[units]

    def show_mode(self, i):
        n = max(100, 20*(i+1))
        em = self.eigen_mode(i)
        x = [self.tether.L*j/n for j in range(n+1)]
        plt.plot(x, [em(t) for t in x])
        plt.title(f'{i}-th eigen mode')
        plt.grid(which='major', linestyle='dashed', color='#888')
        plt.grid(which='minor', linestyle='dotted', color='#888')
        plt.show()

class UniformStressTether(SETether):
    __slots__ = ['L', 'tau', 's_tau', 'ell', 'M', 'Mt', 'R', 'planet']

    def __init__(self, /, L=80e6, tau=3e+7, rho0=1, rho_add=0, planet=Earth):
        self.tau = tau
        self.s_tau = tau**0.5
        self._rho_add = rho_add
        self._s0 = rho0+rho_add
        super().__init__(L, planet)
        if self.rho(self.L) < self._rho_add:
            raise Exception("additional load is too big for this length")

    def _log_rho_der(self, s):
        return -self._s0*self.planet.F(self.R+s)/self.tau

    def rho(self, s):
        return self._s0*exp((self.planet.U(self.R+s)-self.planet.U(self.R))/self.tau)

    def counterweight(self) -> float:
        return self.tau*(self.rho(self.L)-self._rho_add)/self.planet.F(self.ell)

    def P(self, s):
        return (self.rho(s)-self._rho_add)*self.tau

    # def phiL(self, s_lamb) -> float:
    #     mu = self.planet.mu
    #     om = self.planet.omega
    #
    #     def f(s, phi):
    #         r = s+self.R
    #         return s_lamb/self.s_tau + (1/r+0.5*(mu/r**2 - om**2*r)/self.tau) * sin(2*phi)
    #
    #     res = solve_ivp(f, t_span=(0, self.L), y0=[0], atol=1e-12, rtol=1e-7)
    #     if res.t[-1]!=self.L:
    #         raise Exception(f'res[-1] = {res[-1]} != L={self.L}')
    #     return float(res.y[0, -1])

    # def eL(self, s_lamb):
    #     return pi/2 if s_lamb<=0 else atan(self.rho(self.L)*self.s_tau/(self.M * s_lamb))

    # def eq(self, n):
    #     return lambda s_lamb: self.phiL(s_lamb) - self.eL(s_lamb) - pi*n


def solve(f, x0, e, xtol=None, rtol=1e-5) -> RootResults: # function f grows
    f0 = f(x0)
    l = r = x0
    if f0>0:
        while f(x0-e)>0:
            e *= 2
        l = x0-e
    else:
        while f(x0+e) < 0:
            e *= 2
        r = x0+e
    return root_scalar(f, method='secant', bracket=(l, r), x0=l, x1=r, xtol=xtol, rtol=rtol)


def find_eigenvals(lc: SETether, n, verbose=1, find_modes=False, start=0, **kwargs):
    res = []
    curr = 0
    e = d = pi/lc.Z
    for i in range(start, n):
        sol = solve(lc.eq(i), curr, e, **kwargs)
        if verbose >= 2:
            print(f'Eigenvalue {i}: {sol}')
        elif verbose >= 1:
            print(f'Eigenvalue {i}: {sol.root:10.8f}, {sol.iterations} iterations')
        x = sol.root
        e = abs(x - curr)
        curr = x + d
        res.append(x)
    if find_modes:
        return res, [lc.phiL(s_l, whole_func=True) for s_l in res]
    return res
