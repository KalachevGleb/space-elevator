G = 6.6743015e-11

class Planet:
    __slots__ = ['mass', 'radius', 'omega', 'mu', 'geo_radius']

    def __init__(self, m, r, om):
        self.mass = m
        self.radius = r
        self.omega = om
        self.mu = G * m
        self.geo_radius = (self.mu/self.omega**2)**(1/3.)

    def U(self, r):
        return -self.mu / r - (self.omega * r) ** 2 / 2

    def F(self, r):
        return self.omega ** 2 * r - self.mu / r ** 2


Earth = Planet(5.972e24, 6378e+3, 7.292e-5)
Moon = Planet(7.346e22, 1.738e6, 2.6617e-6)
Mars = Planet(6.4171e23, 3.3962e6, 7.08821e-5)
