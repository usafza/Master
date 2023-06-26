import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import Conf as cf
from math import sqrt
from scipy.special import erf


# Cumulative distribution function for a normal distribution with mean 1000 and standard deviation 200.
def F(y):
    print("F:", 0.5 * (1 + erf((y - cf.mu_d) / (cf.sigma_d * sqrt(2)))))
    return 0.5 * (1 + erf((y - cf.mu_d) / (cf.sigma_d * sqrt(2))))  # 0.5 * (1 + erf(y / sqrt(2)))


# Calculates S(q) given a cumulative distribution function F.
def S(q, F):
    lower_limit = cf.mu_d - q * cf.sigma_d
    integral, error = integrate.quad(F, lower_limit, cf.mu_d)  # integrate.quad(F, 0, q)
    return q - integral


# Leftover inventory
def I(q, F):
    return q - S(q, F)


# Transfer cost
# def T(q):
#     return 0

class Contract:
    def __init__(self):
        self.c = cf.c  # unit production cost
        self.c_a = cf.c_a  # unit production cost savings if investment A is made
        self.c_r = cf.c_r  # unit production cost at retailer
        self.p = cf.p  # unit selling price
        self.v = cf.v  # unit salvage value
        self.w = cf.w  # wholesale price
        self.r = cf.r  # discount value

        # Investment cost distribution A
        self.A_min = 10000
        self.A_max = 55000

        # Demand distributions
        self.x = np.linspace(300, 700, 401)
        f = norm.pdf(self.x, cf.mu_d, cf.sigma_d)  # pdf of demand
        self.F = norm.cdf(self.x, cf.mu_d, cf.sigma_d)  # cdf of demand


    # Highest agreeable value for A
    def A_star(self, q):
        return (self.c_a * q ^ 2) / 2

    # E[A | A < A_star]
    def exp_A(self, q):
        return (self.A_star(q) + self.A_min) / 2

    # Boolean for whether investment is made
    def delta(self, q):
        return 1 if self.exp_A(q) <= self.A_star(q) else 0

    # Get prob(X < value)
    def get_cdf(self, cdf, value):
        i, = np.where(np.isclose(self.x, value))
        return cdf[i][0]

    def revenue_retailer(self, q):
        return self.p * S(q, F) + self.v * I(q, F)

    def exp_profit_retailer(self, q, T):
        #print("exp_profit:", (self.revenue_retailer(q) - self.c_r * q - T) \
        #* ((T - (self.c - self.c_r) * (q ** 2) / 2 - self.A_min) / (self.A_max - self.A_min)))
        #print("exp_A:", ((T - (self.c - self.c_r) * (q ** 2) / 2 - self.A_min) / (self.A_max - self.A_min)))
        return (self.revenue_retailer(q) - self.c_r * q - T) \
               * ((T - (self.c - self.c_r)*(q**2)/2 - self.A_min) / (self.A_max - self.A_min))

    def numerical_max(self, min_q, max_q, step_q, min_T, max_T, step_T):
        best_q, best_T = None, None
        x_points = np.array(range(min_q, max_q, step_q))
        q_ax = []
        y_ax = []
        t_ax = []
        max_profit = - 10
        for q in range(min_q, max_q, step_q):
            if (q % 100 == 0):
                print(q)
            for T in range(min_T, max_T, step_T):
                profit = self.exp_profit_retailer(q, T)
                # print(profit,max_profit)
                if profit > max_profit:
                    max_profit = profit
                    best_q, best_T = q, T
                y_ax.append(max(0, profit))
                t_ax.append(T)
                q_ax.append(q)
        q_points = np.array(q_ax)
        t_points = np.array(t_ax)
        y_points = np.array(y_ax)
        ax = plt.axes(projection='3d')
        ax.scatter3D(q_points, t_points, y_points)
        # plt.plot(x_points, y_points)
        plt.show()
        print("Best q:", best_q)
        print("Best T:", best_T)
        print("Maximum profit:", max_profit)
        # print((T - (cf.c - cf.c_r) / 2 * (q ** 2) - self.A_min) / (self.A_max - self.A_min))


    # Variable cost supplier
    def var_c(self, q):
        return (self.c - self.delta(q) * self.c_a) * (q ^ 2) / 2

    def cost_supplier(self, q):
        return self.var_c(q) + self.delta(q) * integrate.quad(lambda A: A, self.A_min, self.A_max) / \
               (self.A_max - self.A_min)

    def exp_profit_supplier(self, q):
        return T(q) - self.cost_supplier(q)


    def value_to_go(self, j, q, k, total):
        j_next = self.exp_profit_supplier(q) * np.exp(-cf.r * k)
        print(j_next)
        # if abs(j_next - j)/cf.r < cf.buffer:
        print("buffer:", abs(j_next - j) / cf.r)
        if k == 3:
            return 1 if total >= 0 else 0

        print(q, k, total)
        return self.value_to_go(j_next, q, k + 1, total + j_next)




connie = Contract()
connie.numerical_max(400, 600, 20, 10000, 20000, 5000)

