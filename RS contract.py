import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import Conf as cf
from math import sqrt
from scipy.special import erf


# Cumulative distribution function for a normal distribution with mean 1000 and standard deviation 200.
def F(y):
    return norm.cdf(y, loc=1000, scale=200)


# Calculates S(q) given a cumulative distribution function F.
def S(q, F):
        integral, error = integrate.quad(F, 0, q)
        return q - integral


# Leftover inventory
def I(q, F):
    return q - S(q, F)


class Contract:
    def __init__(self):
        self.c = cf.c  # unit production cost
        self.c_a = cf.c_a  # unit production cost savings if investment A is made
        self.c_d = cf.c_d  # unit production cost at downstream
        self.p = cf.p  # unit selling price
        self.v = cf.v  # unit salvage value
        self.r = cf.r  # discount value
        self.alpha = cf.alpha   # something
        self.granularity = cf.granularity
        self.granularity_reduction = cf.granularity_reduction
        self.phi_max = cf.phi_granularity

        # Investment cost distribution A
        self.A_min = cf.A_min
        self.A_max = cf.A_max

        # Demand distributions
        self.x = np.linspace(300, 700, 401)
        f = norm.pdf(self.x, cf.mu_d, cf.sigma_d)  # pdf of demand
        self.F = norm.cdf(self.x, cf.mu_d, cf.sigma_d)  # cdf of demand

        self.cumulative_centralized_profit = 0

        # Graphing
        self.q_ax = []
        self.J_I_ax = []
        self.J_NI_ax = []
        self.factor_ax = []
        self.profit_ax = []
        self.deadweight_ax = []
        self.period_ax = []

    # Calculates the expected costs for the supplier
    def supplier_cost(self, q):
        A_star = min((self.c_a * q ** (1/(1-self.alpha))) / 2, self.A_max)          # Highest agreeable value for A
        exp_A = (A_star + self.A_min) / 2                          # E[A | A < A_star]
        delta = (A_star - self.A_min) / (self.A_max - self.A_min)  # Probability for whether an investment is made
        cost = (self.c - delta * self.c_a) * (q ** (1/(1-self.alpha))) / 2 + delta * exp_A
        return max(cost, 0)

    # Calculates the expected revenue for the retailer
    def retailer_revenue(self, q, phi):
        return self.p * S(q, F)  # + self.v * I(q, F)

    def J_s(self, q, T, J, prob, phi):
        cost = self.supplier_cost(q)
        exp_profit_supplier = T - cost + phi * self.retailer_revenue(q, phi)
        return max(exp_profit_supplier + J, 0)

    def J_r(self, q, T, J, prob, phi):
        revenue = self.retailer_revenue(q, phi)
        profit = (1 - phi) * revenue - T
        return (profit + J) * prob

    # Calculate Transfer cost
    def T(self, q, J_s, J_r, phi):
        T_cost = (self.c - self.c_a) * q ** (1/(1-self.alpha)) / 2 - J_s - phi * self.retailer_revenue(q, phi)
        T_q = (T_cost - phi * self.p * S(q, F) + self.A_min + self.retailer_revenue(q, phi)) / 2
        T_max = T_cost + self.A_max
        T_min = T_cost + self.A_min
        T_I = max(T_min, min(T_max, T_q))
        prob_accept = max(0, (
            min(1, (T_I - T_cost + 2 * J_s - self.A_min) / (self.A_max - self.A_min))))

        T_NI = self.c * q ** (1/(1-self.alpha)) / 2 - J_s - phi * self.p * S(q, F)

        J_I = self.J_r(q, T_I, J_r, prob_accept, phi)
        J_NI = self.J_r(q, T_NI, J_r, 1, phi)
        if J_I < J_NI:
            return T_NI, 1, J_I, J_NI
        return T_I, prob_accept, J_I, J_NI

    def centralized_profit(self, q):
        return self.retailer_revenue(q, 0) - self.supplier_cost(q)

    def numerical_max(self, min_q, max_q, step_q, J_s, J_r):
        q_temp, factor_temp, profit_temp, J_I_temp, J_NI_temp = [], [], [], [], []
        best_q, best_T, prob, best_prob, max_profit = 0, 0, 0, 0, -100000
        print(min_q, max_q, step_q, self.alpha)
        for q in range(min_q, max_q, step_q):
            for phi in range(0, self.phi_max):
                calc_T, prob, J_I, J_NI = self.T(q, J_s, J_r, phi/self.phi_max)
                profit = self.J_r(q, calc_T, J_r, prob, phi/self.phi_max)
                if q == 497:
                    print(phi/self.phi_max, calc_T, profit)
                if profit > max_profit:
                    max_profit = profit
                    best_q, best_T, best_prob = q, calc_T, prob
                    best_phi = phi/self.phi_max

            q_temp.append(q)
            factor_temp.append(self.alpha)
            profit_temp.append(profit)
            J_I_temp.append(J_I)
            J_NI_temp.append(J_NI)

        if step_q == self.granularity:
            print("Best q:", best_q)
            print("Best T:", best_T)
            print("Best phi:", best_phi)
            print("Best prob:", best_prob)
            print("Maximum profit:", max_profit)
            print("Cost to go retailer and supplier:", J_r, J_s)
            return best_q, best_T, best_prob, best_phi, q_temp, factor_temp, profit_temp, J_I_temp, J_NI_temp

        return self.numerical_max(max(0, best_q - step_q), best_q + step_q, step_q//self.granularity_reduction, J_s, J_r)

    def numerical_max_c(self, min_q, max_q, step_q):
        best_q, max_profit_centralized = 0,  -100000
        for q in range(min_q, max_q, step_q):
            profit_c = self.centralized_profit(q)
            if profit_c > max_profit_centralized:
                max_profit_centralized = profit_c
                best_q = q

        if step_q == self.granularity:
            print("Best q_c:", best_q)
            print("Maximum profit integrated:", max_profit_centralized)
            print()
            return max_profit_centralized
        return self.numerical_max_c(max(0, best_q - step_q), best_q + step_q, step_q // self.granularity_reduction)

    def dyn_prog(self, min_q, max_q, step_q, J_s, J_r, period_temp, deadweight_temp):
        q, T, prob, phi, q_temp, factor_temp, profit_temp, J_I_temp, J_NI_temp = self.numerical_max(min_q, max_q, step_q, J_s, J_r)
        J_s_1 = self.J_s(q, T, J_s, prob, phi) * np.exp(-cf.r)
        J_r_1 = self.J_r(q, T, J_r, prob, phi) * np.exp(-cf.r)
        self.cumulative_centralized_profit = (self.cumulative_centralized_profit + self.numerical_max_c(min_q, max_q, step_q)) * np.exp(-cf.r)
        print(self.cumulative_centralized_profit, J_s_1, J_r_1)
        deadweight_loss = (self.cumulative_centralized_profit - (J_s_1 + J_r_1))/self.cumulative_centralized_profit
        period_temp.append(int(-(len(deadweight_temp))))
        deadweight_temp.append(deadweight_loss)
        print('deadweight loss:', deadweight_temp)
        print()
        if (abs(J_s_1 - J_s)/np.exp(-cf.r))/J_s_1 < cf.buffer and (abs(J_r_1 - J_r)/np.exp(-cf.r))/J_r_1 < cf.buffer:
            self.q_ax.append(q_temp)
            # self.factor_ax.extend(factor_temp)
            self.profit_ax.append(profit_temp)
            self.J_I_ax.append(max(J_I_temp))
            self.J_NI_ax.append(max(J_NI_temp))
            self.period_ax.append(period_temp)
            self.deadweight_ax.append(deadweight_loss)
            # self.plot_deadweight()
            return J_r_1, J_s_1
        return self.dyn_prog(min_q, max_q, step_q, J_s_1, J_r_1, period_temp, deadweight_temp)

    def analysis(self, min_q, max_q, step_q, J_s, J_r, factors):

        for factor in factors:
            period_temp = []
            deadweight_temp = []
            self.integrated_profit = 0
            self.alpha = factor
            self.dyn_prog(min_q, max_q, step_q, J_s, J_r, period_temp, deadweight_temp)
        #self.plot_deadweight(factors)
        self.plot_alpha(factors)

    def clear_plot(self):
        self.q_ax = []
        self.J_I_ax = []
        self.J_NI_ax = []
        self.factor_ax = []

    def plot_deadweight(self):
        #for i in range(len(factors)):
        #    name = "Deadweight loss, alpha = " + str(factors[i] / 100)
        #    plt.scatter(self.period_ax[i], self.deadweight_ax[i], label=name)
        plt.scatter(self.period_ax[0], self.deadweight_ax[0], label="")
        plt.xlabel("Period")
        plt.ylabel("Deadweight Loss")
        plt.legend()
        plt.show()

    def plot_alpha(self, factors):
        #for i in range(len(factors)):
         #   name1 = "Investment made, alpha = " + str(factors[i]/100)
          #  name2 = "No investment made, alpha = " + str(factors[i]/100)
           # plt.scatter(self.factors[i]/100, self.J_I_ax[i], label=name1)
            #plt.scatter(self.q_ax[i], self.J_NI_ax[i], label=name2)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(self.q_ax, self.factor_ax, self.profit_ax,)
        # plt.scatter(self.q_ax, self.J_I_ax, label="J_I")
        # plt.scatter(self.q_ax, self.J_NI_ax, label="J_NI")
        plt.scatter(factors, self.J_I_ax, label='J_I')
        plt.scatter(factors, self.J_NI_ax, label='J_NI')
        plt.xlabel("alpha")
        plt.ylabel("profit")
        plt.legend()
        plt.show()


connie = Contract()
connie.dyn_prog(300, 800, 100, 0, 0, [], [])
factors = [x/100 for x in range(40, 60)]
#connie.analysis(0, 2000, 100, 0, 0, factors)

