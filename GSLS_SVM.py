import numpy as np
import sys

class GSLS_Regression:
    def __init__(self, Kernel, gamma, n = 10):
        self.kernel = Kernel
        self.gamma = gamma

        #self.w = np.array([0])
        self.b = 0.0
        self.betta = np.array([0])
        self.S = np.array([])
        self.x = np.array([])

        self.n = n


    def objective_function(self, S, x, y):
        result = 0.0
        for i in range(S.size):
            for j in range(S.size):
                result += 0.5 * self.betta[i] * self.betta[j] * self.kernel(x[S[i]], x[S[j]])
        
        l =  y.size
        for i in range(l):
            subresult = y[i]
            for j in range(S.size):
                subresult -= self.betta[j] * self.kernel(x[i], x[S[j]])
            subresult -= self.b
            result += self.gamma / l * subresult ** 2
    
        return result
    
    def new_omega_vector(self, S, x):
        omega = np.array([[0]] * (S.size - 1))
        l = np.size(x, 0)
        for i in range((S.size - 1)):
            omega[i, 0] = l / (2 * self.gamma) * self.kernel(x[S[i]], x[S[-1]])
            for r in range(l):
                omega[i, 0] += self.kernel(x[r], x[S[-1]]) * self.kernel(x[r], x[S[i]])
        return omega

    
    def new_diag_omega_elem(self, s: int, x, y):
        l = np.size(x, 0)
        omega = l / (2 * self.gamma) * self.kernel(x[s], x[s])
        for r in range(l):
            omega += self.kernel(x[r], x[s]) * self.kernel(x[r], x[s])
        return omega


    def new_phi_elem(self, s, x):
        l = np.size(x, 0)
        phi = 0
        for j in range(l):
            phi += self.kernel(x[s], x[j])
        return phi

    def init_c(self, S, x, y):
        l = y.size
        c = np.array([])
        for i in S:
            new_c = 0
            for j in range(l):
                new_c += y[j] * self.kernel(x[i], x[j])
            c = np.append(c, new_c)
        return c
    

    # [A  b]^ = [A^ + 1/k * A^ * b*b' * A^     -1/k * A^ * b]
    # [b' c]    [      -1/k * b' * A^               1/k     ]
    # k = c - b' * A^ * b
    def inverse_block_matrix(self, inverse_A, b, c):
        k = c - np.matmul(np.matmul(np.transpose(b), inverse_A), b)
        new_A = inverse_A + 1/k * np.matmul(np.matmul(np.matmul(inverse_A, b), np.transpose(b)), inverse_A)
        new_b = -1/k * np.matmul(inverse_A, b)
        new_b_t = -1/k * np.matmul(np.transpose(b), inverse_A)
        return np.hstack([np.vstack([new_A, new_b_t]), np.vstack([new_b, 1/k])])


    def fit(self, x, y):
        S = np.array([0], dtype=np.int32)
        omega = np.array([[]])
        l = y.size
        H = np.array([[l]])
        inversive_H = np.array([[1/l]])
        opt_inversive_H = np.array([[1/l]])
        objective = np.array([])
        #init Y
        Y = np.sum(y)
        #first iteration
        index = 0
        obj = sys.float_info.max
        phi = 0
        for i in range(l):
            S[-1] = i
            phi = np.array([[self.new_phi_elem(i, x)]])
            om = self.new_diag_omega_elem(i, x, y)
            new_inverse_H = self.inverse_block_matrix(inversive_H, phi, om)
            #H*[b, betta] = [Y, c]
            right = np.append(Y, self.init_c(S, x, y))
            solve = np.matmul(new_inverse_H, right)

            self.betta = solve[1:]
            self.b = solve[0]

            #new_H = np.hstack([np.vstack([l, phi]), np.vstack([phi, om])])
            #print(new_H @ solve - right)

            new_obj = self.objective_function(S, x, y)
            #print(self.betta, self.b, S, new_obj)
            if (obj > new_obj):
                index = i
                obj = new_obj
                opt_inversive_H = np.copy(new_inverse_H)
                #opt_H = np.hstack([np.vstack([l, phi]), np.vstack([phi, om])])
                opt_betta = np.copy(self.betta)
                opt_b = np.copy(self.b)
        S[-1] = index
        inversive_H = np.copy(opt_inversive_H)
        objective = np.append(objective, obj)
        #H = opt_H
        self.betta = np.copy(opt_betta)
        self.b = np.copy(opt_b)
        #print(H @ inversive_H)

        for j in range(1, self.n, 1):
            index = 0
            obj = sys.float_info.max
            S = np.append(S, -1)
            #Phi = np.append(Phi, 0)
            for i in range(l):
                if (i in S):
                    continue
                S[-1] = i
                #Phi[-1] = self.new_phi_elem(i, x)
                phi = np.array([[self.new_phi_elem(i, x)]])
                om = self.new_diag_omega_elem(i, x, y)
                omega = self.new_omega_vector(S, x)
                new_inverse_H = self.inverse_block_matrix(inversive_H, np.vstack([phi, omega]), om)
                #H*[b, betta] = [Y, c]
                right = np.append(Y, self.init_c(S, x, y))
                solve = np.matmul(new_inverse_H, right)

                self.betta = solve[1:]
                self.b = solve[0]

                #new_H = np.hstack([np.vstack([H, np.transpose(np.vstack([phi, omega]))]), np.vstack([np.vstack([phi, omega]), np.array([om])])])

                new_obj = self.objective_function(S, x, y)
                #print(new_obj, end=' ')
                if (obj > new_obj):
                    index = i
                    obj = new_obj
                    opt_inversive_H = np.copy(new_inverse_H)
                    #opt_H = np.hstack([np.vstack([H, np.transpose(np.vstack([phi, omega]))]), np.vstack([np.vstack([phi, omega]), np.array([om])])])
                    opt_betta = np.copy(self.betta)
                    opt_b = np.copy(self.b)
            S[-1] = index
            inversive_H = np.copy(opt_inversive_H)
            objective = np.append(objective, obj)
            #H = opt_H
            self.betta = np.copy(opt_betta)
            self.b = np.copy(opt_b)
            #print(H @ inversive_H)
        self.S = S
        self.x = x
        return objective

    def calc(self, x):
        result = 0
        for i in range(self.S.size):
            result += self.betta[i] * self.kernel(self.x[self.S[i]], x)
        result += self.b
        return result


