import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

class CausalQuasiPeriodicKernel(Kernel):
    def __init__(self, #vert_length_scale_input = 1.0, vert_length_scale_input_bounds = (0.1, 100.0), 
                 hor_length_scale_input=1.0, hor_length_scale_input_bounds=(0.1, 30.0),
                 vert_length_scale=1.0, vert_length_scale_bounds=(0.1, 100.0),
                 hor_length_scale_cov_per=1.0,  hor_length_scale_cov_per_bounds=(0.1, 100.0),
                 period_cov=2*np.pi, period_cov_bounds=(0.01, 100*np.pi),
                 hor_length_scale_cov_se=1.0 , hor_length_scale_cov_se_bounds=(0.1, 100.0)):
        

        self.hor_length_scale_input = hor_length_scale_input
        self.hor_length_scale_input_bounds = hor_length_scale_input_bounds

        self.vert_length_scale = vert_length_scale
        self.vert_length_scale_bounds = vert_length_scale_bounds

        self.hor_length_scale_cov_per = hor_length_scale_cov_per
        self.hor_length_scale_cov_per_bounds = hor_length_scale_cov_per_bounds

        self.period_cov = period_cov
        self.period_cov_bounds = period_cov_bounds

        self.hor_length_scale_cov_se = hor_length_scale_cov_se
        self.hor_length_scale_cov_se_bounds = hor_length_scale_cov_se_bounds

        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.zeros((X.shape[0], Y.shape[0]))
        if eval_gradient:
            K_gradient = np.zeros((X.shape[0], X.shape[0], self.n_dims))
        
        if eval_gradient:
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dist = self.mahalanobis_distance(x, y)
                    kernel_val = self.quasi_periodic_kernel(dist)
                    K[i, j] = kernel_val
                    kernel_grad = self.kernel_gradient(x,y)
                    K_gradient[i, j, :] = kernel_grad
            return K, K_gradient

        else:
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dist = self.mahalanobis_distance(x, y)
                    kernel_val = self.quasi_periodic_kernel(dist)
                    K[i, j] = kernel_val
                        
            return K               
        
    
    @property #3
    def hyperparameter_hor_length_scale_input(self):
        return Hyperparameter('hor_length_scale_input', 'numeric', self.hor_length_scale_input_bounds, len(self.hor_length_scale_input))
    
    @property #5
    def hyperparameter_vert_length_scale(self):
        return Hyperparameter('vert_length_scale', 'numeric', self.vert_length_scale_bounds)

    @property #1
    def hyperparameter_hor_length_scale_cov_per(self):
        return Hyperparameter('hor_length_scale_cov_per', 'numeric', self.hor_length_scale_cov_per_bounds)

    @property #4
    def hyperparameter_period_cov(self):
        return Hyperparameter('period_cov', 'numeric', self.period_cov_bounds)

    @property #2
    def hyperparameter_hor_length_scale_cov_se(self):
        return Hyperparameter('hor_length_scale_cov_se', 'numeric', self.hor_length_scale_cov_se_bounds)
    
    def quasi_periodic_kernel(self, dist):
        return self.vert_length_scale **2 * np.exp(-2*np.sin(np.pi*np.abs(dist)/self.period_cov)**2 / self.hor_length_scale_cov_per**2 - (dist**2 / self.hor_length_scale_cov_se**2))
    
    @property
    def n_dims(self):
        # amount of initialized parameter dimensions 
        return len(self.hor_length_scale_input) + 4
    def is_stationary(self):
        return True 
    def diag(self,X):
        dists = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dists[i] = self.mahalanobis_distance(X[i], X[i])
        K_diag = self.quasi_periodic_kernel(dists)
        return K_diag 
    
    def mahalanobis_distance(self, x, y):
        """ This function keeps the causal relation between the points in time"""
        distance = 0
        dx = np.abs(x - y)

        # distance = dx.T * L * dx
        # where L is a lower triangular matrix that is computed as follows
        L= np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if i >= j:
                    L[i,j] = np.exp(-((i-j)/self.hor_length_scale_input[i])**2)
                else:
                    pass
        
        distance = dx.T @ L @ dx

       
        # Alternatively it can also be computed as 
         # for indx in range(len(x)):
        #     for indy in range(len(y)): 
        #         if indx >= indy:
        #             distance+= dx[indx]*dx[indy] *np.exp(-((indx-indy)/self.hor_length_scale_input[indx])**2)
        #         else:
        #             pass

        return distance

    def d_mahalanobis_distance_d_x(self, x, y):
        # Compute the absolute difference dx
        dx = np.abs(x - y)

        # Compute the L matrix based on hor_length_scale_input
        L = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if i >= j:
                    L[i, j] = np.exp(-((i - j) / self.hor_length_scale_input[i])**2)

        # Compute the gradient of distance with respect to each component of x
        d_distance_d_x = np.zeros(len(x))
        for i in range(len(x)):
            sign_dx = np.sign(x[i] - y[i])
            
            # Sum over j for both L[i, j] and L[j, i] contributions
            d_distance_d_x[i] = sign_dx * (np.sum(L[i, :] * dx) + np.sum(L[:, i] * dx))

        return d_distance_d_x
    def kernel_gradient(self, x, y):
        """ Compute the gradient of the kernel in alphabetic order, so:
            1. hor_length_scale_cov_per
            2. hor_length_scale_cov_se
            3. hor_length_scale_input
            4. period_cov
            5. vert_length_scale
        """
        dist = self.mahalanobis_distance(x, y)
        kernel_val = self.quasi_periodic_kernel(dist)
        grad = np.zeros(self.n_dims)

        # 1. hor_length_scale_cov_per
        d_kernel_d_hor_length_scale_cov_per = kernel_val * (4 * np.sin(np.pi * np.abs(dist) / self.period_cov)**2 / self.hor_length_scale_cov_per**3)
        grad[0] = d_kernel_d_hor_length_scale_cov_per

        # 2. hor_length_scale_cov_se
        d_kernel_d_hor_length_scale_cov_se = kernel_val * (2 * dist**2 / self.hor_length_scale_cov_se**3)
        grad[1] = d_kernel_d_hor_length_scale_cov_se

        # 3. hor_length_scale_input
        d_distance_d_hor_length_scale_input = np.zeros_like(self.hor_length_scale_input)
        d1 = np.abs(x - y)
        
        for k in range(len(self.hor_length_scale_input)):
            for indy in range(len(y)):
                if k >= indy:
                    exp_term = np.exp(-((k - indy) / self.hor_length_scale_input[k])**2)
                    d_distance_d_hor_length_scale_input[k] += (
                        d1[k] * d1[indy] * exp_term * 2 * (k - indy)**2 / (self.hor_length_scale_input[k]**3)
                    )
        # Compute each part of the derivative
        term1 = -2 * np.pi * np.cos(2 * np.pi * np.abs(dist) / self.period_cov) / (self.period_cov * self.hor_length_scale_cov_per**2)
        term2 = -2 * dist / self.hor_length_scale_cov_se**2

        # Combine terms to get the derivative
        d_kernel_d_dist = kernel_val * (term1 + term2)

        d_kernel_d_hor_length_scale_input = d_kernel_d_dist * d_distance_d_hor_length_scale_input

        grad[2:2 + len(self.hor_length_scale_input)] = d_kernel_d_hor_length_scale_input

        # 4. period_cov
        d_kernel_d_period_cov = kernel_val * (4 * np.pi * np.sin(np.pi * np.abs(dist) / self.period_cov) * np.cos(np.pi * np.abs(dist) / self.period_cov) / (self.period_cov**2 * self.hor_length_scale_cov_per**2))
        
        grad[2 + len(self.hor_length_scale_input)] = d_kernel_d_period_cov

        # 5. vert_length_scale
        d_kernel_d_vert_length_scale = kernel_val *2 / self.vert_length_scale

        grad[2 + len(self.hor_length_scale_input) + 1] = d_kernel_d_vert_length_scale

        return grad
    def gradient_x(self, x, X_train):
        # Compute the gradient of the kernel with respect to x
        dist = np.zeros(X_train.shape[0])
        for i in range(X_train.shape[0]):
            dist[i] = self.mahalanobis_distance(x,X_train[i])

        # Compute each part of the derivative 
        term1 = -2 * np.pi * np.cos(2 * np.pi * np.abs(dist) / self.period_cov) / (self.period_cov * self.hor_length_scale_cov_per**2)
        term2 = -2 * dist / self.hor_length_scale_cov_se**2
        kernel_val = self.quasi_periodic_kernel(dist)
        # Combine terms to get the derivative
        d_kernel_d_dist = kernel_val * (term1 + term2)
        # 3. hor_length_scale_input
        grad = np.zeros((X_train.shape[0], len(x)))

        for i in range(X_train.shape[0]):
            grad[i] = d_kernel_d_dist[i] * self.d_mahalanobis_distance_d_x(x, X_train[i])
        
        return grad

class CausalRBFKernel(Kernel):
    def __init__(self, #vert_length_scale_input = 1.0, vert_length_scale_input_bounds = (0.1, 100.0), 
                 hor_length_scale_input=1.0, hor_length_scale_input_bounds=(0.1, 30.0),
                 vert_length_scale=1.0, vert_length_scale_bounds=(0.1, 100.0),
                 hor_length_scale_cov_se=1.0 , hor_length_scale_cov_se_bounds=(0.1, 100.0)):
        

        self.hor_length_scale_input = hor_length_scale_input
        self.hor_length_scale_input_bounds = hor_length_scale_input_bounds

        self.vert_length_scale = vert_length_scale
        self.vert_length_scale_bounds = vert_length_scale_bounds

        self.hor_length_scale_cov_se = hor_length_scale_cov_se
        self.hor_length_scale_cov_se_bounds = hor_length_scale_cov_se_bounds

        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.zeros((X.shape[0], Y.shape[0]))
        if eval_gradient:
            K_gradient = np.zeros((X.shape[0], X.shape[0], self.n_dims))
        
        if eval_gradient:
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dist = self.mahalanobis_distance(x, y)
                    kernel_val = self.rbf_kernel(dist)
                    K[i, j] = kernel_val
                    kernel_grad = self.kernel_gradient(x,y)
                    K_gradient[i, j, :] = kernel_grad
            return K, K_gradient

        else:
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dist = self.mahalanobis_distance(x, y)
                    kernel_val = self.rbf_kernel(dist)
                    K[i, j] = kernel_val
                        
            return K               
        
    
    @property #2
    def hyperparameter_hor_length_scale_input(self):
        return Hyperparameter('hor_length_scale_input', 'numeric', self.hor_length_scale_input_bounds, len(self.hor_length_scale_input))
    
    @property #3
    def hyperparameter_vert_length_scale(self):
        return Hyperparameter('vert_length_scale', 'numeric', self.vert_length_scale_bounds)

    @property #1
    def hyperparameter_hor_length_scale_cov_se(self):
        return Hyperparameter('hor_length_scale_cov_se', 'numeric', self.hor_length_scale_cov_se_bounds)
    
    def rbf_kernel(self, dist):
        return self.vert_length_scale **2 * np.exp(- (dist**2 / self.hor_length_scale_cov_se**2))
    
    @property
    def n_dims(self):
        # amount of initialized parameter dimensions 
        return len(self.hor_length_scale_input) + 2
    def is_stationary(self):
        return True 
    def diag(self,X):
        dists = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dists[i] = self.mahalanobis_distance(X[i], X[i])
        K_diag = self.rbf_kernel(dists)
        return K_diag 
    
    def mahalanobis_distance(self, x, y):
        """ This function keeps the causal relation between the points in time"""
        distance = 0
        dx = np.abs(x - y)

        # distance = dx.T * L * dx
        # where L is a lower triangular matrix that is computed as follows
        L= np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if i >= j:
                    L[i,j] = np.exp(-((i-j)/self.hor_length_scale_input[i])**2)
                else:
                    pass
        
        distance = dx.T @ L @ dx

       
        # Alternatively it can also be computed as 
         # for indx in range(len(x)):
        #     for indy in range(len(y)): 
        #         if indx >= indy:
        #             distance+= dx[indx]*dx[indy] *np.exp(-((indx-indy)/self.hor_length_scale_input[indx])**2)
        #         else:
        #             pass

        return distance

    def d_mahalanobis_distance_d_x(self, x, y):
        # Compute the absolute difference dx
        dx = np.abs(x - y)

        # Compute the L matrix based on hor_length_scale_input
        L = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if i >= j:
                    L[i, j] = np.exp(-((i - j) / self.hor_length_scale_input[i])**2)

        # Compute the gradient of distance with respect to each component of x
        d_distance_d_x = np.zeros(len(x))
        for i in range(len(x)):
            sign_dx = np.sign(x[i] - y[i])
            
            # Sum over j for both L[i, j] and L[j, i] contributions
            d_distance_d_x[i] = sign_dx * (np.sum(L[i, :] * dx) + np.sum(L[:, i] * dx))

        return d_distance_d_x
    def kernel_gradient(self, x, y):
        """ Compute the gradient of the kernel in alphabetic order, so:
            1. hor_length_scale_cov_per
            2. hor_length_scale_cov_se
            3. hor_length_scale_input
            4. period_cov
            5. vert_length_scale
        """
        dist = self.mahalanobis_distance(x, y)
        kernel_val = self.rbf_kernel(dist)
        grad = np.zeros(self.n_dims)


        # 1. hor_length_scale_cov_se
        d_kernel_d_hor_length_scale_cov_se = kernel_val * (2 * dist**2 / self.hor_length_scale_cov_se**3)
        grad[0] = d_kernel_d_hor_length_scale_cov_se

        # 2. hor_length_scale_input
        d_distance_d_hor_length_scale_input = np.zeros_like(self.hor_length_scale_input)
        d1 = np.abs(x - y)
        
        for k in range(len(self.hor_length_scale_input)):
            for indy in range(len(y)):
                if k >= indy:
                    exp_term = np.exp(-((k - indy) / self.hor_length_scale_input[k])**2)
                    d_distance_d_hor_length_scale_input[k] += (
                        d1[k] * d1[indy] * exp_term * 2 * (k - indy)**2 / (self.hor_length_scale_input[k]**3)
                    )
    

        # Combine terms to get the derivative
        d_kernel_d_dist = kernel_val * -2 * dist / self.hor_length_scale_cov_se**2

        d_kernel_d_hor_length_scale_input = d_kernel_d_dist * d_distance_d_hor_length_scale_input

        grad[1:1 + len(self.hor_length_scale_input)] = d_kernel_d_hor_length_scale_input

        # 3. vert_length_scale
        d_kernel_d_vert_length_scale = kernel_val *2 / self.vert_length_scale

        grad[1 + len(self.hor_length_scale_input)] = d_kernel_d_vert_length_scale

        return grad
    def gradient_x(self, x, X_train):
        # Compute the gradient of the kernel with respect to x
        dist = np.zeros(X_train.shape[0])
        for i in range(X_train.shape[0]):
            dist[i] = self.mahalanobis_distance(x,X_train[i])

        
        kernel_val = self.rbf_kernel(dist)
        # Combine terms to get the derivative
        d_kernel_d_dist = kernel_val * -2 * dist / self.hor_length_scale_cov_se**2
        # 3. hor_length_scale_input
        grad = np.zeros((X_train.shape[0], len(x)))

        for i in range(X_train.shape[0]):
            grad[i] = d_kernel_d_dist[i] * self.d_mahalanobis_distance_d_x(x, X_train[i])
        
        return grad
    