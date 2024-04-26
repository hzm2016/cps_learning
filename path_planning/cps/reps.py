"""
    Numpy implementation of the CREPS optimizer and upper-level policy.  
    This implementation will generally be faster for relatively small problems
    comapred to the Theano and Torch implementations.    
"""   
import numpy as np     
from scipy.optimize import fmin_l_bfgs_b   
from numpy.random import multivariate_normal as mvnrnd        


def computeSampleWeighting(R, F, eps):
    """Compute sample weights for the upper-level policy update.  
    Computes the sample weights used to update the upper-level policy, according
    to the set of features and rewards found by interacting with the environment.  
    Parameters
    ----------
    R: numpy.ndarray, shape (n_samples, 1)
        Rewards
    F: numpy.ndarray, shape (n_samples, n_context_features)
        Context features  
    eps: float
        Epsilon
    Returns
    -------   
    p: numpy.ndarray, shape (n_samples,)
        Weights for policy update
    """
    assert(R.shape[1] == 1 and
           R.shape[0] == F.shape[0]
           ), "Incorrect parameter size"

    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    # Dual function with analyitical gradients   
    def dual_fnc(x):   
        eta = x[0]
        theta = x[1:].reshape(-1, 1)

        F_mean = F.mean(0).reshape(1, -1)
        R_over_eta = (R - F.dot(theta)) / eta
        R_over_eta_max = R_over_eta.max()
        Z = np.exp(R_over_eta - R_over_eta_max).T
        Z_sum = Z.sum()
        log_sum_exp = R_over_eta_max + np.log(Z_sum / F.shape[0])

        f = eta * (eps + log_sum_exp) + F_mean.dot(theta)
        d_eta = eps + log_sum_exp - (Z.dot(R_over_eta) / Z_sum)
        d_theta = F_mean - (Z.dot(F) / Z_sum)
        return f, np.append(d_eta, d_theta)

    # Initial point
    x0 = [1] + [1] * F.shape[1]

    # Bounds
    min_eta = 1e-10
    bds = np.vstack(([[min_eta, None]], np.tile(None, (F.shape[1], 2))))

    # Minimize using L-BFGS-B algorithm
    x = fmin_l_bfgs_b(dual_fnc, x0, bounds=bds)[0]

    # ----------------------------------------------------------------------
    # Determine weights of individual samples for policy update
    # ----------------------------------------------------------------------
    eta = x[0]
    theta = x[1:].reshape(-1, 1)

    R_baseline_eta = (R - F.dot(theta)) / eta
    p = np.exp(R_baseline_eta - R_baseline_eta.max())
    p /= p.sum()

    return p.reshape(-1,)  


class UpperPolicy:  
    """Upper-level policy.   

    Upper-level policy \pi(w | s) implemented as a linear-Gaussian model
    parametrized by {a, A, sigma}: 
            \pi(w | s) = N(w | a + As, sigma)  

    Parameters
    ----------

    n_context: int
        Number of context features

    verbose: bool, optional (default: False) 
        If True prints the policy parameters after a policy update  
    """
    def __init__(self, n_context, verbose = False):
        self.n_context = n_context 
        self.verbose = verbose

    def set_parameters(self, a, A, sigma): 
        """Set the paramaters of the upper-level policy.

        Parameters
        ----------

        a: numpy.ndarray, shape (1, n_lower_policy_weights)
            Parameter 'a'

        A: numpy.ndarray, shape (n_context_features, n_lower_policy_weights)
            Parameter 'A'

        sigma: numpy.ndarray, shape (n_lower_policy_weights,
                                    n_lower_policy_weights)
            Covariance matrix
        """
        n_lower_policy_weights = a.shape[1]   
        assert(a.shape[0] == 1 and
               A.shape[1] == n_lower_policy_weights and
               A.shape[0] == self.n_context and
               sigma.shape[0] == n_lower_policy_weights and
               sigma.shape[1] == n_lower_policy_weights
               ), "Incorrect parameter sizes"
        self.a = a
        self.sigma = sigma  
        self.A = A   

    def sample(self, S):   
        """Sample the upper-level policy given the context features.

        Sample distribution \pi(w | s) = N(w | a + As, sigma)

        Parameters
        ----------

        S: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
           Sampled lower-policy parameters.
        """
        W = np.zeros((S.shape[0], self.a.shape[1]))
        mus = self.mean(S)
        for sample in range(S.shape[0]):
            W[sample, :] = mvnrnd(mus[sample, :], self.sigma)
        return W  

    def mean(self, S):
        """Return the upper-level policy mean given the context features.

        The mean of the distribution is N(w | a + As, sigma)

        Parameters
        ----------

        S: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
           Distribution mean for contexts
        """
        return self.a + S.dot(self.A)   

    def update(self, w, F, p):  
        """Update the upper-level policy parametersself.

        Update is done using weighted maximum likelihood.

        Parameters
        ----------

        w: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
            Lower-level policy weights

        F: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        p: numpy.ndarray, shape (n_samples,)
            Sample weights
        """
        n_samples = w.shape[0]
        n_lower_policy_weights = self.a.shape[1]
        assert(w.shape[1] == n_lower_policy_weights and
               F.shape[0] == n_samples and
               F.shape[1] == self.n_context and
               p.shape[0] == n_samples and
               p.ndim == 1
               ), "Incorrect parameter size"

        S = np.concatenate((np.ones((p.size, 1)), F), axis = 1)
        P = np.diag(p)

        # Compute new mean
        bigA = np.linalg.pinv(S.T.dot(P).dot(S)).dot(S.T).dot(P).dot(w)
        a = bigA[0, :].reshape(1, -1)

        # Compute new covariance matrix
        wd = w - a
        sigma = (p * wd.T).dot(wd)

        # Update policy parameters  
        self.set_parameters(a, bigA[1:, :], sigma)

        if self.verbose:
            print('Policy update: a, A, mean of sigma')
            print(self.a) 
            print(self.A)   
            print(self.sigma.mean())  


class GPREPS(object):   
    def __init__(self, 
                 context_dim=2, 
                 para_dim=2, 
                 para_lower_bound=0, 
                 para_upper_bound=1, 
                 eps=0.25
        ):   
        self.context_dim = context_dim  
        self.para_dim = para_dim   
        self.a = np.random.random((1, self.para_dim))      
        self.A = np.random.random((self.context_dim, self.para_dim))  
        self.sigma = 0.5 * np.eye(self.para_dim)
        self.eps = eps  
        
        self.para_lower_bound = para_lower_bound  
        self.para_upper_bound = para_upper_bound  
        
        self.p_list = None  
    
    
    def mean(self, Z):  
        """
        Return the upper-level policy mean given the context features.
        The mean of the distribution is N(w | a + As, sigma)
        Parameters
        ----------
        S: numpy.ndarray, shape (n_samples, n_context_features)
            Context features  
        Returns
        -------
        W: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
           Distribution mean for contexts
        """
        return self.a + Z.dot(self.A)      
         
    
    def get_para(self, context_list):    
        para_list = np.zeros((context_list.shape[0], self.para_dim))   
        para_mean = self.mean(context_list)    
        for i in range(context_list.shape[0]):   
            para_list[i, :] = np.clip(np.random.multivariate_normal(mean=para_mean[i, :], cov=self.sigma, size=1), 
                                        self.para_lower_bound, self.para_upper_bound)    
        return para_list  
    
    
    def set_para(self, a, A, sigma): 
        """Set the paramaters of the upper-level policy.   
        Parameters
        ----------
        a: numpy.ndarray, shape (1, n_lower_policy_weights)
            Parameter 'a'  
        A: numpy.ndarray, shape (n_context_features, n_lower_policy_weights)
            Parameter 'A'
        sigma: numpy.ndarray, shape (n_lower_policy_weights,
                                    n_lower_policy_weights)
            Covariance matrix
        """
        n_lower_policy_weights = a.shape[1]   
        assert(a.shape[0] == 1 and
               A.shape[1] == n_lower_policy_weights and
               A.shape[0] == self.context_dim and  
               sigma.shape[0] == n_lower_policy_weights and
               sigma.shape[1] == n_lower_policy_weights
               ), "Incorrect parameter sizes"
        self.a = a
        self.sigma = sigma  
        self.A = A   


    def update_para(self, w, F, p):    
        """Update the upper-level policy parametersself.   
        Update is done using weighted maximum likelihood.    
        Parameters
        ----------
        w: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
            Lower-level policy weights  
        F: numpy.ndarray, shape (n_samples, n_context_features)
            Context features  
        p: numpy.ndarray, shape (n_samples,)   
            Sample weights   
        """
        n_samples = w.shape[0]   
        n_lower_policy_weights = self.a.shape[1]   
        assert(w.shape[1] == n_lower_policy_weights and
               F.shape[0] == n_samples and
               F.shape[1] == self.context_dim and  
               p.shape[0] == n_samples and
               p.ndim == 1   
               ), "Incorrect parameter size"

        S = np.concatenate((np.ones((p.size, 1)), F), axis = 1)
        P = np.diag(p)

        # Compute new mean
        bigA = np.linalg.pinv(S.T.dot(P).dot(S)).dot(S.T).dot(P).dot(w)
        a = bigA[0, :].reshape(1, -1)

        # Compute new covariance matrix
        wd = w - a
        sigma = (p * wd.T).dot(wd)    

        # Update policy parameters    
        self.set_para(a, bigA[1:, :], sigma)        
    
    
    def update_policy(self, context_list, para_list, reward_list):    
        ### calculate sample weights ### 
        print("update policy for one time !!!")
        self.p_list = computeSampleWeighting(reward_list, context_list, self.eps)    
        
        self.update_para(para_list, context_list, self.p_list)   
    