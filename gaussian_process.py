#------------------------------------------------------------------------
# Author: Oscar Javier Hernandez
#
#
#
#------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv, det
from scipy import stats
from scipy.optimize import minimize

class guassian_process:
    
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.N = len(x)
        
        self.K = np.zeros((self.N,self.N)) # The Kernel
        self.sigf = 1.0
        self.l = 5.0
        self.sig0 = 0.5
        self.init_Kernel()
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        
        self.slope = slope
        self.intercept = intercept
    
    
        
    #-------------------------------------------------------------------
    # The radial Basis Functions
    #-------------------------------------------------------------------
    def RBF(self,x1,x2,sigf,sig0,l):
        
        X=[]
        T=[]
        
        s = (sigf**2)*np.exp(-0.5*((x1-x2)/l)**2)
        
        if(x1==x2):
            s = s+sig0**2
        return s
        
    def mu(self,x):
        
        return self.slope*x+self.intercept
    
    #-------------------------------------------------------------------
    #
    # This function initializes the Kernel for the Gaussian process
    #
    #-------------------------------------------------------------------
    def init_Kernel(self):
        
        sig0 = self.sig0
        sigf = self.sigf
        l = self.l
        
        for i in range(0,self.N):
            for j in range(i,self.N):
                xi = self.x[i]
                xj = self.x[j]
                # sigf,sig0,l
                self.K[i][j] = self.RBF(xi,xj,sigf,sig0,l)
                self.K[j][i] = self.K[i][j] 
    
    #
    # Maximizing the log likelyhood means,  
    #
    # The Log likelyhood of the gp process
    #
    def gp_log_likelyhood(self,phi_vec):
        
                
        K = np.zeros((self.N,self.N))
             
        # Fill in the matrix     
        for i in range(0,self.N):
            for j in range(0,self.N):
                xi = self.x[i]
                xj = self.x[j]
                K[i][j] = self.RBF(xi,xj,phi_vec[0],phi_vec[1],phi_vec[2])
        
        K_inv = inv(K) # Compute the inverse
        
        det_K = det(K) # The determinant of K
        
        print phi_vec,det_K
        
        
        log_p = -0.5*np.matmul(self.y,np.matmul(K_inv,self.y.T)) -0.5*np.log(det_K)
        log_p = log_p - 0.5*self.N*np.log(2.0*np.pi)
        
        
        return log_p
    
    
    #-------------------------------------------------------------------
    # Optimize the hyperparameters of GP using a simple grid search
    #-------------------------------------------------------------------
    def optimize_grid_search(self):
        
        # sigf,sig0,l
        x0 = [1.0,1.0,1.0]
        
        res = minimize(self.gp_log_likelyhood, x0, method='Nelder-Mead', tol=1e-6)
        
        # Reinitialize Kernels
        self.sigf = res.x[0]
        self.sig0 = res.x[1]
        self.l = res.x[2]
        
        self.init_Kernel()
        
        
        return res.x
    
    
    #-------------------------------------------------------------------
    # The prediction class 
    #
    #
    #
    #-------------------------------------------------------------------
    def predict(self,xp):
        
        KT = np.zeros((self.N+1,self.N+1))
        KpVec = np.zeros(self.N)
        Kpp = self.RBF(xp,xp,self.sigf,self.sig0,self.l)
        
        
        for j in range(0,self.N):
            xj = self.x[j]
            KpVec[j] = self.RBF(xp,xj,self.sigf,self.sig0,self.l)

        
        Kinv = inv(self.K)
        
        
        mu_p = np.mean(self.y)
        #mean_p = self.mu(xp) + np.matmul(KpVec.T,np.matmul(Kinv,self.y - self.mu(self.x))) 
        mean_p = mu_p + np.matmul(KpVec.T,np.matmul(Kinv,self.y - mu_p)) 
        stdev_p = Kpp - np.matmul(KpVec,np.matmul(Kinv,KpVec.T))
        
        return mean_p,stdev_p
    
        
