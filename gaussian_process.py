#------------------------------------------------------------------------
# Author: Oscar Javier Hernandez
#
#
#
#------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
from scipy import stats

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
                self.K[i][j] = self.RBF(xi,xj,sigf,sig0,l)
                self.K[j][i] = self.K[i][j] 
    
    
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
    
        
