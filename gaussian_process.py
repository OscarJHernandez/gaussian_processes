#------------------------------------------------------------------------
# Author: Oscar Javier Hernandez
#
#
#
#------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv

class guassian_process:
    
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.N = len(x)
        
        self.K = np.zeros((self.N,self.N)) # The Kernel
        self.sigf = 0.5
        self.l = 1.0
        self.init_Kernel()
    
    
        
    #-------------------------------------------------------------------
    # The radial Basis Functions
    #-------------------------------------------------------------------
    def RBF(self,x1,x2,sigf,l):
        
        
        s = (sigf**2)*np.exp(-0.5*((x1-x2)/l)**2)
        
        if(x1==x2):
            s = s+0.5
        return s
    
    #-------------------------------------------------------------------
    #
    # This function initializes the Kernel for the Gaussian process
    #
    #-------------------------------------------------------------------
    def init_Kernel(self):
        
        sigf = self.sigf
        l = self.l
        
        for i in range(0,self.N):
            for j in range(0,self.N):
                xi = self.x[i]
                xj = self.x[j]
                #print xi,xj
                self.K[i][j] = self.RBF(xi,xj,sigf,l)
        
       # print self.K
        #exit()
    
    
    #-------------------------------------------------------------------
    #
    #
    #
    #
    #-------------------------------------------------------------------
    def predict(self,xp):
        
        KT = np.zeros((self.N+1,self.N+1))
        KpVec = np.zeros(self.N)
        Kpp = self.RBF(xp,xp,self.sigf,self.l)
        
        
        for j in range(0,self.N):
            xj = self.x[j]
            KpVec[j] = self.RBF(xp,xj,self.sigf,self.l)

        
        Kinv = inv(self.K)
        
        
        mean_p = np.matmul(KpVec.T,np.matmul(Kinv,self.y)) 
        stdev_p = Kpp - np.matmul(KpVec,np.matmul(Kinv,KpVec.T))
        
        return mean_p,stdev_p
    
        
