
# Import all of the libraries that we need
import gaussian_process as gp
import numpy as np
import matplotlib.pyplot as plt



# Read in all available weight data from data file
fname = "weight2.txt"
data = np.loadtxt(fname)
Ndata = len(data)

# Extract the indices of the days
x1 = np.asarray([data[k][0] for k in range(0,Ndata)])

# Extract the body weight per day in lbs
y1 = np.asarray([data[k][1] for k in range(0,Ndata)])

#-----------------------------------------------------------------------


# Read in all available weight data from data file
fname = "weight2.txt"
data = np.loadtxt(fname)
Ndata = len(data)

# Extract the indices of the days
days = np.asarray([data[k][0] for k in range(0,Ndata)])

# Extract the body weight per day in lbs
weight = np.asarray([data[k][1] for k in range(0,Ndata)])

#x = days
#y = weight
x = np.arange(0,40,2)
y = 0.001*x**2


GaussP = gp.guassian_process(x, y)


phi_vec = [1.0,1.0,1.0]
print GaussP.gp_log_likelyhood(phi_vec)

phi_vec = [1.0,0.1,1.0]
print GaussP.optimize_grid_search()


#d = np.arange(-5,16,0.10)
d = np.arange(1,x[-1]+20,0.10)
p = np.asarray([GaussP.predict(xp)[0] for xp in d])
dp = np.asarray([GaussP.predict(xp)[1] for xp in d])

print "mean", np.mean(x),np.mean(y)
#plt.plot(days,weight,"o")
#plt.plot(x1,y1,"-o")
plt.plot(x,y,"o")
plt.plot(d,p+dp,"-")
plt.plot(d,p,"-")
plt.plot(d,p-dp,"-")
plt.show()
