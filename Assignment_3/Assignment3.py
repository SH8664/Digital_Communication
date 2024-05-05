import numpy as np
import matplotlib.pyplot as plt



#Utility Function to calculate the magnitude
def mag(s):
    return np.sqrt(np.dot(s,s)/sample_num)

sample_num = 100


# Gram-Schmidt Orthogonalization
# TODO 1:  Write a Matlab function “[phi1,phi2]=GM_Bases(s1,s2)”
# - The function calculates the Gram-Schmidt orthonormal bases functions (phi1 &phi 2) for two input signals (s1 & s2)
# - The inputs s1 and s2: are two 1× 𝑁 vectors that represent the input signals
# - The outputs phi1 & phi2: are two 1× 𝑁 vectors that represent the two orthonormal bases functions (using Gram-Schmidt). If s1 & s2 have one basis function, then phi2 is 1× 𝑁 zero vector

# Gram-Schmidt Orthogonalization
def GM_Bases(s1, s2):
    phi1 = s1 / mag(s1)
    phi2 = s2 -((np.dot(phi1, s2) * phi1)/sample_num)
    phi2 = phi2 / mag(phi2)
    return phi1, phi2



# Signal Space representation
# TODO 2:  Write a matlab function “[v1,v2]=signal_space(s, phi1,phi2)”
# - The function calculates the signal space representation of input signal s over the orthonormal bases functions (phi1 & phi 2)
# - The inputs s: is a 1× 𝑁 vectors that represent the input signal
# - The inputs phi1 & phi2: are two 1× 𝑁 vectors that represent the two orthonormal bases functions .
# - The output [v1,v2]: is the projections (i.e. the correlations) of s over phi1 and
# phi2 respectively

def signal_space(s, phi1, phi2):
    v1 = np.dot(s, phi1)/sample_num
    v2 = np.dot(s, phi2)/sample_num
    return v1, v2   


# Effect of AWGN on signal space representation
# Now consider the signals 𝑟1(𝑡) = 𝑠1(𝑡) + 𝑤(𝑡); 𝑟2(𝑡) = 𝑠2(𝑡) + 𝑤(𝑡)
# Where 𝑤(𝑡) is a zero mean AWGN with variance 𝜎2

t = np.linspace(0,1,sample_num)

# s1 and s2 are two 1× 𝑁 vectors that represent the input signals
s1 = np.ones(sample_num)
s2 = np.ones(sample_num)
s2[75:100] = -1

print(mag(s1))
# phi1 and phi2 are two 1× 𝑁 vectors that represent the two orthonormal bases functions
phi1,phi2=GM_Bases(s1,s2)

#plot phi1 and phi2 seperately 

plt.plot(t,phi1,linewidth=2)
plt.title("Gram-Schmidt orthonormal bases")
plt.xlabel("Time")
plt.ylabel("phi1(t)")
plt.legend()
plt.show()


plt.plot(t,phi2,linewidth=2)
plt.title("Gram-Schmidt orthonormal bases")
plt.xlabel("Time")
plt.ylabel("phi2(t)")
plt.legend()
plt.show()


#  Use your “signal_space” function (along with the bases from 1) to get the signal space 
# representation of s1(t) & s2(t).
v1,v2=signal_space(s1,phi1,phi2)
v3,v4=signal_space(s2,phi1,phi2)

# Plot the signal space representation for the two signals
plt.plot(v1,v2,'bo',label='s1')
plt.plot(v3,v4,'ro',label='s2')
plt.plot([0, v1], [0, v2], 'b')
plt.plot([0, v3], [0, v4], 'g')
plt.title("Signal Space representation of signals s1,s2")
plt.xlabel("phi1(t)")
plt.ylabel("phi2(t)")
plt.legend()
plt.show()

# calculate the Energy of each signal
e1 = np.dot([v1,v2],[v1,v2])
e2 = np.dot([v3,v4],[v3,v4])



# Function to add noise to a signal based on SNR in dB
def add_noise(s, SNR_dB,E):
    sigma2 = E * 10**(-SNR_dB/10)
    noise = np.random.normal(0, np.sqrt(sigma2), sample_num)
    return s + noise



# SNRs to test
SNRs = [-5, 0, 10]

for SNR in SNRs:
    # Signal space projection
    v1_r1_org, v2_r1_org = signal_space(s1, phi1, phi2)
    v1_r2_org, v2_r2_org = signal_space(s2, phi1, phi2)
    plt.figure()
    plt.title('Signal Space Representation at SNR = {} dB'.format(SNR))

    for i in range(sample_num):
        r1 = add_noise(s1, SNR,e1)
        r2 = add_noise(s2, SNR,e2)

        v1_r1, v2_r1 = signal_space(r1, phi1, phi2)
        v1_r2, v2_r2 = signal_space(r2, phi1, phi2)
        # Plotting
        plt.scatter(v1_r1, v2_r1, color='magenta',facecolors='none', marker='o',label='r1')
        plt.scatter(v1_r2, v2_r2, color='cyan',facecolors='none',marker='o',label='r2')    

        plt.scatter(v1_r1_org, v2_r1_org, color='blue', label='s1')
        plt.scatter(v1_r2_org, v2_r2_org, color='red', label='s2')
    plt.legend(['r1','r2','s1','s2'])
    plt.grid(True)
    plt.xlabel('Projection onto phi1')
    plt.ylabel('Projection onto phi2')
    plt.show()

# How does the noise affect the signal space? Does the noise effect increase or decrease 
# with increasing 𝜎2?

# as SNR increases the noise effect decreases as sigma decreases
# so it enhances the difference between the two signals in the signal space and minimizes the probability of error 

