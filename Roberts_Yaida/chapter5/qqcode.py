import numpy as np
import math

from common.ffn.ffn_relu import ParametricReLUNet
#from common.ffn.ffn_tanh import TanhNet
import matplotlib.pyplot as plt

'''n0: # dimension of x
    nk: # dimension of hidden layers
    nl: # dimension of output layer
    ln: # number of layers
    nd: # number of points in train-set'''
n_all = 500
n0,nk,nl,ln=n_all, n_all, n_all, 100
nd=2

xx = np.random.normal(size=(n0, nd)).astype(np.float32)

'''Cb and Cw constants'''
cb, cw = 0, 1

slope_plus, slope_minus= math.sqrt(1.5), math.sqrt(0.5)#1,1 #math.sqrt(2), 0
aux = ParametricReLUNet()
aux.set_slopes(slope_plus, slope_minus)
act = np.vectorize(aux.PReLUz)  #np.tanh activation function

class MCSimulator:
    def __init__(self):
        self.x = None
        self.a = None
    
    def generate_samples(self, cov, act, n_samples=5000000):
        self.x = np.random.multivariate_normal(mean=np.zeros_like(cov[0]), cov=cov, size=n_samples)
        self.a = act(self.x)
        self.aa = self.a[:, :, None] * self.a[:, None, :]
        self.xx = self.x[:, :, None] * self.x[:, None, :]
        self.xx_c = self.xx - cov

    def calculate_form1(self):
        '''<sigma sigma sigma sigma>'''
        aa = self.aa
        return (aa[..., None, None] * aa[:, None, None, ...]).mean(axis=0)

    def calculate_form2(self):
        '''<sigma sigma>'''
        return self.aa.mean(axis=0)

    def calculate_form3(self):
        '''<sigma sigma (z z - g)>'''
        return (self.aa[..., None, None] * self.xx_c[:, None, None, ...]).mean(axis=0)

    def calculate_form4(self):
        '''<sigma sigma (z z - K)(z z - K)>'''
        return (self.aa[..., None, None, None, None] * self.xx_c[:, None, None, ..., None, None] * self.xx_c[:, None, None, None, None, ...]).mean(axis=0)

    def calculate_form5(self, cov):
        '''<sigma sigma (-2 z z + K)>'''
        return (self.aa[..., None, None] * (-2 * self.xx + cov)[:, None, None, ...]).mean(axis=0)

def calculate_V_upper(V, G_inv):
    '''(4d, 2d) -> 4d'''
    a = np.transpose(V @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    return a


#--calculate theoretical values--
G = np.zeros((ln, nd, nd))
K = np.zeros_like(G)
K_inv = np.zeros_like(G)
G1 = np.zeros_like(G)

V = np.zeros((ln, nd, nd, nd, nd))
V_upper = np.zeros_like(V)


K[0] = cb + cw * (xx[:, None, :] * xx[:, :, None]).mean(axis=0)
G[0] = K[0]
K_inv[0] = np.linalg.inv(K[0])
sim = MCSimulator()
for i in range(1, ln):
    sim.generate_samples(K[i-1], act, n_samples=5000000)
    
    form2 = sim.calculate_form2()
    K[i] = cb + cw * form2
        
    form3 = sim.calculate_form3()
    #4.90
    V[i] = cw**2 * (sim.calculate_form1() - form2[..., None, None] * form2
        ) + cw**2 / 4 * np.sum(V_upper[i-1] * form3[:, :, None, None, :, :, None, None] * form3 [None, None, :, :, None, None, :, :], axis=(4, 5, 6, 7))
    
    form4 = sim.calculate_form4()
    form5 = sim.calculate_form5(K[i-1])
    # 4.115
    G1[i] = cw * (1/2 * np.sum(K_inv[i-1][:, None, :, None] * K_inv[i-1][None, :, None, :] * G1[i-1] * form3[..., None, None], axis=(2, 3, 4, 5))
        + 1/8 * np.sum(V_upper[i-1] * form4, axis=(2, 3, 4, 5)) / nk
        + 1/4 * np.sum(np.transpose(V_upper[i-1], (0, 2, 1, 3)) * K[i-1] * form5[..., None, None], axis=(2, 3, 4, 5)) / nk)
    
    G[i] = K[i] + G1[i]
    K_inv[i] = np.linalg.inv(K[i])
    V_upper[i] = calculate_V_upper(V[i], K_inv[i])
    print('-', end='')

#--simulate with neural nets--
testNet = ParametricReLUNet(n0=n0,nk=nk,nl=nl,l=ln)
testNet.set_slopes(slope_plus, slope_minus)
#testNet = TanhNet(n0=n0,nk=nk,nl=nl,l=ln)
testNet.set_log_level("info")
if (nd > 1):
    testNet.set_gmetric_recording_indices([(1,1),(1,2),(2,2)])
else:
    testNet.set_gmetric_recording_indices([(1,1)])

experiments_number = 200
yy = np.zeros((experiments_number, nl, nd))
G00_records = []
if (nd > 1):
    G11_records = []
    G01_records = []

#for each experiment re-initialisation of the weights with recalculation
for experiment_number in range(experiments_number):
    testNet.init_weights(cb, cw)
    res = testNet.forward(xx)
    yy[experiment_number] = res
    G00_records.append(testNet.get_gmetric(1,1).copy())
    if (nd > 1):
        G11_records.append(testNet.get_gmetric(2,2).copy())
        G01_records.append(testNet.get_gmetric(1,2).copy())
    
    print('-', end='')

#--plots--
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

G00_max = np.max(G00_records, axis=0)
G00_min = np.min(G00_records, axis=0)
G00_avg = np.average(G00_records, axis=0) #/len(G00_records)
cord_x = np.arange(0, len(G00_records[0]))

ax1.plot(G00_avg, color='g', label=f"Average experimental values")
#ax1.plot(G00_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax1.plot(G00_max, color='y', alpha=0.5, label=f"Max experimental values")
ax1.plot(G[:,0,0], color='r', label=f"Theoretical values")
ax1.plot(K[:,0,0], color='c', label=f"leading term")
#ax1.fill_between(x=cord_x, y1=G00_min, y2=G00_max, color='y', alpha=.2)
ax1.legend()

ax2.plot(G00_avg, color='g', label=f"Average experimental values")
#ax2.plot(G00_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax2.plot(G00_max, color='y', alpha=0.5, label=f"Max experimental values")
ax2.plot(G[:,0,0], color='r', label=f"Theoretical values")
ax2.plot(K[:,0,0], color='c', label=f"leading term")
ax2.fill_between(x=cord_x, y1=G00_min, y2=G00_max, color='y', alpha=.2)
ax2.set_yscale("log")
ax2.legend()

fig.suptitle(f"G00 n0=nk=nl={n_all}")
fig.savefig(f"figure0.jpg")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

G11_max = np.max(G11_records, axis=0)
G11_min = np.min(G11_records, axis=0)
G11_avg = np.average(G11_records, axis=0) #/len(G00_records)
cord_x = np.arange(0, len(G11_records[0]))

ax1.plot(G11_avg, color='g', label=f"Average experimental values")
#ax1.plot(G11_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax1.plot(G11_max, color='y', alpha=0.5, label=f"Max experimental values")
ax1.plot(G[:,1,1], color='r', label=f"Theoretical values")
ax1.plot(K[:,1,1], color='c', label=f"leading term")
#ax1.fill_between(x=cord_x, y1=G11_min, y2=G11_max, color='y', alpha=.2)
ax1.legend()

ax2.plot(G11_avg, color='g', label=f"Average experimental values")
#ax2.plot(G11_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax2.plot(G11_max, color='y', alpha=0.5, label=f"Max experimental values")
ax2.plot(G[:,1,1], color='r', label=f"Theoretical values")
ax2.plot(K[:,1,1], color='c', label=f"leading term")
ax2.fill_between(x=cord_x, y1=G11_min, y2=G11_max, color='y', alpha=.2)
ax2.set_yscale("log")
ax2.legend()

fig.suptitle(f"G11 n0=nk=nl={n_all}")
fig.savefig(f"figure1.jpg")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

G01_max = np.max(G01_records, axis=0)
G01_min = np.min(G01_records, axis=0)
G01_avg = np.average(G01_records, axis=0) #/len(G00_records)
cord_x = np.arange(0, len(G01_records[0]))

ax1.plot(G01_avg, color='g', label=f"Average experimental values")
#ax1.plot(G01_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax1.plot(G01_max, color='y', alpha=0.5, label=f"Max experimental values")
ax1.plot(G[:,0,1], color='r', label=f"Theoretical values")
ax1.plot(K[:,0,1], color='c', label=f"leading term")
#ax1.fill_between(x=cord_x, y1=G01_min, y2=G01_max, color='y', alpha=.2)
ax1.legend()

ax2.plot(G01_avg, color='g', label=f"Average experimental values")
#ax1.plot(G01_min, color='c', alpha=0.5, label=f"Min experimental values")
#ax1.plot(G01_max, color='y', alpha=0.5, label=f"Max experimental values")
ax2.plot(G[:,0,1], color='r', label=f"Theoretical values")
ax2.plot(K[:,0,1], color='c', label=f"leading term")
ax2.fill_between(x=cord_x, y1=G01_min, y2=G01_max, color='y', alpha=.2)
ax2.set_yscale("log")
ax2.legend()

fig.suptitle(f"G01 n0=nk=nl={n_all}")
fig.savefig(f"figure2.jpg")