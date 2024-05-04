import numpy as np

#--Theoretical NTK-values calculator, based on nlo-calculator by Zhang Allan--
class NTKSimulator:
    def __init__(self, cb, cw, lb, lw, act, drv, n_samples=1000000):
        self.x = None
        self.a = None
        self.cb, self.cw, self.lb, self.lw = cb, cw, lb, lw
        self.act, self.drv = act, drv
        self.n_samples = n_samples
    
    def calculate_form2(self, cov, n_samples=1000000):
        raw = np.random.multivariate_normal(mean=np.zeros_like(cov[0]), cov=cov, size=n_samples)
        raw_act = self.act(raw).mean(axis=0)
        raw_drv = self.drv(raw).mean(axis=0)
        form2_act = (raw_act[:, None] * raw_act[None, :])
        form2_drv = (raw_drv[:, None] * raw_drv[None, :])
        #form2_drv = (raw_drv[:, :, None] * raw_drv[:, None, :]).mean(axis=0)
        return form2_act, form2_drv
    
    def calculate_layer0(self, KK, Theta, xx):
        form2_0 = (xx[:, :, None] * xx[:, None, :]).mean(axis=0)
        KK[0] = self.cb + self.cw * form2_0
        Theta[0] = self.lb + self.lw * form2_0

    def calculate_layer(self, KK, Theta, idx):
        form2_act, form2_drv = self.calculate_form2(KK[idx-1], self.n_samples)
        KK[idx] = self.cb + self.cw * form2_act
        Theta[idx] = self.lb + self.lw * form2_act + self.cw * np.multiply(form2_drv, Theta[idx-1])
