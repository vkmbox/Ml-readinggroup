import numpy as np

#--Theoretical values calculator, submitted by Zhang Allan--
class MCSimulator:
    def __init__(self, cb, cw, act, n_samples=5000000):
        self.x = None
        self.a = None
        self.cb = cb
        self.cw = cw
        self.act = act
        self.n_samples = n_samples
    
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
    
    def calculate_layer(self, KK, KK_inv, GG, GG1, VV, VV_upper, nk_2, idx):
        self.generate_samples(KK[idx-1], self.act, self.n_samples)
        
        form2 = self.calculate_form2()
        KK[idx] = self.cb + self.cw * form2
            
        form3 = self.calculate_form3()
        #4.90
        VV[idx] = self.cw**2 * (self.calculate_form1() - form2[..., None, None] * form2
            ) + self.cw**2 / 4 * np.sum(VV_upper[idx-1] * form3[:, :, None, None, :, :, None, None] \
                                * form3 [None, None, :, :, None, None, :, :], axis=(4, 5, 6, 7))
        
        form4 = self.calculate_form4()
        form5 = self.calculate_form5(KK[idx-1])
        # 4.115
        GG1[idx] = self.cw * (1/2 * np.sum(KK_inv[idx-1][:, None, :, None] * KK_inv[idx-1][None, :, None, :] * GG1[idx-1] \
                                    * form3[..., None, None], axis=(2, 3, 4, 5))
            + 1/8 * np.sum(VV_upper[idx-1] * form4, axis=(2, 3, 4, 5)) / nk_2
            + 1/4 * np.sum(np.transpose(VV_upper[idx-1], (0, 2, 1, 3)) * KK[idx-1] * form5[..., None, None], axis=(2, 3, 4, 5)) / nk_2)
        
        GG[idx] = KK[idx] + GG1[idx]
        KK_inv[idx] = np.linalg.inv(KK[idx])
        VV_upper[idx] = MCSimulator.calculate_V_upper(VV[idx], KK_inv[idx])        

    @staticmethod
    def calculate_V_upper(V, G_inv):
        '''(4d, 2d) -> 4d'''
        buff = np.transpose(V @ G_inv, (-1, 0, 1, 2))
        buff = np.transpose(buff @ G_inv, (-1, 0, 1, 2))
        buff = np.transpose(buff @ G_inv, (-1, 0, 1, 2))
        buff = np.transpose(buff @ G_inv, (-1, 0, 1, 2))
        return buff
