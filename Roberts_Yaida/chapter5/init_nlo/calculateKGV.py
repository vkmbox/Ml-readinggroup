import numpy as np
import torch


class GVCalculator:
    def __init__(self, n_layers, n_neurons_hidden, dim_output, dim_x, n_data, cw, cb, activation):
        '''
        n_layers: # layers
        n_neurons: # neurons of every hidden layer, int or list of int
        dim_output: dimension of the output of the last layer
        dim_x: dimension of input
        n_data: number of data points
        '''
        self.n_layers = n_layers
        dims = [dim_x]
        if isinstance(n_neurons_hidden, int):
            dims += [n_neurons_hidden] * (n_layers - 1)
        else:
            dims += n_neurons_hidden
        dims.append(dim_output)
        self.dims = dims
        self.n_data = n_data
        self.cw = cw
        self.cb = cb
        self.activation = activation

    def calculate_gv(self, x, n_samples=1000000, n_batches=1, device='cpu', verbose=1):
        '''
        x has to have shape (dim_x, n_data), numpy array
        n_samples: number of samples in Monte-Carlo simulation
        returns K, G, V
        K, G have shape (n_layers, n_data, n_data)
        V has shape (n_layers, n_data, n_data, n_data, n_data)
        '''
        G = np.zeros((self.n_layers, self.n_data, self.n_data), dtype=np.float32)
        K = np.zeros_like(G)
        K_inv = np.zeros_like(G)
        G1 = np.zeros_like(G)

        V = np.zeros((self.n_layers, self.n_data, self.n_data, self.n_data, self.n_data), dtype=np.float32)
        V_upper = np.zeros_like(V)

        cb, cw, dims = self.cb, self.cw, self.dims

        K[0] = cb + cw * (x[:, None, :] * x[:, :, None]).mean(axis=0)
        G[0] = K[0]
        K_inv[0] = np.linalg.inv(K[0])
        sim = MCSimulator(self.n_data, n_samples, n_batches, device=device)
        for i in range(1, self.n_layers):
            form1, form2, form3, form4, form5 = sim.calculate_forms(K[i-1], self.activation)
            # 4.118
            K[i] = cb + cw * form2
            #4.119
            V[i] = cw**2 * (form1 - form2[..., None, None] * form2)
            V[i] += cw**2 / 4 * dims[i] / dims[i-1] * np.sum(V_upper[i-1] * form3[:, :, None, None, :, :, None, None] * form3 [None, None, :, :, None, None, :, :], axis=(4, 5, 6, 7))
            # 4.115
            G1[i] = 1/2 * np.sum(K_inv[i-1][:, None, :, None] * K_inv[i-1][None, :, None, :] * G1[i-1] * form3[..., None, None], axis=(2, 3, 4, 5))
            G1[i] += 1/8/dims[i-1] * np.sum(V_upper[i-1] * form4, axis=(2, 3, 4, 5))
            G1[i] += 1/4/dims[i-1] * np.sum(np.transpose(V_upper[i-1], (0, 2, 1, 3)) * K[i-1] * form5[..., None, None], axis=(2, 3, 4, 5))
            G1[i] *= cw
            
            G[i] = K[i] + G1[i]
            K_inv[i] = np.linalg.inv(K[i])
            V_upper[i] = calculate_V_upper(V[i], K_inv[i])
            if verbose:
                print('-', end='')
        return K, G, V


class MCSimulator:
    def __init__(self, n, n_samples=1000000, n_batches=1, device='cpu'):
        self.n = n
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.device = device
        self.x = torch.empty(n_samples, n, dtype=torch.float32, device=device)
        self.a = torch.empty_like(self.x)
        self.aa = torch.empty(n_samples, n, n, dtype=torch.float32, device=device)
        self.xx = torch.empty_like(self.aa)
        self.xx_c = torch.empty_like(self.aa)
        self.aa_xx_c = torch.empty(n_samples, n, n, n, n, dtype=torch.float32, device=device)

    def generate_samples(self, cov, act):
        self.cov = torch.tensor(cov, dtype=torch.float32).to(self.device)
        mn = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(self.cov[0]), self.cov)
        x = mn.sample([self.n_samples])
        self.x[:] = x
        self.a[:] = act(self.x)
        self.aa[:] = self.a[:, :, None] * self.a[:, None, :]
        self.xx[:] = self.x[:, :, None] * self.x[:, None, :]
        self.xx_c[:] = self.xx - torch.tensor(cov, dtype=torch.float32).to(self.device)

    def calculate_form1(self):
        '''<sigma sigma sigma sigma>'''
        aa = self.aa
        return (aa[..., None, None] * aa[:, None, None, ...]).mean(axis=0).cpu().numpy()

    def calculate_form2(self):
        '''<sigma sigma>'''
        return self.aa.mean(axis=0).cpu().numpy()

    def calculate_form3(self):
        '''<sigma sigma (z z - g)>'''
        self.aa_xx_c[:] = self.aa[..., None, None] * self.xx_c[:, None, None, ...]
        return self.aa_xx_c.mean(axis=0).cpu().numpy()

    def calculate_form4(self):
        '''<sigma sigma (z z - K)(z z - K)>'''
        return (self.aa_xx_c[..., None, None] * self.xx_c[:, None, None, None, None, ...]).mean(axis=0).cpu().numpy()

    def calculate_form5(self, cov):
        '''<sigma sigma (-2 z z + K)>'''
        cov = torch.tensor(cov, dtype=torch.float32).to(self.device)
        return (self.aa[..., None, None] * (-2 * self.xx + cov)[:, None, None, ...]).mean(axis=0).cpu().numpy()
    
    def calculate_forms(self, cov, act):
        form1 = []
        form2 = []
        form3 = []
        form4 = []
        form5 = []
        for _ in range(self.n_batches):
            self.generate_samples(cov, act)
            form1.append(self.calculate_form1())
            form2.append(self.calculate_form2())
            form3.append(self.calculate_form3())
            form4.append(self.calculate_form4())
            form5.append(self.calculate_form5(cov))
        return sum(form1) / self.n_batches, sum(form2) / self.n_batches, sum(form3) / self.n_batches, sum(form4) / self.n_batches, sum(form5) / self.n_batches


def calculate_V_upper(V, G_inv):
    '''(4d, 2d) -> 4d'''
    a = np.transpose(V @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    a = np.transpose(a @ G_inv, (-1, 0, 1, 2))
    return a


if __name__=="__main__":
    n_layers = 64
    n_neurons_hidden = 16
    dim_output = 2
    dim_x = 16
    n_data = 2
    cw = 1
    cb = 0
    activation = torch.tanh
    # act = lambda x: torch.nn.ReLU()(x) * torch.sqrt(torch.tensor(2))

    x = np.random.normal(size=(dim_x, n_data)).astype(np.float32)
    gvcalculator = GVCalculator(n_layers, n_neurons_hidden, dim_output, dim_x, n_data, cw, cb, activation)
    K, G, V = gvcalculator.calculate_gv(x, n_samples=100000, device='cpu')
    # K, G, V = gvcalculator.calculate_gv(x, n_samples=1000000, device='cuda')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(K[:, 0, 0])
    plt.plot(G[:, 0, 0])
    plt.show()


