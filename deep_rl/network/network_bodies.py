#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CTgraphConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(CTgraphConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(4 * 4 * 16, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class MNISTConvBody(nn.Module):
    def __init__(self, in_channels=1, noisy_linear=False):
        super(MNISTConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        #self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(6 * 6 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(6 * 6 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        #y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CombinedNet(nn.Module, BaseNet):
    ''' not sure what I'm doing here (AS) need to review'''
    def __init__(self, bodyPredict):
        super(CombinedNet, self).__init__()
#        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
#        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.bodyPredict = bodyPredict
        self.to(Config.DEVICE)

    def returnFeatures(self, x, to_numpy=False):
        phi = self.bodyPredict(tensor(x))
        if to_numpy:
            return phi.cpu().detach().numpy()
        return phi
    #        value = self.fc_value(phi)
#        advantange = self.fc_advantage(phi)
#        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
#        if to_numpy:
#            return q.cpu().detach().numpy()
#        return q

class Mod1LNatureConvBody_direct(nn.Module):
    '''1 layer modulation, direct,
    simple RELU'''
    def __init__(self, in_channels=4):
        super(Mod1LNatureConvBody_direct, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm = layer_init(nn.Conv2d(2*in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_nm):
        y = F.relu(self.conv1(x))
        y_nm = F.relu(self.conv1_nm(x_nm))
        y = y*y_nm
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod1LNatureConvBody_diff(nn.Module):
    '''1 layer modulation, simple RELU: I doubt this can work because RELU will output close to 0 for 0 input, which reduces the gain of the network'''
    def __init__(self, in_channels=4):
        super(Mod1LNatureConvBody_diff, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_fea = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_comb = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_nm):
        y0 = F.relu(self.conv1(x))
        y_nm = F.relu(self.conv1_nm_fea(x_nm))
        y_nm = y0 - y_nm
        y_nm = F.relu(self.conv1_nm_comb(y_nm))
        y = y0*y_nm
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod1LNatureConvBody_diff_sig(nn.Module):
    '''1 layer modulation, implemented by YH
    modulation through sigmoid'''
    def __init__(self, in_channels=4):
        super(Mod2LNatureConvBody_diff_sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_fea = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_comb = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_nm):
        y0 = F.relu(self.conv1(x))
        y_nm = F.relu(self.conv1_nm_fea(x_nm))
        y_nm = y0 - y_nm
        y_nm = 2*torch.sigmoid(self.conv1_nm_comb(y_nm))
        y = y0*y_nm
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod2LNatureConvBody_diff_sig(nn.Module):
    ''' 2-layer modulated with modulation computed from the difference of 2 conv layers. Output of modulation through sigmoid with factor 2, thus in range [0, 1]'''
    def __init__(self, in_channels=4):
        super(Mod2LNatureConvBody_diff_sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_diff = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_diff = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y = y0 * 2 * torch.sigmoid(self.conv1_diff(y0 - y_mem0))

        y0 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod = 2 * torch.sigmoid(self.conv2_diff(y0 - y_mem1))
        y = y0 * y_mod

        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_diff_sig(nn.Module):
    '''3 layer modulation with mod computed with difference of the conv layers. 2*sigmoid at the end.'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_diff_sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_fea = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_nm_comb = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_nm_fea = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_nm_comb = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_nm_fea = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_nm_comb = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_nm):
        y0 = F.relu(self.conv1(x))
        y_nm0 = F.relu(self.conv1_nm_fea(x_nm))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv1_nm_comb(y_nm))
        y = y0*y_nm

        y0 = F.relu(self.conv2(y))
        y_nm0 = F.relu(self.conv2_nm_fea(y_nm0))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv2_nm_comb(y_nm))
        y = y0*y_nm

        y0 = F.relu(self.conv3(y))
        y_nm0 = F.relu(self.conv3_nm_fea(y_nm0))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv3_nm_comb(y_nm))
        y = y0*y_nm

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod2LNatureConvBody_direct_sig(nn.Module):
    '''2-layer modulated (AS), memory modulates directly layer 1 and 2 without computing the difference. TODO: this network should be 0.5 + 0.5*sigmoid, and not 1+ 0.5 sigmoid
'''
    def __init__(self, in_channels=4):
        super(Mod2LNatureConvBody_direct_sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_diff = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_diff = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod = 2 * torch.sigmoid(y_mem0)
        y = y0 * y_mod

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = 2 * torch.sigmoid(y_mem1)
        y = y1 * y_mod1

        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_fix(nn.Module):
    '''Correction to the Mod3L direct: changing 1+ 0.5sig to 0.5 + 0.5sig
    19/12/18: does not work, to delete'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_fix, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod0 = 0.5 + 0.5 * torch.sigmoid(y_mem0)
        y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = 0.5 + 0.5 * torch.sigmoid(y_mem1)
        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        y_mem2 = F.relu(self.conv3_mem_features(y_mem1))
        y_mod2 = 0.5 + 0.5 * torch.sigmoid(y_mem2)
        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_2Sig(nn.Module):
    ''' direct modulation going through a sigmoid * 2 so that the modulation can change plasticity in the rage [0,2], but by mistake was used after relu, which means that effectively the mod range is [1,2]'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_2Sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod0 = 2 * torch.sigmoid(y_mem0)
        y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = 2 * torch.sigmoid(y_mem1)
        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        y_mem2 = F.relu(self.conv3_mem_features(y_mem1))
        y_mod2 = 2 * torch.sigmoid(y_mem2)
        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_4Sig(nn.Module):
    ''' direct modulation going through a sigmoid * 2 so that the modulation can change plasticity in the rage [0,2]'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_4Sig, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod0 = 4 * torch.sigmoid(y_mem0)
        y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = 4 * torch.sigmoid(y_mem1)
        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        y_mem2 = F.relu(self.conv3_mem_features(y_mem1))
        y_mod2 = 4 * torch.sigmoid(y_mem2)
        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_relu_shift1(nn.Module):
    ''' direct modulation going through a relu with input shifted so that an input of 0 results in an output of 1'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_relu_shift1, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.y_mod0 = []
        self.y_mod1 = []
        self.y_mod2 = []

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        self.y_mod0 = F.relu(self.conv1_mem_features(x_mem + 1))
        y = y0 * self.y_mod0

        y1 = F.relu(self.conv2(y))
        self.y_mod1 = F.relu(self.conv2_mem_features(self.y_mod0 + 1))
        y = y1 * self.y_mod1

        y2 = F.relu(self.conv3(y))
        self.y_mod2 = F.relu(self.conv3_mem_features(self.y_mod1 + 1))
        y = y2 * self.y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_relu6_shift05p05(nn.Module):
    ''' direct modulation going through a relu with input shifted so that an input of 0 results in an output of 1 but min mod 0.5, normal relu'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_relu6_shift05p05, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.y_mod0 = []
        self.y_mod1 = []
        self.y_mod2 = []

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        self.y_mod0 = 0.5 + F.relu6(self.conv1_mem_features(x_mem) + 0.5)
        y = y0 * self.y_mod0
    #    y_mod0 = 0.5 +     F.relu6(self.conv1_mem_features(x_mem) + 0.5)
    #    y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        self.y_mod1 = 0.5 + F.relu6(self.conv2_mem_features(self.y_mod0) + 0.5)
        y = y1 * self.y_mod1
#        y_mod1 = 0.5 + F.relu6(self.conv2_mem_features(y_mod0) + 0.5)
#        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        self.y_mod2 = 0.5 + F.relu6(self.conv3_mem_features(self.y_mod1) + 0.5)
        y = y2 * self.y_mod2
#        y_mod2 = 0.5 + F.relu6(self.conv3_mem_features(y_mod1) + 0.5)
#        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class diff_relu6_shift05p05(nn.Module):
    ''' differetial modulation going through a relu with input shifted so that an input of 0 results in an output of 1 but min mod 0.5, normal relu'''
    def __init__(self, in_channels=4):
        super(diff_relu6_shift05p05, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_comb = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_comb = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_comb = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.y_mod0 = []
        self.y_mod1 = []
        self.y_mod2 = []

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        yf0 = F.relu(self.conv1_mem_features(x_mem))
        y0diff = y0 - yf0

        self.y_mod0 = 0.5 + F.relu6(self.conv1_comb(y0diff) + 0.5)
        y = y0 * self.y_mod0

        y1 = F.relu(self.conv2(y))
        yf1 = F.relu(self.conv2_mem_features(yf0))
        y1diff = y1 - yf1
        self.y_mod1 = 0.5 + F.relu6(self.conv2_comb(y1diff) + 0.5)
        y = y1 * self.y_mod1

        y2 = F.relu(self.conv3(y))
        yf2 = F.relu(self.conv3_mem_features(yf1))

        self.y_mod2 = 0.5 + F.relu6(self.conv3_comb(yf2) + 0.5)
        y = y2 * self.y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

        y0 = F.relu(self.conv1(x))
        y_nm0 = F.relu(self.conv1_nm_fea(x_nm))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv1_nm_comb(y_nm))
        y = y0*y_nm

        y0 = F.relu(self.conv2(y))
        y_nm0 = F.relu(self.conv2_nm_fea(y_nm0))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv2_nm_comb(y_nm))
        y = y0*y_nm

        y0 = F.relu(self.conv3(y))
        y_nm0 = F.relu(self.conv3_nm_fea(y_nm0))
        y_nm = y0 - y_nm0
        y_nm = 2*torch.sigmoid(self.conv3_nm_comb(y_nm))
        y = y0*y_nm

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))

class Mod3LNatureConvBody_direct_relu_shift05p05(nn.Module):
    ''' direct modulation going through a relu with input shifted so that an input of 0 results in an output of 1'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_relu_shift05p05, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.y_mod0 = []
        self.y_mod1 = []
        self.y_mod2 = []

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        self.y_mod0 = 0.5 + F.relu(self.conv1_mem_features(x_mem) + 0.5)
        y = y0 * self.y_mod0

        y1 = F.relu(self.conv2(y))
        self.y_mod1 = 0.5 + F.relu(self.conv2_mem_features(self.y_mod0) + 0.5)
        y = y1 * self.y_mod1

        y2 = F.relu(self.conv3(y))
        self.y_mod2 = 0.5 + F.relu(self.conv3_mem_features(self.y_mod1) + 0.5)
        y = y2 * self.y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class Mod3LNatureConvBody_direct_new(nn.Module):
    ''' direct modulation going through a sigmoid and relu along different paths'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_direct_new, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.y_mod0 = []
        self.y_mod1 = []
        self.y_mod2 = []

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        ym0 = self.conv1_mem_features(x_mem)
        self.y_mod0 = 2 * torch.sigmoid(ym0)
        y = y0 * self.y_mod0

        y1 = F.relu(self.conv2(y))
        ym1 = self.conv2_mem_features(F.relu(ym0))
        self.y_mod1 = 2 * torch.sigmoid(ym1)
        y = y1 * self.y_mod1

        y2 = F.relu(self.conv3(y))
        ym2 = self.conv3_mem_features(F.relu(ym1))
        self.y_mod2 = 2 * torch.sigmoid(ym2)
        y = y2 * self.y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class ModNatureConvBody_Prediction(nn.Module):
    '''Working progress to get the latent space prediction (AS)'''
    def __init__(self, in_channels=4):
        super(ModNatureConvBodyPrediction, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class FCBody_CL(nn.Module): # fcbody for continual learning setup
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x, task_label=None):
        if task_label is not None: x = torch.cat([x, task_label], dim=1)
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class FCBody_CL_NM(nn.Module): # fcbody for continual learning setup with neuromodulation
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL_NM, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        #self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            l = NMLinear(dim_in, dim_out, 64)
            l = layer_init_nm(l)
            self.layers.append(l)
        self.layers = nn.ModuleList(self.layers)

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x, task_label=None):
        if task_label is not None: x = torch.cat([x, task_label], dim=1)

        importance_parameters = []
        for layer in self.layers:
            x, impt_params = layer(x, task_label)
            x = self.gate(x)
            importance_parameters.append(impt_params)
        return x, importance_parameters # TODO give importance params name of the module

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DummyBody_CL(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody_CL, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x, task_label=None):
        impt_params = []
        return x, impt_params

class Mod3LNatureConvBody_directTH(nn.Module):
    '''direct neuromodulation with tanh so that plasticity is modulated in the range [-1,1].
    19/12/18 does not work, to delete'''
    def __init__(self, in_channels=4):
        super(Mod3LNatureConvBody_directTH, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv1_mem_features = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv2_mem_features = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.conv3_mem_features = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x, x_mem):
        y0 = F.relu(self.conv1(x))
        y_mem0 = F.relu(self.conv1_mem_features(x_mem))
        y_mod0 = torch.tanh(y_mem0)
        y = y0 * y_mod0

        y1 = F.relu(self.conv2(y))
        y_mem1 = F.relu(self.conv2_mem_features(y_mem0))
        y_mod1 = torch.tanh(y_mem1)
        y = y1 * y_mod1

        y2 = F.relu(self.conv3(y))
        y_mem2 = F.relu(self.conv3_mem_features(y_mem1))
        y_mod2 = torch.tanh(y_mem2)
        y = y2 * y_mod2

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y
