import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size):
#         super(ConvLSTMCell, self).__init__()

#         assert hidden_channels % 2 == 0

#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.num_features = 4

#         self.padding = int((kernel_size - 1) / 2)

#         self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
#         self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

#         self.Wci = None
#         self.Wcf = None
#         self.Wco = None

#     def forward(self, h, c):
#         # ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c.detach() * self.Wci)
#         # see https://github.com/automan000/Convolutional_LSTM_PyTorch/issues/20

#         ci = torch.sigmoid(self.Whi(h) + c * self.Wci)
#         cf = torch.sigmoid(self.Whf(h) + c * self.Wcf)
#         cc = cf * c + ci * torch.tanh(self.Whc(h))
#         co = torch.sigmoid(self.Who(h) + cc * self.Wco)
#         ch = co * torch.tanh(cc)
#         return ch, cc

#     def init_hidden(self, batch_size, hidden, shape):
#         if self.Wci is None:
#             self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
#             self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
#             self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
#         else:
#             assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
#             assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
#         return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
#                 Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


# class ConvLSTM(nn.Module):
#     # input_channels corresponds to the first input feature map
#     # hidden state is a list of succeeding lstm layers.
#     def __init__(self, input_channels, hidden_channels, kernel_size, step=1):
#         super(ConvLSTM, self).__init__()
#         self.input_channels = [input_channels] + hidden_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.num_layers = len(hidden_channels)
#         self.step = step
#         self._all_layers = []
#         for i in range(self.num_layers):
#             name = 'cell{}'.format(i)
#             cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
#             setattr(self, name, cell)
#             self._all_layers.append(cell)

#     def forward(self, x):
#         """
#         x: [bsize, channel, x, y]
#         """
#         # print(x.shape)
#         # print(self.step)
#         internal_state = []
#         outputs = []
#         for step in range(self.step):
#             for i in range(self.num_layers):
#                 # all cells are initialized in the first step
#                 name = 'cell{}'.format(i)
#                 if step == 0:
#                     bsize, _, height, width = x.size()
#                     (_, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
#                                                              shape=(height, width))
#                     internal_state.append((_, c))

#                 # do forward
#                 (_, c) = internal_state[i]
#                 x, new_c = getattr(self, name)(x, c)
#                 internal_state[i] = (x, new_c)
#             outputs.append(x)

#         return outputs, (x, new_c)

class Generator(nn.Module):
    def __init__(self, ngf=32, nc=3, nz=512):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input Z; First
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # 2nd
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        
            # 3rd
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 3, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
        
            # 4th
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        
            # output nc * 64 * 64
            nn.ConvTranspose2d(ngf, nc, 5, 3, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input)

class FrameGenerator(nn.Module):
    def __init__(self):
        super(FrameGenerator, self).__init__()
        self.img_gen = Generator()

    def forward(self, inputs):
        return [self.img_gen(x) for x in inputs]

# class FrameGenerator(nn.Module):
#     def __init__(self, 
#                 input_channels, hidden_channels, kernel_size, 
#                 step=1):
#         super(FrameGenerator, self).__init__()
#         self.convlstm = ConvLSTM(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, step=step)
#         self.img_gen = ImageGenerator()

#     def forward(self, inputs):
#         inputs = inputs[:,-49:,:].reshape(inputs.shape[0],7,7,inputs.shape[-1]).permute(0,3,1,2)
#         xs = self.convlstm(inputs)[0]
#         return [self.img_gen(x) for x in xs]


# if __name__ == '__main__':
#     # gradient check
#     convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
#                         effective_step=[4]).cuda()
#     loss_fn = torch.nn.MSELoss()

#     input = Variable(torch.randn(5, 1, 512, 64, 32)).cuda()
#     target = Variable(torch.randn(5, 1, 32, 64, 32)).double().cuda()

#     output = convlstm(input)
#     output = output[0][0].double()
#     res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
#     print(res)
    