"""
Modèle adapté depuis SleepEEGNet par Fabrice Simo Defo et Mattéo Fabre
Effectué en Automne 2020
"""

# Maths and Logic
import numpy as np
from matplotlib import pyplot as plt

# Complements
from Preprocessing import get_data
import classes

# SciKit Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch

data = np.load('freestyle_45s_2000Hz_2.npy')  # File to load

file_path = "data/freestyle_45s_2000Hz_2.npy"
X, Y = get_data(file_path)
X = np.stack(X, axis=0)
X_inputs_mean = np.mean(X, axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X_inputs_mean, Y, test_size=0.20, random_state=32)
# X_inputs = np.concatenate(X)

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Define the CNN architecture
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # LEFT CNNS
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50,
                               stride=6)  # TODO: modifier le kernel size (ratio Sleep_EEG)
        self.pool1 = nn.MaxPool1d(8, stride=8)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        # dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)

        # RGHT CNNS
        self.conv1r = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50)
        self.pool1r = nn.MaxPool1d(4, stride=4)
        self.conv2r = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1)
        self.conv3r = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1)
        self.conv4r = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1)
        self.pool2r = nn.MaxPool1d(2, stride=2)
        # dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers for LEFT CNNS
        xl = self.pool1(F.relu(self.conv1(x)))
        xl = self.dropout(xl)
        xl = F.relu(self.conv2(xl))
        xl = F.relu(self.conv3(xl))
        xl = self.pool2(F.relu(self.conv4(xl)))
        # flatten input
        xl = xl.view(-1)

        # add sequence of convolutional and max pooling layers for RIGHT CNNS xr = x_right
        xr = self.pool1r(F.relu(self.conv1r(x)))
        xr = self.dropout(xr)
        xr = F.relu(self.conv2r(xr))
        xr = F.relu(self.conv3r(xr))
        xr = self.pool2r(F.relu(self.conv4r(xr)))
        # flatten input
        xr = xr.view(-1)

        output = torch.cat((xr, xl))

        return output


"""
    BiRNN parameters with pytorch
"""


class BiRNN(nn.Module):

    def __init__(self, hidden_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GU(input_sRize=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


"""
    LSTM Decoder (with attention) with pytorch
"""


class Attention_LSTM_Decoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, seq_len):
        super(Attention_LSTM_Decoder, self).__init__()

        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTMCell(self.encoder_dim, self.decoder_dim)
        self.W_1 = torch.nn.Linear(self.seq_len, self.seq_len)
        self.W_2 = torch.nn.Linear(2 * self.seq_len, self.seq_len)

    '''
    # Context vector creation with pytorch ==> TODO: check if useful or/and working
    
    def context_vector(self,        
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor,  # [seq_length, encoder_dim]
                    ):
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        weights = torch.tanh(weights)
        alpha = torch.nn.functional.softmax(weights, dim=0)
        context_vector = weights*values
        return context_Vector
    '''

    def attention_context(self, out_encoder, hidden_decoder):
        hidden_decoder_repeat = hidden_decoder.repeat(out_encoder.size(0), 1)
        hidden_decoder_repeat = hidden_decoder_repeat.T  # [1 x 3072]
        input = out_encoder.reshape(1, 2 * out_encoder.shape[0])  # [1 x 6144]
        e_ij = self.W_2(input.float()) + self.W_1(hidden_decoder_repeat.float())
        e_ij = torch.tanh(e_ij.float())
        alpha_ij = F.softmax(e_ij, dim=1).unsqueeze(0)
        alpha_ij = alpha_ij.reshape(alpha_ij.shape[2], 1, 1)
        c_i = torch.sum(alpha_ij * out_encoder)  # warning verifier que cette ligne fonctionne
        return c_i

    def forward(self, out_encoder):
        first_input = self.attention_context(out_encoder, torch.tensor(0))
        h_t, c_t = self.lstm(first_input)
        out_decoder = [h_t]
        for i in range(self.seq_len - 1):  # TODO: find better implementation
            h_t, c_t = self.lstm(self.attention_context(out_encoder, h_t), (h_t, c_t))
            out_decoder.append(h_t)
        return out_decoder


class Classifier(nn.Module):

    def __init__(self, seq_len, n_classes):
        super(Classifier, self).__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.fc1 = nn.Linear(seq_len, n_classes)

    def forward(self, x):
        out = F.softmax(self.fc1(x), dim=0)
        return out


class Sleep_EEG():

    def __init__(self, hidden_size, encoder_dim, decoder_dim, n_classes, x):
        self.model = Net()
        self.x = self.model.forward(x)
        self.seq_len = self.x.shape[0]
        self.model2 = BiRNN(hidden_size)
        self.model3 = Attention_LSTM_Decoder(encoder_dim, decoder_dim, self.seq_len)
        self.model4 = Classifier(self.seq_len, n_classes)
        """
        Net.__init__()
        self.x = Net.forward(self, x)
        self.seq_len = self.x.shape[0]
        BiRNN.__init__(self, hidden_size)
        Attention_LSTM_Decoder.__init__(self, encoder_dim, decoder_dim, self.seq_len)
        Classifier.__init__(self, seq_len, n_classes)
        """

    def forward(self):
        x = self.x
        x = x.reshape(x.shape[0], 1, 1)
        hidden = self.model2.initHidden()
        x, hidden = self.model2.forward(x, hidden)  # TODO : utiliser encoder hidden dans decoder hidden

        x = self.model3.forward(x)  # TODO: rajouter hidden en param quand on aura modifier BiRNN

        x = (torch.tensor(x)).float()
        x = self.model4.forward(x)

        return x


"""
# create models
model = Net()
model2 = BiRNN(1)

# In[3]:


batch = 5000
data_tensor = data[0][0:batch].reshape(1, 1, batch)
data_tensor = torch.from_numpy(data_tensor)

# In[4]:


first_pass = model(data_tensor.float())

# In[5]:


hidden = model2.initHidden()
second_pass, next_hidden = model2(first_pass.reshape(first_pass.shape[0], 1, 1), hidden)

# In[7]:


model3 = Attention_LSTM_Decoder(2, 1, second_pass.shape[0])
third_pass = model3(second_pass)

# In[8]:


third_pass = (torch.tensor(third_pass)).float()
model4 = Classifier(third_pass.shape[0], 10)
out = model4(third_pass)

# In[9]:
"""
import glob

names = glob.glob(r"C:\Users\matfa\Documents\METIS\Classification\Acquisition\Data\7_electrodes\*.npy")
outputs = []

for path in names:
    name = path[76:-15]
    outputs.append(name)

x, y = [], []

for path in names:
    X_train, Y_train = get_data(path, kAcquisitionTime=3, outputs=outputs)
    x.append(X_train)
    y.append(Y_train)

X_train, Y_train = get_data(
    r"C:\Users\matfa\Documents\METIS\Classification\Acquisition\Data\7_electrodes\no training\freestyle_45s_2000Hz_2.npy")

X_train = np.stack(X_train)
Y_train = np.stack(Y_train)
for i, xi in enumerate(x):
    xi = np.stack(xi)
    yi = np.stack(y[i])
    X_train = np.concatenate((X_train, xi), axis=0)
    Y_train = np.concatenate((Y_train, yi), axis=0)

X_train = np.mean(X_train, axis=1)

train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], Y_train[i]])

"""
batch = 5000
data_tensor = data[0][0:batch].reshape(1, 1, batch)
data_tensor = torch.from_numpy(data_tensor)
final_model = Sleep_EEG(1, 2, 1, 10, data_tensor.float())
out = final_model.forward()
print("tensor probas : ", out)
"""

data_tensor = torch.from_numpy(X_train[0].reshape(1, 1, 200))
final_model = Sleep_EEG(1, 2, 1, 10, data_tensor.float())

import torch.optim as optim

LR = 1e-4
optimizer = optim.Adam(Sleep_EEG.parameters(), lr=LR)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
epoch = 500
losses = []
criterion = nn.MSELoss()
for k in range(epoch):
    loss = 0
    for batch_features in train_loader:
        optimizer.zero_grad()
        out = Sleep_EEG(batch_features.float())
        train_loss = criterion(out.float(), batch_features.float())
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
        # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(k + 1, epoch, loss))

"""
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.adam(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
"""
