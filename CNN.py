import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Using gpu: %s ' % torch.cuda.is_available())
import torch.nn as nn


def trainNNet():
    return 0


z_dim = 32
hidden_dim = 128
label_dim = 2


class generator(nn.Module):
    def __init__(self, z_dim=z_dim, label_dim=label_dim, hidden_dim=hidden_dim):
        super(generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + label_dim, hidden_dim),
                                 nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, input, label_onehot):
        x = torch.cat([input, label_onehot], 1)
        return self.net(x)


class discriminator(nn.Module):
    def __init__(self, z_dim=z_dim, label_dim=label_dim, hidden_dim=hidden_dim):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(2 + label_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1),
                                 nn.Sigmoid())

    def forward(self, input, label_onehot):
        x = torch.cat([input, label_onehot], 1)
        return self.net(x)


net_CG = generator().to(device)
net_CD = discriminator().to(device)

batch_size = 50
lr = 1e-3
nb_epochs = 1000

optimizer_CG = torch.optim.Adam(net_CG.parameters(), lr=lr)
optimizer_CD = torch.optim.Adam(net_CD.parameters(), lr=lr)
loss_D_epoch = []
loss_G_epoch = []

for e in range(nb_epochs):
    rperm = np.random.permutation(X.shape[0]);
    np.take(X, rperm, axis=0, out=X);
    np.take(Y, rperm, axis=0, out=Y);
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)
    real_labels = torch.from_numpy(Y).type(torch.LongTensor)
    loss_G = 0
    loss_D = 0
    for real_batch, real_batch_label in zip(real_samples.split(batch_size), real_labels.split(batch_size)):
        # improving D
        z = torch.empty(batch_size, z_dim).normal_().to(device)
        #
        # your code here
        # hint: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
        #

        loss = -torch.mean(torch.log(1 - D_scores_on_fake) + torch.log(D_scores_on_real))
        optimizer_CD.zero_grad()
        loss.backward()
        optimizer_CD.step()
        loss_D += loss

        # improving G
        z = torch.empty(batch_size, z_dim).normal_().to(device)
        #
        # your code here
        #

        loss = -torch.mean(torch.log(D_scores_on_fake))
        optimizer_CG.zero_grad()
        loss.backward()
        optimizer_CG.step()
        loss_G += loss

    loss_D_epoch.append(loss_D)
    loss_G_epoch.append(loss_G)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), (1, 1))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), (1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), (1, 1))
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 1)

    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]
        return F.mse_loss(Y_pred, Y_target)

    def forward(self, data):
        x = data['input']

        # Convolutions mixed with pooling layers
        x = x.view(-1, 2, 19, 19)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))

        x = F.max_pool2d(x, (4, 4))[:, :, 0, 0]
        x = F.dropout(x, p=0.2)

        x = self.layer1(x)
        x = self.layer2(F.tanh(x))

        return {'target': x}