from torch import save,device,optim,nn,cuda
from torch.utils.data import DataLoader
from CharacterIdentification.model import ChineseCharNet as ccn_model, ChineseCharDataLoader
from CharacterIdentification.model.ChineseCharNet import ChineseCharNet
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
char_dataset = ChineseCharDataLoader.Chinese_char_dataset()
print("dataset create complete")
char_loader = DataLoader(char_dataset, batch_size=8, shuffle=False)
model = ccn_model.ChineseCharNet(char_dataset.__len__()).to('cuda')
optimizer = optim.Adam(model.parameters(), lr = 1e-1)
criterion = nn.CrossEntropyLoss()

model = ChineseCharNet(char_loader.__len__())
device = device('cuda:0' if cuda.is_available() else 'cpu')
print("use: "+ str(device))
model.to(device)
N_epoch = 100
for epoch in range(N_epoch):
    running_loss = 0
    for i, data in enumerate(char_loader):
        ims, labels = data[0].to(device), data[1].to(device).long()
        optimizer.zero_grad()
        outs = model(ims)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print("epoch:"+str(epoch)+" batch:"+str(i)+" loss:"+str(running_loss / 200))
            running_loss = 0
    if epoch % 10 == 0:
        print('Save checkpoint...')
        save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()},"./log")
        pass
save(model.state_dict(), "chinese_char_model_" + str(N_epoch) + ".pt")