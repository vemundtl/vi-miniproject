import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET
import segmentation_models_pytorch as smp

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "mps"
    print('Running on the MPS')

def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data): 
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
    
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        

def main(
    MODEL_PATH = './model.pt',
    LOAD_MODEL = False,
    ROOT_DIR = './data',
    IMG_HEIGHT = 256,
    IMG_WIDTH = 512,
    BATCH_SIZE = 16,
    LEARNING_RATE = 0.0005,
    EPOCHS = 2,
    model_is_unet=True, 
    model=smp.Unet(in_channels=3, classes=19)
):
    global epoch
    epoch = 0   # epoch is initially assigned to 0. If LOAD_MODEL is true then
                # epoch is set to the last value + 1. 
    LOSS_VALS = [] # Defining a list to store loss values after every epoch
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    ]) 

    train_set = get_cityscapes_data(
        split='train',
        mode='fine',
        relabelled=True,
        root_dir=ROOT_DIR,
        transforms=transform,
        batch_size=BATCH_SIZE,
    )

    print('Data Loaded Successfully!')

    # Defining the model, optimizer and loss function
    if model_is_unet:
        net = UNET(in_channels=3, classes=19).to(DEVICE).train()
    else:
        net = model.to(DEVICE).train()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=255) 

    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    

    #Training the model for every epoch. 
    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')
        loss_val = train_function(train_set, net, optimizer, loss_function, DEVICE)
        print(f'Loss: {loss_val}')
        LOSS_VALS.append(loss_val) 
        torch.save({
            'model_state_dict': net.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        print("Epoch completed and model successfully saved!")


if __name__ == '__main__':
    main()