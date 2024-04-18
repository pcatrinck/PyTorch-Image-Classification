import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import multiprocessing
from data import dataSetup
from show_data import show_image
from trainer import train_model
from prediction import visualize_model

cudnn.benchmark = True
#plt.ion()   # interactive mode

def main():
    data_dir = 'data/hymenoptera_data'
    dataloaders, dataset_sizes, class_names = dataSetup(data_dir)
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    show_image(out, title=[class_names[x] for x in classes])
    plt.show()

    ######################################################################
    # Finetuning the ConvNet
    # ----------------------
    # Load a pretrained model and reset final fully connected layer
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2 (len(class_names))
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device='cpu')
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ------------------
    model_ft = train_model(dataset_sizes, dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device='cpu')
    visualize_model(class_names, dataloaders, model_ft)

    #plt.ioff()
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()