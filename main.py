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
from custom_img_eval import visualize_model_predictions

cudnn.benchmark = True

# WANT TO UPDATE ALL WEIGHTS (FT) OR JUST THE FINAL LAYER(CONV)?
CONV = True
FT = False

def main():
    ######################################################################
    # Load Data
    # ---------
    # We will use torchvision and torch.utils.data packages for loading the data.
    data_dir = 'data/hymenoptera_data'
    dataloaders, dataset_sizes, class_names, data_transforms = dataSetup(data_dir)
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
    if FT:
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
    # ConvNet as fixed feature extractor
    # ------------------
    # We need to set requires_grad = False to freeze the parameters so that the gradients are not computed in backward()
    if CONV:    
        model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)
        model_conv = model_conv.to(device='cpu')
        criterion = nn.CrossEntropyLoss()
        # Observe that only parameters of final layer are being optimized as opposed to before
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ------------------
    if FT:
        model_ft = train_model(dataset_sizes, dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device='cpu')
        visualize_model(class_names, dataloaders, model_ft)
        plt.show()
    if CONV:
        model_conv = train_model(dataset_sizes, dataloaders, model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=2, device='cpu')
        visualize_model(class_names, dataloaders, model_conv)
        plt.show()

    ######################################################################
    # Evaluate as image out of the dataset
    # ------------------
    if FT:
        visualize_model_predictions(class_names,data_transforms,model_ft,img_path='data\\hpersonal_data\\20170726-abelhas_indigenas_jatai-300x260.jpg')
        plt.show()
    if CONV:
        visualize_model_predictions(class_names,data_transforms,model_conv,img_path='data\\hpersonal_data\\20170726-abelhas_indigenas_jatai-300x260.jpg')
        plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()