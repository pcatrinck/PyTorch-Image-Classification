import torch
import matplotlib.pyplot as plt
from PIL import Image
from show_data import show_image

def visualize_model_predictions(class_names,data_transforms,model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device='cpu')

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        show_image(img.cpu().data[0])

        model.train(mode=was_training)