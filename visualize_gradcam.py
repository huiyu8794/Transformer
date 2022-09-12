from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenGradCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.ViT import *
# from Transformer.models.Resnet34 import * 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torchvision.transforms as T
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def imshow_np(img, filename, image_dir=""):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.axis("off")
    mkdir("./save_image/" + image_dir + "/")
    plt.savefig("./save_image/" + image_dir + "/" + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()


def reshape_transform(tensor, height=14, width=14): 
    # tensor.shape: [1, 197, 768]
    result = tensor[:, :, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # result.shape: [1, 14, 14, 768]

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    # result.shape: [1, 768, 14, 14]
    return result

gpu_id = 'cuda:0'

# replay casia Oulu MSU
dataset = "MSU"

livedata = np.load('/shared/alwayswithme/data/domain-generalization/' + dataset + '_images_live.npy')
len_livedata = len(livedata)
spoofdata = np.load('/shared/alwayswithme/data/domain-generalization/' + dataset + '_images_spoof.npy')
len_spoofdata = len(spoofdata)

total_data = np.concatenate((livedata, spoofdata), axis=0)
total_data = torch.tensor(np.transpose(total_data, (0, 3, 1, 2)))

fake_label = np.zeros((len_spoofdata), dtype=np.int64) 
real_label = np.ones((len_livedata), dtype=np.int64)

real_fake_label = np.concatenate((real_label, fake_label), axis=0) 
real_fake_label = torch.tensor(real_fake_label)

trainset_D = torch.utils.data.TensorDataset(total_data, real_fake_label)

batch_size = 1
trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=batch_size, shuffle=False)

C_model = vit_base_patch16_224(pretrained=True).to(gpu_id)

model_path = "/shared/huiyu8794/Transformer/test_O/FASNet-162.tar" # load your pretrained model

target_layers = [C_model.patch_embed]  # Specific layer of pretrained model

C_model.load_state_dict(torch.load(model_path, map_location=gpu_id)) 
C_model.eval()

# Transformer need "reshape_transform" argument, CNN not
cam = GradCAM(model=C_model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=True)

for i, data in enumerate(trainloader_D, 0):
    print(str("{0:04d}".format(i)))
    image, labels = data
    image = image.to(gpu_id)
    labels = labels.to(gpu_id)
    
    Resize = T.Resize([224,224])
    image = Resize(image)
    grayscale_cam = cam(input_tensor=image)

    grayscale_cam = grayscale_cam[0, :] 
    grayscale_cam = cv2.medianBlur(grayscale_cam, 5) 
    
    # Draw original image + cam image
    cam_image = show_cam_on_image(np.transpose(image[0].detach().cpu().numpy()*255, (1, 2, 0)), grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    # Save original image
    imshow_np(np.transpose(NormalizeData(image[0, :, :, :].cpu().detach().numpy()), (1, 2, 0)),
              str("{0:04d}".format(i)) + "_" + str(labels[0].item()), image_dir)

    save_file = "/home/huiyu8794/Transformer/activation_image/" + dataset +"_LayerCAM"+ "/" 
    mkdir(save_file)
    
    # Save original image + cam image
    cv2.imwrite(save_file + str("{0:04d}".format(i)) + "_cam" + ".jpg", cam_image)
