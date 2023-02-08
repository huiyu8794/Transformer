# from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from models.ViT_tsne import *
import torch
from matplotlib import markers
import torchvision.transforms as transforms
import os
import warnings
import math
warnings.filterwarnings('ignore')

gpu_id = 'cuda:0'
data_name = 'casia'

# load your trained model
results_filename = 'test_C_nd_mnp'
results_path = '/shared/alwayswithme/PD_Transformer/save_model_fixed/' + results_filename
Net_path = results_path + "/FASNet-90.tar"
Fas_Net = vit_base_patch16_224(pretrained=True).to(gpu_id)

savefig_path = '/home/huiyu8794/Transformer/tsne/'+results_filename+'_allred/'

live_path = '/shared/alwayswithme/data/domain-generalization/' + data_name + '_images_live.npy'
print_path = '/shared/alwayswithme/data/domain-generalization/' + data_name + '_print_images.npy'
replay_path = '/shared/alwayswithme/data/domain-generalization/' + data_name + '_replay_images.npy'

live_data = np.load(live_path)
print_data = np.load(print_path)
replay_data = np.load(replay_path)

spoof_data = np.concatenate((print_data, replay_data), axis=0)
total_data = np.concatenate((live_data, spoof_data), axis=0)

live_label = np.ones(len(live_data), dtype=np.int64)*2
print_label = np.ones(len(print_data), dtype=np.int64)
replay_label = np.zeros(len(replay_data), dtype=np.int64)

spoof_label = np.concatenate((print_label, replay_label), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))

data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1,
                                          shuffle=False)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir(savefig_path)
Fas_Net.load_state_dict(torch.load(Net_path))
Fas_Net.eval()

Resize = transforms.Resize([224, 224])

for tsne_num in range(200):
    print(tsne_num)
    data_list = []
    label_list = []
    quality_list = []
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = images.to(gpu_id)
        labels = labels.to(gpu_id)
        images = Resize(images)
        label_pred, feat = Fas_Net(NormalizeData_torch(images), train=False)
        feat_vec = torch.flatten(feat[0]).detach().cpu().numpy()

        print(feat_vec.shape)

        data_list.append(feat_vec)
        label_list.append(labels)

    # draw samples
    X_tsne = manifold.TSNE(perplexity=5, early_exaggeration=5).fit_transform(data_list)

    # Normalize
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    plt.figure(figsize=(6, 6))

    # customize your marker
    # colors = ['b','g','c','m','k']
    # markers = ['.','^']
    msize = 40
    for i in range(X_tsne.shape[0]):
        if label_list[i]== 2:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="g", s=msize, marker='.')
        elif label_list[i]== 1:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="r", s=msize, marker='.')
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="r", s=msize, marker='.')


    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.savefig(savefig_path + str(tsne_num) + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()
