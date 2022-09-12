import torch
from models.ViT import *
import torch.optim as optim
import torch.nn as nn
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import logging
from pytz import timezone
from datetime import datetime
import timm
import sys
import torchvision.transforms as T

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def imshow_np(img, filename):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./save_image_oulu/dynamic/" + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()


def get_data_loader(data_path="", data_path2="live", batch_size=5, shuffle=True, drop_last=True):
    # data path
    data = None
    live_spoof_label = None
    live_related = None
    spoof_related = None
    material_label = None

    if data_path2 == "live":
        data = np.load(data_path)
        material_label = np.ones(len(data), dtype=np.int64)

        live_spoof_label = np.ones(len(data), dtype=np.int64)
        # live_related = np.load(amap_path)
        bimap_labels = np.ones((len(data), 256, 16, 16), dtype=np.float32)
    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)

        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2
        material_label = np.concatenate((print_lab, replay_lab), axis=0)

        live_spoof_label = np.zeros(len(data), dtype=np.int64)
        bimap_labels = np.zeros((len(data), 256, 16, 16), dtype=np.float32)


    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label),
                                              torch.tensor(bimap_labels))
    # free memory
    import gc
    del data
    # del live_related
    # del spoof_related
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader

def get_inf_iterator(data_loader):
    while True:
        for images, live_spoof_labels, bimap_labels in data_loader:
            yield (images, live_spoof_labels, bimap_labels)



def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

# seed = 666
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

device_id = 'cuda:0'
batch_size = 10

prot = 'p3_6'
results_filename = 'oulu_' + prot
results_path = '/shared/huiyu8794/Transformer/' + results_filename

file_handler = logging.FileHandler(filename='/home/huiyu8794/Transformer/logger/' + results_filename + '_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# replay casia Oulu MSU
dataset1 = "oulu"
logging.info(f"Train on {dataset1}")

live_path = '/shared/Processed_Oulu/train_live_' + prot + ".npy"
print_path = '/shared/Processed_Oulu/train_print_' + prot + ".npy"
replay_path = '/shared/Processed_Oulu/train_replay_' + prot + ".npy"

# Fas_Net = timm.create_model('vit_relpos_base_patch32_plus_rpn_256', pretrained=False, num_classes=2).to(device_id)
# Fas_Net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to(device_id)
Fas_Net = vit_base_patch16_224(pretrained=True).to(device_id)

criterion_ce = nn.CrossEntropyLoss().to(device_id)
criterionCls = nn.CrossEntropyLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id)
optimizer = optim.RAdam(Fas_Net.parameters(), lr=1e-5, betas=(0.5, 0.8)) #
# optimizer = optim.Adam(Fas_Net.parameters(), lr=1e-5, betas=(0.5, 0.8),weight_decay=1e-6)
Fas_Net.train()

model_save_step = 50
model_save_epoch = 1

save_index = 0

data_real = get_data_loader(data_path=live_path, data_path2="live",
                             batch_size=batch_size, shuffle=True)

data_fake = get_data_loader(data_path=print_path, data_path2=replay_path,
                             batch_size=batch_size, shuffle=True)


iternum = max(len(data_real), len(data_fake))
log_step = 20
logging.info(f"iternum={iternum}")

data_real = get_inf_iterator(data_real)
data_fake = get_inf_iterator(data_fake)

T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop(224),
)


for epoch in range(200):

    for step in range(iternum):
        # ============ one batch extraction ============#
        img_real, ls_lab_real, map_lab_real = next(data_real)
        img_fake, ls_lab_fake, map_lab_fake = next(data_fake)

        # ============ one batch collection ============#
        catimg = torch.cat([img_real, img_fake], 0).to(device_id)
        ls_lab = torch.cat([ls_lab_real, ls_lab_fake], 0).to(device_id)
        map_lab = torch.cat([map_lab_real, map_lab_fake], 0).to(device_id)

        batchidx = list(range(len(catimg)))
        random.shuffle(batchidx)

        img_rand = catimg[batchidx, :]
        ls_lab_rand = ls_lab[batchidx]
        map_lab_rand = map_lab[batchidx]

        # Resize = transforms.Resize([224, 224])
        img_rand = T_transform(img_rand) 

        pred = Fas_Net(NormalizeData_torch(img_rand)) 

        Loss_cls = criterionCls(pred.squeeze(), ls_lab_rand)
        # Loss_map = criterionCls(pred_bimap, map_lab_rand)

        Loss_all = Loss_cls 

        optimizer.zero_grad()
        Loss_all.backward()
        optimizer.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Loss_cls %.8f'
                  % (epoch, step, Loss_cls.item()))

        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path, "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path, "FASNet-{}.tar".format(save_index)))
