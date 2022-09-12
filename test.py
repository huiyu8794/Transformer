import numpy as np
import os
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import torch
from models.ViT import *
import timm
import logging
from pytz import timezone
from datetime import datetime
import torchvision.transforms as transforms
import sys
import torch.nn.functional as F
import torch.nn as nn

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device_id = 'cuda:0'
batch_size = 50

### DG
# test_dataset = 'replay'
# live_path = '/shared/domain-generalization/' + test_dataset + '_images_live.npy'
# spoof_path = '/shared/domain-generalization/' + test_dataset + '_images_spoof.npy'

### Oulu intra
prot = 'p3_6'
live_path = '/shared/Processed_Oulu/test_real_images_' + prot + ".npy"
spoof_path ='/shared/Processed_Oulu/test_fake_images_' + prot + ".npy"


result_filename = 'oulu_' + prot
result_path = '/shared/huiyu8794/Transformer/' + result_filename

file_handler = logging.FileHandler(filename='/home/huiyu8794/Transformer/logger/'+ result_filename +'_test.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
# logging.info(f"Test on {test_dataset}")



live_data = np.load(live_path)
spoof_data = np.load(spoof_path)
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))

data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Net = swin_s(weights=Swin_S_Weights.DEFAULT, progress=True).to(device_id)
# Fas_Net = timm.create_model('vit_relpos_base_patch32_plus_rpn_256', pretrained=False, num_classes=2).to(device_id)
# Fas_Net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to(device_id)
# Fas_Net = vit_base_patch16_224(pretrained=True).cuda() #.to(device_id)
Fas_Net = vit_base_patch16_224(pretrained=True).to(device_id)

logging.info(f"# of testing: {len(total_data)}")
logging.info(f"path: {result_path}")

for epoch in range(1, 500):

    Net_path = result_path + "/FASNet-" + str(epoch) + ".tar"
    Fas_Net.load_state_dict(torch.load(Net_path)) # map_location=device_id
    Fas_Net.eval()

    score_list = []
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = images.to(device_id)
        Resize = transforms.Resize([224, 224])
        images = Resize(images)
        label_pred = Fas_Net(NormalizeData_torch(images), train=False)
        # score = (label_pred).cpu().detach().numpy()
        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]  # multi class

        for j in range(images.size(0)):
            score_list.append(score[j])
            label_list.append(labels[j])

    score_list = NormalizeData(score_list)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, score_list)
    threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    for i in range(len(score_list)):
        score = score_list[i]
        if (score >= threshold and label_list[i] == 1):
            TP += 1
        elif (score < threshold and label_list[i] == 0):
            TN += 1
        elif (score >= threshold and label_list[i] == 0):
            FP += 1
        elif (score < threshold and label_list[i] == 1):
            FN += 1

    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)

    acer = '{:.5f}'.format(np.round((APCER + NPCER) / 2, 4))
    apcer = '{:.5f}'.format(np.round(APCER, 4))
    npcer = '{:.5f}'.format(np.round(NPCER, 4))
    auc = '{:.5f}'.format(roc_auc_score(label_list, score_list))

    # logging.info(f"Epoch {epoch} ACER {acer} AUC {auc}")
    logging.info(f"Epoch {epoch}  ACER {acer}  APCER {apcer}  NPCER {npcer}")
    