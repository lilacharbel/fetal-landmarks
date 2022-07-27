
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import time
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for HRNet
import sys
# sys.path.append("/media/df4-projects/Lilach/HRNet-Image-Classification")
# import tools._init_paths
import _init_paths
#import models as models_hrnet
import mod_hrnet
from config import config
from config import update_config
# from utils.utils import get_optimizer
from utils import get_optimizer
#

from loader import NiftiDataset, random_split
import transforms as tfs

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import MongoObserver

from sklearn.mixture import GaussianMixture

save_dir = os.path.join(".", "models")

writer = SummaryWriter('runs')
writer.add_text('mini', 'minimal model', 0)


ex = Experiment('slicenet')
ex.observers.append(MongoObserver.create(url=f'mongodb://mongo_user:mongo_password@localhost:27017/?authMechanism=SCRAM-SHA-1', db_name='sacred'))

def getTimeName():
    """Return the current time in format <day>-<month>_<hour><minute> for use in filenames."""
    from datetime import datetime
    t = datetime.now()
    return "{:02d}-{:02d}_{:02d}{:02d}".format(t.day,t.month,t.hour,t.minute)

def save_model(model, epoch, directory, metrics, val_data, filename=None):
    """Save the state dict of the model in the directory,
    with the save name metrics at the given epoch.
    epoch: epoch number(<= 4 digits)
    directory: where to save statedict
    metrics: dictionary of metrics to append to save filename
    filename: if a name is given, it overrides the above created filename
    Returns the save file name
    """
    # save state dict
    postfix = ""
    if filename is None:
        filename = f"epoch{epoch:04d}_{getTimeName()}_"
        postfix = "_".join([f"{name}{val:0.4f}" for name, val in metrics.items()])

    valdata_name = os.path.join(directory, filename + postfix + "_val_data.csv")
    filename = os.path.join(directory, filename + postfix + ".statedict.pkl")
    
    print('filename:', filename)
    print('valdata_name:', valdata_name)

    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save(state, filename)
    val_data.to_csv(valdata_name)

    print(f"Saved model at {filename}")

    return filename


def ptsFromGaussian(maps):
    maps = maps.cpu().detach().numpy()
    M = np.size(maps, 0)  # number of imgs
    N = np.size(maps, 1)  # number of points = 2
    pts = np.empty([M, N, 2])

    for i in range(M):
        for j in range(N):
            pts[i, j, 0] = np.argmax(np.amax(maps[i, j, :, :], 1))
            pts[i, j, 1] = np.argmax(np.amax(maps[i, j, :, :], 0))

    return pts


def compute_nme(preds, targets):
    preds_pts = ptsFromGaussian(preds)
    targets_pts = ptsFromGaussian(targets)

    N = preds.shape[0]  # number of slices
    L = preds.shape[1]  # number of points = 2
    rmse = np.zeros(N)

    for i in range(N):
        rmse[i] = np.sum(np.linalg.norm(preds_pts - targets_pts, axis=1)) / L

    return rmse


def determine_direction(pts_arr, do_plot=False):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(pts_arr)
    if do_plot:
        plt.scatter(pts_arr[::2,0], pts_arr[::2,1], alpha=.1)
        plt.scatter(pts_arr[1::2,0], pts_arr[1::2,1], color='r',alpha=.1)
        plt.plot(gmm.means_[:,0], gmm.means_[:,1], color='k')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
    return gmm.means_


def classify_direction(target_maps, d_pts):
    target_dim = target_maps.size()
    
    pts_arr = ptsFromGaussian(target_maps)
    d_vector = d_pts[1, :] - d_pts[0, :]

    ap = pts_arr  
    pt_part1 = np.dot(d_vector[np.newaxis, :], ap.T) / np.linalg.norm(d_vector)
    pts_class = pt_part1.flatten()
    pts_class = pts_class.reshape(-1, 2)
    pts_class = np.stack((np.argmin(pts_class, axis=1), np.argmax(pts_class, axis=1)), axis=1) 
    
    pts_remap = np.concatenate((np.sort(np.tile(np.arange(pts_class.shape[0]),2)) , np.reshape(pts_class, (-1))), axis=0)
    pts_remap = np.transpose(np.reshape(pts_remap, (-1,target_maps.shape[0]*target_maps.shape[1])))
    target_maps = target_maps[pts_remap[:, 0],pts_remap[:,1]]
    target_maps = np.reshape(target_maps, target_dim)
    
    return target_maps


@ex.capture
def create_dataloaders(cuda, batch_size, db_params, data):
    #Input validation
    if not os.path.exists(db_params['csv']):
        raise ValueError('CSV file not exists {}'.format(db_params['csv']))

    if not os.path.isdir(db_params['root_dir']):
        raise ValueError('Root dir for DB does not exists {}'.format(db_params['root_dir']))
    #Dataloader creation

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda in [0,1] else {}
    dataset = NiftiDataset(csv_file=db_params['csv'],
                           nii_dir=db_params['root_dir'],
                           seg_dir=db_params['seg_dir'],
                           filter_quality=db_params['quality'],
                           only_tag=True,
                           tagname=data["selection_idx"],
                           transform=transforms.Compose([
                            tfs.CreateGaussianTargets(sigma=data['sigma'], measure_name=data['measure_idx']),
                            #tfs.RandomRotate(),
                            tfs.cropByBBox(min_upcrop=1.0, max_upcrop=1.3),
                            tfs.PadZ(data['context']),
                            tfs.Rescale((224,224)),

                            tfs.ToTensor(),
                            tfs.Normalize(mean=0.456,
                                            std=0.224),
                            tfs.SampleFrom3D(db_params['pos_neg_ratio'],
                                             sample_idx=data['selection_idx'],
                                             context=data['context']),
                            #tfs.RandomFlip(),
                            tfs.toXYZ("image", data['selection_idx'], "target_maps"),
                           ]))
    dataset_size = len(dataset) 
    train_size = int(db_params['train_test_split'] * dataset_size)
    test_size = dataset_size - train_size
    (train_ds, test_ds) , (train_idx, test_idx) = random_split(dataset, [train_size, test_size])

    #It is not going to be elegant.... special augmentation for test
    test_ds.dataset = copy.copy(dataset)
    test_ds.dataset.transform = transforms.Compose([

                                            tfs.CreateGaussianTargets(sigma=data['sigma'], measure_name=data['measure_idx']),
                                            tfs.cropByBBox(min_upcrop=None, max_upcrop=None),
                                            tfs.PadZ(data['context']),
                                            tfs.Rescale((224,224)),

                                            tfs.ToTensor(),
                                            tfs.Normalize(mean=0.456,
                                                            std=0.224),
                                            tfs.SampleFrom3D(0,
                                                             sample_idx=data['selection_idx'],
                                                             context=data['context']),
                                            #tfs.RandomFlip(),
                                            tfs.toXYZ("image", data['selection_idx'], "target_maps"),
                                           ])
    train_val_ds = copy.copy(test_ds)
    train_val_ds.indices = train_ds.indices

    ex.info["db_split_train"] = dataset.get_metadata(train_idx).to_json()
    ex.info["db_split_test"] = dataset.get_metadata(test_idx).to_json()
    train_loader = torch.utils.data.DataLoader(train_ds,
                                    batch_size=batch_size, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                    batch_size=1, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)
    train_val_loader = torch.utils.data.DataLoader(train_val_ds,
                                    batch_size=1, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)


    dataloaders = {}
    dataloaders["train"] = train_loader
    dataloaders["val"] = test_loader
    dataloaders["val_train"] = train_val_loader
    return dataloaders

@ex.capture
def create_model(optimizer_params):
    
    config.merge_from_file("/media/df4-projects/Lilach/fetal-landmarks/cls_landmark_config.yaml")
    config['MODEL']['PRETRAINED'] = "/hrnet_w18_small_model_v2.pth"
    model_ft = mod_hrnet.get_HRnet(config)
    pretrained = "/hrnet_w18_small_model_v2.pth"
    model_ft.init_weights(pretrained)

    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)


    criterion_cls = nn.CrossEntropyLoss()
    criterion_lm = nn.MSELoss(size_average=True)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=optimizer_params['lr'], momentum=optimizer_params['momentum'])
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=optimizer_params['lr'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=optimizer_params['step_size'], gamma=optimizer_params['gamma'])
    # exp_lr_scheduler= lr_scheduler.CyclicLR(optimizer_ft, base_lr=optimizer_params['lr'], max_lr=optimizer_params['lr']*10)

    

    # optimizer_ft = get_optimizer(config, model_ft)
    # last_epoch = config.TRAIN.BEGIN_EPOCH

    # if isinstance(config.TRAIN.LR_STEP, list):
    #     exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer_ft, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
    #         last_epoch-1
    #     )
    # else:
    #     exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer_ft, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
    #         last_epoch-1
    #     )

    # ####

    return model_ft, criterion_cls, criterion_lm, optimizer_ft, exp_lr_scheduler

@ex.capture
def train_model(model, criterion_cls, criterion_lm, optimizer, scheduler, dataloaders, device, _run, num_epochs, optimizer_params, data):
    since = time.time()
    run_dir = os.path.join(save_dir, str(_run._id))
    os.makedirs(run_dir, exist_ok=True)
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_epoch = 0
    best_loss = 100
    best_nme = 100
    nme = 0

    # find d_vector
    data_table = dataloaders['train'].dataset.dataset.metadata
    train_idx = dataloaders['train'].dataset.indices
    train_data = data_table.iloc[train_idx]
    
    columns_to_ids = ["_X1", "_X2", "_Y1", "_Y2"]
    measure_names = [data['measure_idx'] + a for a in columns_to_ids]
    measure_names_idx = [train_data.columns.get_loc(col) for col in measure_names]

    allpts = []
    for idx, l in train_data.iterrows():

        # filename = "Pat{patient_id:02}_Se{series:02}_Res{res_x}_{res_y}_Spac{res_z:.1f}.nii"
        # fn = filename.format(patient_id=l["Subject"],
        #                     series=l["SeriesNum"],
        #                     res_x=l["resX"],
        #                     res_y=l["resY"],
        #                     res_z=l["resZ"],)

        # img_name = os.path.join(db_params['root_dir'], fn)
        # image = nibabel.load(img_name).get_fdata().astype(np.float)
        # f_dim = image.shape[:-1][::-1]
        f_dim = (512, 512)
        
        pts = l[measure_names_idx].values
        
        pts = pts.astype('float').reshape(-1, 2) / f_dim
        allpts.append(pts)

    allpts = np.concatenate(allpts)
    d_vector = determine_direction(allpts, do_plot=True)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch == optimizer_params['freeze_from']:
            print('Freezing model')
            for param in model.parameters():
                param.requires_grad = False
            #Unfreeze last layer

            for param in model.classifier.parameters():
                param.requires_grad = True


        #if epoch % 10 == 0:
        #dataloaders['train'].dataset.dataset.transform.transforms[6].negative_slides = int((30 - epoch) / 3) + 1
        #dataloaders['train'].dataset.dataset.transform.transforms[6].negative_slides = int(((epoch) / 3)) + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val_train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_choose_acc = 0.0
            running_shift = 0.0

            nme_count = 0
            nme_batch_sum = 0
            nme = 0

            epoch_elem_size = 0
            # Iterate over data.
            for inputs, labels, target_maps in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                # target_maps = target_maps.to(device).float()
                target_maps = classify_direction(target_maps, d_vector).to(device).float()

                epoch_elem_size += inputs.size(0)
                # zero the parameter gradients
                if phase == 'train':
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, output_maps = model(inputs)
                    

                    # take only positive slices
                    slice_idxs = np.nonzero(labels)[:,0].cpu().numpy()

                    _, preds = torch.max(outputs, 1)
                    loss = (criterion_lm(output_maps[slice_idxs,:,:,:], target_maps) + criterion_cls(outputs, labels))/2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                nme_temp = compute_nme(output_maps, target_maps)
                nme_batch_sum += np.sum(nme_temp)
                nme_count = nme_count + output_maps.size(0)

                if phase in ['val', 'val_train']:
                    output_sfmax = torch.nn.functional.softmax(outputs,dim=1)
                    max_val, max_idx = torch.max(output_sfmax[:,1],dim=0)
                    max_val_lbl, max_val_idx = torch.max(labels.data,dim=0)
                    running_choose_acc += (1. - (abs(float(max_idx) - float(max_val_idx))/float(inputs.size(0))))
                    running_shift += abs(float(max_idx) - float(max_val_idx))
                    #print(max_idx, max_val_idx, running_idx)
                    

            epoch_loss = running_loss / epoch_elem_size
            epoch_acc = running_corrects.double() / epoch_elem_size
            nme = nme_batch_sum / nme_count

            if phase in ['val', 'val_train']:
                epoch_choose_acc = running_choose_acc / len(dataloaders[phase])
                epoch_choose_shift = running_shift / len(dataloaders[phase])
                print('{} Loss: {:.4f} NME: {:.4f} Acc: {:.4f} ChooseAcc {:.4f} ChooseShift {:.4f}'.format(phase, epoch_loss, nme, epoch_acc, epoch_choose_acc, epoch_choose_shift))
                # print('{} Loss: {:.4f}  NME: {:.4f}'.format(phase, epoch_loss, nme))
            else:
                print('{} Loss: {:.4f} NME: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, nme, epoch_acc))
                # print('{} Loss: {:.4f}  NME: {:.4f}'.format(phase, epoch_loss, nme))


            if phase == 'val':
                # Calculate specific Val accuracy
                # Assuming BS=1 in Val
                _run.log_scalar('val_choose_acc', float(epoch_choose_acc))
                _run.log_scalar('val_choose_shift', float(epoch_choose_shift))
                _run.log_scalar('val_loss', float(epoch_loss))
                _run.log_scalar('val_accuracy', float(epoch_acc))
                _run.log_scalar('val_nme', float(nme))
            elif phase == 'val_train':
                _run.log_scalar('vtrain_choose_acc', float(epoch_choose_acc))
                _run.log_scalar('vtrain_choose_shift', float(epoch_choose_shift))
                _run.log_scalar('vtrain_loss', float(epoch_loss))
                _run.log_scalar('vtrain_accuracy', float(epoch_acc))
                _run.log_scalar('vtrain_nme', float(nme))
            elif phase == 'train':
                _run.log_scalar('train_loss', float(epoch_loss))
                _run.log_scalar('train_accuracy', float(epoch_acc))
                _run.log_scalar('train_nme', float(nme))


            # deep copy the model
            # if phase == 'val' and epoch_choose_acc > best_acc:
            if phase == 'val' and nme < best_nme:
                # best_acc = epoch_choose_acc
                best_nme = nme
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_epoch = epoch
            if phase == 'train':
                scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best nme: {:4f}'.format(best_nme))

    # save the validation set
    val_idx = dataloaders['val'].dataset.indices
    val_data = data_table.iloc[val_idx]

    # Save last epoch model
    # model_fname = save_model(model, epoch, run_dir, {'choose_acc' : epoch_choose_acc})
    model_fname = save_model(model, epoch, run_dir, {'nme': nme}, val_data)
    _run.add_artifact(model_fname, "model_lastepoch")

    # load best model weights
    model.load_state_dict(best_model_wts)

    # model_fname = save_model(model, best_model_epoch, run_dir, {'choose_acc' : best_acc})
    model_fname = save_model(model, best_model_epoch, run_dir, {'nme': nme}, val_data)
    _run.add_artifact(model_fname, "model")
    return model

@ex.config
def get_config():
    batch_size = 6
    num_epochs = 25
    cuda = 1
    # basenet = 'HRNet'
    data = {
        'context': 0,
        'selection_idx': 'TCD_Selection',
        'measure_idx': 'Measure_TCD',
        'sigma': 2.
    }
    db_params = {
        'root_dir': '/media/df4-projects/Lilach/Data/dataset/',
        'seg_dir': '/media/df4-projects/Lilach/Data/seg/',
        'csv': '/media/df4-projects/Lilach/Data/data_set.xlsx',
        'quality': None,
        'pos_neg_ratio': 0,
        'train_test_split': 0.8,
    }
    optimizer_params = {
        'lr' : 0.0001,
        'momentum' : 0.9,
        'step_size' : 7,
        'gamma' : 0.1,
        'freeze_from' : -1,
    }


# @ex.config
# def initialize(cuda):#, seed):
    # # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=8, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=2, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # args = parser.parse_args(args_str)
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(seed)
    # if cuda > 0:
    #     torch.cuda.manual_seed(seed)

@ex.capture()
def create_device(cuda):
    cuda_int = int(cuda)
    torch.cuda.set_device(cuda_int)
    device = torch.device("cuda:%d" % (cuda_int, ))
    return device

@ex.main
def main():
    # initialize()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = create_device()
    dataloaders = create_dataloaders()

    model, criterion_cls, criterion_lm, optimizer, scheduler = create_model()
    model = train_model(model, criterion_cls, criterion_lm, optimizer, scheduler, dataloaders, device)
    

if __name__ == '__main__':
    ex.run_commandline()
