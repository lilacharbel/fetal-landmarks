import pickle
import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import os

from trainer import create_model, create_device, ptsFromGaussian
from loader import NiftiDataset, random_split
import transforms as tfs

#################################
##### experiment parameters #####
#################################
model_path = '../fetal-landmarks/models/154/epoch0001_29-08_1133_nme96.5424'
measure = 'TCD'


path_split = model_path.split('/')
folder_dir = '/'.join(path_split[:-1])#+'/plots/'
# stats_file_dir = '/'.join(path_split[:-1])#+'/stats.xlsx'

# config
batch_size = 6
num_epochs = 25
cuda = 0
# basenet = 'HRNet'
data = {
    'context': 0,
    'selection_idx': measure+'_Selection',
    'measure_idx': 'Measure_'+measure,
    'sigma': 2.
}
db_params = {
    'root_dir': '/media/df4-projects/Lilach/Data/dataset/',
    'seg_dir': '/media/df4-projects/Lilach/Data/seg/',
    'train_csv' : model_path+'_train_data.xlsx',
    'val_csv': model_path+'_val_data.xlsx',
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

def create_dataloaders(cuda, batch_size, db_params, data, csv):

    #Dataloader creation

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda in [0,1] else {}
    dataset = NiftiDataset(csv_file=csv,
                           nii_dir=db_params['root_dir'],
                           seg_dir=db_params['seg_dir'],
                           filter_quality=db_params['quality'],
                           only_tag=True,
                           tagname=data["selection_idx"],
                           transform=transforms.Compose([
                            tfs.CreateGaussianTargets(sigma=data['sigma'], measure_name=data['measure_idx']),
                            tfs.cropByBBox(min_upcrop=None, max_upcrop=None),
                            tfs.PadZ(data['context']),
                            tfs.Rescale((224,224)),

                            tfs.ToTensor(),
                            tfs.Normalize(mean=0.456,
                                            std=0.224),
                            tfs.SampleFrom3D(None,
                                             sample_idx=data['selection_idx'],
                                             context=data['context']),
                            #tfs.RandomFlip(),
                            tfs.toXYZ("image", data['selection_idx'], "target_maps"),
                           ]))

    loader = torch.utils.data.DataLoader(dataset,
                                    batch_size=1, shuffle=False, collate_fn=tfs.custom_collate_fn, **kwargs)

    data = loader.dataset.metadata
    data = data.set_index('index')
    data = data.drop(['level_0', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    idx = data.index

    return loader, data, idx

def measures(name, data, loader, idx, folder_dir):
    with torch.no_grad():
        model.eval()

        i=0

        for inputs, labels, target_maps in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            target_maps = target_maps.to(device).float()
            
            index = idx[i]
            print('index:', index)
            
            _, target_slice = torch.max(labels.data,dim=0)
            print(measure+'_slice:', target_slice)
            
            pix_len = data.loc[index,'resX']
            print('pix_len:', pix_len)
            
            target_measure = data.loc[index,'TCD']
            print(measure+':', target_measure)
            
            targets = ptsFromGaussian(target_maps) * pix_len
            targets = targets[0]
            
            measure_calc = scipy.spatial.distance.euclidean(targets[0,:], targets[1,:]) 
            print(measure+'_calc:', measure_calc)
            
            factor = target_measure/ measure_calc
            print('factor:', factor)

            # find predictions

            outputs, out_maps = model(inputs)
            
            output_sfmax = torch.nn.functional.softmax(outputs,dim=1)
            _, pred_slice = torch.max(output_sfmax[:,1],dim=0)
            print(measure+'_slice_pred:', pred_slice)

            slice_shift = abs(float(pred_slice) - float(target_slice))
            print(measure+'_slice_shift:', slice_shift)

            preds = ptsFromGaussian(out_maps) * pix_len
            presd_pos = preds[target_slice,:,:]

            pred_measure = scipy.spatial.distance.euclidean(presd_pos[0,:], presd_pos[1,:]) * factor
            print(measure+'_pred:', pred_measure)
            print('------------------------------------\n')
            
            d = {'index':index, measure+' slice':target_slice.item(), measure+' slice pred':pred_slice.item(), 'slice shift':slice_shift,
            'targets':targets, 'preds': presd_pos, 'preds all slices':preds, 
            measure:target_measure, measure+' pred':pred_measure,
            measure+' diff':abs(pred_measure-target_measure), measure+' error':abs(pred_measure-target_measure)/target_measure}
            
            if i==0:
                df = pd.DataFrame([d])
            else:
                df = df.append([d], ignore_index = True)
            
            plt.figure()
            plt.ion()
            plt.imshow(inputs[target_slice,0,:,:].cpu().numpy(), cmap='Greys_r')
            plt.plot(targets[:,1]/pix_len, targets[:,0]/pix_len, color = 'blue')
            plt.plot(presd_pos[:,1]/pix_len, presd_pos[:,0]/pix_len, color = 'red')
            plt.title(measure)
            plt.legend(['Target', 'Prediction'])
            plt.ioff()

            # plt.show()
            
            fig_dir = folder_dir+'/'+name+'_plots/'
            fig_name = '{}{}.png'.format(name, index)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(fig_dir+fig_name)

            i += 1 

    df = df.set_index('index')

    # add mean row
    df.loc['mean'] = df[['slice shift', measure+' diff', measure+' error']].mean()

    # save df
    stats_file_dir = folder_dir+'/'+name+'_stats.xlsx'
    df.to_excel(stats_file_dir)

    plt.figure()
    plt.ion()
    sm.graphics.mean_diff_plot(df['TCD'], df['TCD pred'])
    plt.ioff()
    plt.savefig(fig_dir+'bland_altman.png')

# 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = create_device(cuda)

# create hrnet nodel
model, _, _, _, _ = create_model(optimizer_params)

# import model and validation dataset
model.load_state_dict(torch.load(model_path+'.statedict.pkl'))
model=model.to(device)

# create validation dataloader
val_loader, val_data, val_idx = create_dataloaders(cuda, batch_size, db_params, data, db_params['val_csv'])
# val_data = val_loaders.dataset.metadata
# val_data = val_data.set_index('index')
# val_data = val_data.drop(['level_0', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
# val_idx = val_data.index

# train dataset
train_loader, train_data, train_idx = create_dataloaders(cuda, batch_size, db_params, data, db_params['train_csv'])
# train_data = train_loaders.dataset.metadata
# train_data = train_data.set_index('index')
# train_data = train_data.drop(['level_0', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
# train_idx = train_data.index

# make measurements
measures('val', val_data, val_loader, val_idx, folder_dir)
measures('train', train_data, train_loader, train_idx, folder_dir)

# all_data = pd.read_excel(db_params['csv'])
# all_data = all_data.drop(['Unnamed: 0'], axis=1)
# train_data = all_data.drop(val_idx)

# train_name = model_path+'_train_data.xlsx'
# train_data.to_excel(train_name)
# train_dataloaders = create_dataloaders(cuda, batch_size, db_params, data, train_name)


# measurements of validation
# with torch.no_grad():
#     model.eval()

#     i=0

#     for inputs, labels, target_maps in val_loaders:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         target_maps = target_maps.to(device).float()
        
#         index = val_idx[i]
#         print('index:', index)
        
#         _, target_slice = torch.max(labels.data,dim=0)
#         print(measure+'_slice:', target_slice)
        
#         pix_len = val_data.loc[index,'resX']
#         print('pix_len:', pix_len)
        
#         target_measure = val_data.loc[index,'TCD']
#         print(measure+':', target_measure)
        
#         targets = ptsFromGaussian(target_maps) * pix_len
#         targets = targets[0]
        
#         measure_calc = scipy.spatial.distance.euclidean(targets[0,:], targets[1,:]) 
#         print(measure+'_calc:', measure_calc)
        
#         factor = target_measure/ measure_calc
#         print('factor:', factor)

#         # find predictions

#         outputs, out_maps = model(inputs)
        
#         output_sfmax = torch.nn.functional.softmax(outputs,dim=1)
#         _, pred_slice = torch.max(output_sfmax[:,1],dim=0)
#         print(measure+'_slice_pred:', pred_slice)

#         slice_shift = abs(float(pred_slice) - float(target_slice))
#         print(measure+'_slice_shift:', slice_shift)

#         preds = ptsFromGaussian(out_maps) * pix_len
#         presd_pos = preds[target_slice,:,:]

#         pred_measure = scipy.spatial.distance.euclidean(presd_pos[0,:], presd_pos[1,:]) * factor
#         print(measure+'_pred:', pred_measure)
#         print('------------------------------------\n')
        
#         d = {'index':index, measure+' slice':target_slice.item(), measure+' slice pred':pred_slice.item(), 'slice shift':slice_shift,
#          'targets':targets, 'preds': presd_pos, 'preds all slices':preds, 
#          measure:target_measure, measure+' pred':pred_measure,
#          measure+' diff':abs(pred_measure-target_measure), measure+' error':abs(pred_measure-target_measure)/target_measure}
        
#         if i==0:
#             df = pd.DataFrame([d])
#         else:
#             df = df.append([d], ignore_index = True)
        
#         plt.figure()
#         plt.ion()
#         plt.imshow(inputs[target_slice,0,:,:].cpu().numpy(), cmap='Greys_r')
#         plt.plot(targets[:,1]/pix_len, targets[:,0]/pix_len, color = 'blue')
#         plt.plot(presd_pos[:,1]/pix_len, presd_pos[:,0]/pix_len, color = 'red')
#         plt.title(measure)
#         plt.legend(['Target', 'Prediction'])
#         plt.ioff()

#         # plt.show()
        
#         fig_name = 'val{}.png'.format(index)
#         if not os.path.exists(fig_dir):
#             os.makedirs(fig_dir)
#         plt.savefig(fig_dir+fig_name)

#         i += 1 

# df = df.set_index('index')

# # add mean row
# df.loc['mean'] = df[['slice shift', measure+' diff', measure+' error']].mean()

# # save df
# df.to_excel(stats_file_dir)

# # # bland altman
# # plt.figure()
# # plt.ion()
# # plt.scatter((df['TCD'] + df['TCD_pred'])/2., df['TCD'] - df['TCD_pred'])
# # plt.title('Bland Altman')
# # plt.ioff()
# # plt.savefig(fig_dir+'bland_altman.png')
# # # plt.show()

# plt.figure()
# plt.ion()
# sm.graphics.mean_diff_plot(df['TCD'], df['TCD_pred'])
# plt.ioff()
# plt.savefig(fig_dir+'bland_altman.png')