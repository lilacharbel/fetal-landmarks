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

model_path = '../fetal-landmarks/models/99/epoch0024_27-07_1529_nme2.6885'
path_split = model_path.split('/')
fig_dir = '/'.join(path_split[:-1])+'/plots/'
stats_file_dir = '/'.join(path_split[:-1])+'/stats.xlsx'

# config
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
    'csv': model_path+'_val_data.xlsx',
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

def create_dataloaders(cuda, batch_size, db_params, data):

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

    return loader


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = create_device(cuda)

# create hrnet nodel
model, _, _, _, _ = create_model(optimizer_params)

# import model and validation dataset
model.load_state_dict(torch.load(model_path+'.statedict.pkl'))
model=model.to(device)

# create validation dataloader
dataloaders = create_dataloaders(cuda, batch_size, db_params, data)
val_data = dataloaders.dataset.metadata
val_data = val_data.set_index('index')
val_data = val_data.drop(['level_0', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
val_idx = val_data.index


# measurements of validation
with torch.no_grad():
    model.eval()

    i=0

    for inputs, labels, target_maps in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        target_maps = target_maps.to(device).float()
        
        index = val_idx[i]
        print('index:', index)
        
        _, TCD_slice = torch.max(labels.data,dim=0)
        print('TCD_slice:', TCD_slice)
        
        pix_len = val_data.loc[index,'resX']
        print('pix_len:', pix_len)
        
        TCD = val_data.loc[index,'TCD']
        print('TCD:', TCD)
        
        targets = ptsFromGaussian(target_maps) * pix_len
        targets = targets[0]
        
        TCD_calc = scipy.spatial.distance.euclidean(targets[0,:], targets[1,:]) 
        print('TCD_calc:', TCD_calc)
        
        factor = TCD/ TCD_calc
        print('factor:', factor)

        # find predictions

        outputs, out_maps = model(inputs)
        
        output_sfmax = torch.nn.functional.softmax(outputs,dim=1)
        _, TCD_slice_pred = torch.max(output_sfmax[:,1],dim=0)
        print('TCD_slice_pred:', TCD_slice_pred)

        TCD_slice_shift = abs(float(TCD_slice_pred) - float(TCD_slice))
        print('TCD_slice_shift:', TCD_slice_shift)

        preds = ptsFromGaussian(out_maps) * pix_len
        presd_pos = preds[TCD_slice,:,:]

        TCD_pred = scipy.spatial.distance.euclidean(preds[0,0,:], preds[0,1,:]) * factor
        print('TCD_pred:', TCD_pred)
        print('------------------------------------\n')
        
        d = {'index':index, 'TCD_slice':TCD_slice, 'TCD_slice_pred':TCD_slice_pred, 'slice_shift':TCD_slice_shift, 'targets':targets, 'preds':preds, 'preds pos slice:': presd_pos, 'TCD':TCD, 'TCD_pred':TCD_pred, 'TCD diff':TCD_pred-TCD}
        
        if i==0:
            df = pd.DataFrame([d])
        else:
            df = df.append([d], ignore_index = True)
        
        plt.figure()
        plt.ion()
        plt.imshow(inputs[TCD_slice,0,:,:].cpu().numpy())
        plt.plot(targets[:,1]/pix_len, targets[:,0]/pix_len, color = 'blue')
        plt.plot(presd_pos[:,1]/pix_len, presd_pos[:,0]/pix_len, color = 'red')
        plt.title('Targets = Blue, Predictions = Red')
        plt.ioff()

        # plt.show()
        
        fig_name = 'val{}.png'.format(index)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(fig_dir+fig_name)

        i += 1 

# save df
df.to_excel(stats_file_dir)

# # bland altman
# plt.figure()
# plt.ion()
# plt.scatter((df['TCD'] + df['TCD_pred'])/2., df['TCD'] - df['TCD_pred'])
# plt.title('Bland Altman')
# plt.ioff()
# plt.savefig(fig_dir+'bland_altman.png')
# # plt.show()

plt.figure()
plt.ion()
sm.graphics.mean_diff_plot(df['TCD'], df['TCD_pred'])
plt.ioff()
plt.savefig(fig_dir+'bland_altman.png')