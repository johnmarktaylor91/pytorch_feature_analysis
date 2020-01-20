import tensorflow
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import model_zoo
import torchvision
import numpy as np
from os.path import join as opj
from graphviz import Digraph
import torch
from torch.autograd import Variable
from collections import OrderedDict
import ipdb
import os 
from PIL import Image
import copy 
import random
from scipy import stats
import itertools as it
from sklearn import manifold
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib as mpl
import ipympl
import cv2
import pickle
import seaborn as sns
#from rpy2.robjects import r, pandas2ri
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects
from collections import defaultdict
import pandas as pd
#base = importr('base')
#rstats = importr('stats')
#pandas2ri.activate()
from sklearn import svm
from joblib import Parallel, delayed
import multiprocessing
import sys
from importlib import reload
import utilities as ut
import colormath as cm
from colormath.color_objects import sRGBColor,LCHuvColor
#from colormath.color_conversions import convert_color
import time
import statsmodels.api as sm
from scipy.stats import vonmises
from scipy.special import i0 as bessel
from scipy.optimize import curve_fit
from scipy import math
from scipy import stats
from scipy.special import i0 as bessel
import cornet


# The transform for preprocessing the images; put the images through this before running through any networks:

transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def my_corr(x,y):
    if np.max(np.abs(x-y))==0:
        r=0
    elif (np.sum(np.abs(x))==0) or (np.sum(np.abs(y))==0):
        r=np.nan
    else:
        r = 1-np.corrcoef(x,y)[0,1]
    return r

dissim_defs = {'corr':my_corr,
               'euclidean_distance':lambda x,y:np.linalg.norm(x-y)}

def almost_equals(a,b,thresh=5):
    return all(abs(a[i]-b[i])<thresh for i in range(len(a)))

def create_or_append_csv(df,path):
    '''
    If the path already exists, append the dataframe to it, else make a new one.
    '''
    
    if not os.path.exists(path):
        df.to_csv(path)
    else:
        df.to_csv(path,mode='a',header=False)

def get_array_ranks(array):
    '''
    Put in array, spits out the same array with the entries replaced by their ranks.
    '''
    
    return array.ravel().argsort().argsort().reshape(array.shape)

def get_extreme_dissims(input_rdm,num_sim_pairs,num_dissim_pairs):
    '''
    Put in an rdm, and it spits out a subset of that rdm containing the items that are involved in the most similar
    and the most dissimilar pairs, where the user specifies how many of each you want. 
    '''
    
    dissim_ranks = get_array_ranks(input_rdm)
    inds_to_extract = []
    for pair in range(num_sim_pairs):
        rank = pair*2+input_rdm.shape[0] # do this in order to exlude the diagonal, and to only single-count pairs
        pair_coords = list(np.argwhere(dissim_ranks==rank)[0])
        inds_to_extract.append(pair_coords[0])
        inds_to_extract.append(pair_coords[1])
    for pair in range(num_dissim_pairs):
        rank = input_rdm.size-1-2*pair
        pair_coords = list(np.argwhere(dissim_ranks==rank)[0])
        inds_to_extract.append(pair_coords[0])
        inds_to_extract.append(pair_coords[1])
    inds_to_extract = list(np.unique(np.array(inds_to_extract)))
    filtered_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    return filtered_rdm,inds_to_extract

def get_extreme_dissims_no_replace(input_rdm,num_dissim_pairs):
    
    '''
    Put in an rdm, and it spits out a subset of that rdm containing the items that are involved in the most dissimilar 
    pairs. This version of the function does so without replacement, so each pair of items is removed after each draw.
    '''
    
    current_rdm = copy.copy(input_rdm)
    inds_to_extract = []
    remaining_inds = list(range(input_rdm.shape[0]))
    for pair in range(num_dissim_pairs):
        dissim_ranks = get_array_ranks(current_rdm)
        pair_coords = list(np.argwhere(dissim_ranks==(current_rdm.size-1))[0])
        real_x,real_y = remaining_inds[pair_coords[0]],remaining_inds[pair_coords[1]]
        inds_to_extract.append(real_x)
        inds_to_extract.append(real_y)
        remaining_inds = [ind for ind in remaining_inds if ((ind!=real_x) and (ind!=real_y))]
        current_rdm = input_rdm[np.ix_(remaining_inds,remaining_inds)]
    inds_to_extract = list(np.array(inds_to_extract))
    filtered_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    return filtered_rdm,inds_to_extract

def get_extreme_dissims_total(input_rdm,num_items='all'):
    
    '''
    Put in an rdm, and it spits out a subset of that rdm containing the items that collectively have the 
    highest possible total dissimilarity. Note that this isn't the same as the previous function, because
    that just gets the highest pairwise ones, whereas this gets the total similarity among ALL the objects.
    It uses an iterative approach: start with the maximally dissimilar item pair. Then find the item
    that has the highest total dissimilarity with the two of these. And so on and so forth. 
    And what it spits out will be the filtered rdm, along with the indices of the rows and columns of 
    the original input RDM. 
    '''
    
    if num_items=='all':
        num_items = input_rdm.shape[0]
    
    current_rdm = copy.copy(input_rdm)
    inds_to_extract = []
    remaining_inds = list(range(input_rdm.shape[0]))
    init_pair = list(np.argwhere(current_rdm==np.nanmax(current_rdm))[0])
    inds_to_extract = copy.copy(init_pair)
    for item in range(2,num_items):
        sub_matrix = current_rdm[np.sort(inds_to_extract),:].sum(axis=0)
        max_ind = 'NA'
        max_ind_val = -np.infty
        for i in range(len(sub_matrix)):
            if i in inds_to_extract:
                continue
            if sub_matrix[i]>max_ind_val:
                max_ind = i
                max_ind_val = sub_matrix[i]
        if (max_ind_val == -np.infty) or (np.isnan(max_ind_val)):
            break
        inds_to_extract.append(max_ind)
    output_rdm = current_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    return output_rdm,inds_to_extract        

def get_dissims_that_match(input_rdm,target_val,num_items_final='all'):
    
    '''
    Put in an rdm, and it spits out a subset of that rdm containing the items that collectively have the 
    dissimilarity as close as possible to the target dissimilarity for the upper triangular elements.  
    '''
    
    if num_items_final=='all':
        num_items_final = input_rdm.shape[0]
    
    current_rdm = copy.copy(input_rdm)
    inds_to_extract = []
    num_items_init = input_rdm.shape[0]
    
    remaining_inds = list(range(num_items_init))
    init_pair = list(np.argwhere(abs(current_rdm-target_val)==np.nanmin(abs(current_rdm-target_val)))[0])
    inds_to_extract = copy.copy(init_pair)
    current_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    for item_count in range(2,num_items_final):
        min_ind = 'NA'
        min_ind_val = np.infty 
        # Test every item in the array, look at how close the mean similarity is to the desired value. 
        for test_item in range(num_items_init):
            if test_item in inds_to_extract:
                continue
            test_matrix_inds = inds_to_extract + [test_item]
            test_matrix = input_rdm[np.ix_(test_matrix_inds,test_matrix_inds)]
            test_matrix_vals = test_matrix[np.triu_indices(test_matrix.shape[0],k=1)].flatten()
            total_diff = abs(np.mean(test_matrix_vals)-target_val)
            if total_diff<min_ind_val:
                min_ind_val = total_diff
                min_ind = test_item
        inds_to_extract.append(min_ind)
        current_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    mean_dissim = np.mean(current_rdm[np.triu_indices(current_rdm.shape[0],k=1)])
    return current_rdm,inds_to_extract,mean_dissim       

def get_most_uniform_dissims(input_rdm,num_items_final):
    '''
    Put in an RDM, and it finds the set of indices that collectively produce the most UNIFORM distribution of dissimilarities. 
    This is meant to get a set of items with a range of dissimilarities. It initializes by getting the most dissimilar item,
    then successively adding items that maximize the uniformity of the distribution (most uniform between 0 and 1).
    In case of ties, just chooses the first one. And so on till you have a nice matrix. 
    '''
    
    if num_items_final=='all':
        num_items_final = input_rdm.shape[0]
    
    current_rdm = copy.copy(input_rdm)
    inds_to_extract = []
    num_items_init = input_rdm.shape[0]
    
    remaining_inds = list(range(num_items_init))
    init_pair = list(np.argwhere(current_rdm==current_rdm.max())[0])
    inds_to_extract = copy.copy(init_pair)
    current_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
    for item_count in range(2,num_items_final):
        max_ind = 'NA'
        max_ind_val = -1 # initialize p-value, you want the GREATEST possible p-value (most uniform)
        # Test every item in the array, look at the uniformity of the resulting distribution if you add it in. 
        for test_item in range(num_items_init):
            if test_item in inds_to_extract:
                continue
            test_matrix_inds = inds_to_extract + [test_item]
            test_matrix = input_rdm[np.ix_(test_matrix_inds,test_matrix_inds)]
            test_matrix_vals = test_matrix[np.triu_indices(test_matrix.shape[0],k=1)].flatten()
            trash,p = stats.kstest(test_matrix_vals,stats.uniform(loc=0, scale=1).cdf)
            if p>max_ind_val:
                max_ind_val = p
                max_ind = test_item
        inds_to_extract.append(max_ind)
        current_rdm = input_rdm[np.ix_(inds_to_extract,inds_to_extract)]
            
    return current_rdm,inds_to_extract      

def fetch_layers_internal(current_layer,layer_pointers,layer_indices,layer_counter):
    '''
    Gets the names and "addresses" of all the layers of the network. 
    '''
    layer_type = type(current_layer)
    if layer_type==torch.nn.modules.container.Sequential:
        for i,sub_layer in enumerate(current_layer):
            layer_pointers,layer_counter = fetch_layers_internal(current_layer[i],layer_pointers,layer_indices+[i],layer_counter)
        return(layer_pointers,layer_counter)
    if layer_type==torchvision.models.resnet.Bottleneck:
        sub_layer_list = current_layer.named_children()
        sub_layer_list = [sl[0] for sl in sub_layer_list]
        for sub_layer in sub_layer_list:
            layer_pointers,layer_counter = fetch_layers_internal(getattr(current_layer,sub_layer),layer_pointers,layer_indices+[('attr',sub_layer)],layer_counter)
        return(layer_pointers,layer_counter)
    elif layer_type.__name__=='BasicConv2d':
        sub_layer_list = current_layer.named_children()
        sub_layer_list = [sl[0] for sl in sub_layer_list]
        for sub_layer in sub_layer_list:
            layer_pointers,layer_counter = fetch_layers_internal(getattr(current_layer,sub_layer),layer_pointers,layer_indices+[('attr',sub_layer)],layer_counter)
        return(layer_pointers,layer_counter)
    elif 'Inception' in str(layer_type):
        sub_layer_list = current_layer.named_children()
        sub_layer_list = [sl[0] for sl in sub_layer_list]
        for sub_layer in sub_layer_list:
            layer_pointers,layer_counter = fetch_layers_internal(getattr(current_layer,sub_layer),layer_pointers,layer_indices+[('attr',sub_layer)],layer_counter)
        return(layer_pointers,layer_counter)
    elif 'CORblock_Z' in str(layer_type):
        sub_layer_list = current_layer.named_children()
        sub_layer_list = [sl[0] for sl in sub_layer_list]
        for sub_layer in sub_layer_list:
            layer_pointers,layer_counter = fetch_layers_internal(getattr(current_layer,sub_layer),layer_pointers,layer_indices+[('attr',sub_layer)],layer_counter)
        return(layer_pointers,layer_counter)
    elif 'CORblock_S' in str(layer_type):
        sub_layer_list = current_layer.named_children()
        sub_layer_list = [sl[0] for sl in sub_layer_list]
        for sub_layer in sub_layer_list:
            layer_pointers,layer_counter = fetch_layers_internal(getattr(current_layer,sub_layer),layer_pointers,layer_indices+[('attr',sub_layer)],layer_counter)
        return(layer_pointers,layer_counter)
    elif 'Conv2d' in str(layer_type):
        num_convs = len([layer for layer in layer_pointers if 'conv' in layer[0]])
        layer_pointers[('conv'+str(num_convs+1),layer_counter)] = layer_indices  
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'Linear' in str(layer_type):
        num_linear = len([layer for layer in layer_pointers if 'fc' in layer[0]])
        layer_pointers[('fc'+str(num_linear+1),layer_counter)] = layer_indices       
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'MaxPool2d' in str(layer_type):
        num_maxpools = len([layer for layer in layer_pointers if 'maxpool' in layer[0]])
        layer_pointers[('maxpool'+str(num_maxpools+1),layer_counter)] = layer_indices  
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'AvgPool2d' in str(layer_type):
        num_avgpools = len([layer for layer in layer_pointers if 'avgpool' in layer[0]])
        layer_pointers[('avgpool'+str(num_avgpools+1),layer_counter)] = layer_indices  
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'ReLU' in str(layer_type):
        num_relu = len([layer for layer in layer_pointers if 'relu' in layer[0]])
        layer_pointers[('relu'+str(num_relu+1),layer_counter)] = layer_indices
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'BatchNorm2d' in str(layer_type):
        num_batchnorm = len([layer for layer in layer_pointers if 'batchnorm' in layer[0]])
        layer_pointers[('batchnorm'+str(num_batchnorm+1),layer_counter)] = layer_indices
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'Dropout' in str(layer_type):
        num_dropout = len([layer for layer in layer_pointers if 'dropout' in layer[0]])
        layer_pointers[('dropout'+str(num_dropout+1),layer_counter)] = layer_indices
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'Flatten' in str(layer_type):
        num_flatten = len([layer for layer in layer_pointers if 'flatten' in layer[0]])
        layer_pointers[('flatten'+str(num_flatten+1),layer_counter)] = layer_indices
        layer_counter += 1
        return(layer_pointers,layer_counter)
    elif 'Identity' in str(layer_type):
        num_identity = len([layer for layer in layer_pointers if 'identity' in layer[0]])
        layer_pointers[('identity'+str(num_identity+1),layer_counter)] = layer_indices
        layer_counter += 1
        return(layer_pointers,layer_counter)
    else:
        return(layer_pointers,layer_counter)

def fetch_layers(model):
    '''
    Wrapper convenience function.
    '''
    
    layer_pointers = OrderedDict()
    layer_counter = 1
    for macro_layer in model._modules:
        layer_pointers,layer_counter = fetch_layers_internal(model._modules[macro_layer],layer_pointers,[macro_layer],layer_counter)
    return layer_pointers

def prepare_models(models_dict):
    models_prepped = OrderedDict()
    for model_name in models_dict:
        base_model,weights_url = models_dict[model_name]
        models_prepped[model_name] = prepare_model(base_model,weights_url)
    return models_prepped

def prepare_model(which_model,weights_url='na'):
    '''
    Put in the name of the model and this will prepare it to run through the images and get the activations. 
    
    6/17/19: added an option to specify a model subtype, for different pre-trained models (e.g., Resnet trained for shape).
    Put in a tuple of (subtype_name,weights_url) if you wish to do a different pre-training of the model. 
    '''
    
    if which_model=='googlenet':
        model = torch.hub.load('pytorch/vision', 'googlenet', pretrained=True)
    elif which_model=='cornet_z':
        model = cornet.cornet_z(pretrained=True,map_location='cpu')
    elif which_model=='cornet_s':
        model = cornet.cornet_s(pretrained=True,map_location='cpu')
    else:  
        model = getattr(models,which_model)(pretrained=True)
    model.eval()
    
    if weights_url != 'na':
        checkpoint = model_zoo.load_url(weights_url,map_location='cpu')
        if 'module' in list(checkpoint['state_dict'].keys())[0]:
            new_checkpoint_state_dict = OrderedDict()
            for key in checkpoint['state_dict']:
                new_key = key[7:]
                new_checkpoint_state_dict[new_key] = checkpoint['state_dict'][key]
        else: 
            new_checkpoint_state_dict = checkpoint['state_dict']
        model.load_state_dict(new_checkpoint_state_dict,strict=True)
    
    # Get the pointers to the layers of interest: 
    
    layer_pointers = fetch_layers(model)
        
    # Now, register forward hooks to all these layers:

    for layer_name in layer_pointers:   
        layer_index = layer_pointers[layer_name]
        layer = index_nested(model._modules,layer_index)
        layer.register_forward_hook(get_layer_activation)
        layer.layer_name = layer_name    
    
    return model

def get_layer_names(which_model):
    '''
    Convenience function to spit out the list of layer names. 
    '''
    
    model = getattr(models,which_model)(pretrained=True)
    model.eval()
    
    layer_pointers = fetch_layers(model)
        
    # Now, register forward hooks to all these layers:

    for layer_name in layer_pointers:   
        layer_index = layer_pointers[layer_name]
        layer = index_nested(model._modules,layer_index)
        layer.register_forward_hook(get_layer_activation)
        layer.layer_name = layer_name    
    
    return list(layer_pointers.keys())

def alphatize_image(im,alpha_target,alpha_range):
    '''
    Put in an image, the RGB coordinates to set as alpha, and the range around those coordinates to
    set as transparent.
    '''
    
    image_alpha = im.convert('RGBA')
    pixel_data = list(image_alpha.getdata())

    for i,pixel in enumerate(pixel_data):
        if almost_equals(pixel[:3],alpha_target,alpha_range):
            pixel_data[i] = (255,255,255,0)

    image_alpha.putdata(pixel_data)
    return image_alpha

def get_model_activations_for_object(obj_fname,model):
    '''
    Convenience function that takes in the filename for a stimulus and a model, and spits out the model activations 
    as an ordered dictionary. 
    '''
    
    image = Image.open(obj_fname).convert('RGB')    
    global layer_activations 
    layer_activations = OrderedDict()
    preprocessed_image = preprocess_image(image)
    model.forward(preprocessed_image)
    return layer_activations

def get_image_set_rdm(image_set,model_name,models_dict,rdm_name='rdm_default_name',which_layers={},dissim_metrics=['corr'],kernel_sets=('all','all'),num_perms=0,verbose=False,debug_check=False):
    '''
    Internal helper function for the get_rdms function in order to enable parallel processing. 
    '''
     
    image_set_labels,images,entry_types,combination_list = image_set
    print(f"{','.join(list(image_set_labels.values()))}")
    image_names = list(images.keys())
    combination_names = [comb[0] for comb in combination_list]
    entry_names = image_names + [comb[0] for comb in combination_list] # labels of the rows and columns

    # If you want to do different color spaces, fill this in someday. Fill in funcs that take in RGB and spit out desired color space. 
    color_space_funcs = {}
    color_space_list = []
          
    rdms_dict = defaultdict(lambda:[])
    perm_list = ['orig_data'] + list(range(1,num_perms+1))
    perm = perm_list[0] # Placeholder until the loop is added 

    model = models_dict[model_name]
    print(f"\tModel {model_name}")

    # Get layer activations for images. 

    print("\t\tComputing image activations...")

    obj_activations = OrderedDict()
    for image_name in images:
        if verbose:
            print(image_name)
        image = images[image_name]
        if type(image)==str: # in case it's a file path. 
            image = Image.open(image)
        preprocessed_image = preprocess_image(image)            
        obj_activations[image_name] = OrderedDict()

        obj_activations[image_name][('original_rgb',0),'feature_means'] = np.array(preprocessed_image).squeeze()
        obj_activations[image_name][('original_rgb',0),'unit_level'] = np.array(preprocessed_image).squeeze().mean(axis=(1,2))

        # If you want to include different color spaces. 

        for cs in color_space_list:
            converted_image = copy.copy(preprocessed_image)
            for i,j in it.product(range(converted_image.shape[0]),range(converted_image.shape[1])):
                converted_image[i,j,:] = color_space_funcs[cs](converted_image[i,j,:])
            obj_activations[image_name][f'original_{cs}','with_space'] = preprocessed_image.squeeze()
            obj_activations[image_name][f'original_{cs}','no_space'] = preprocessed_image.squeeze().mean(axis=(0,1))    

        global layer_activations 
        layer_activations = OrderedDict()
        model.forward(preprocessed_image)

        for layer in layer_activations:
            layer_dim = len(layer_activations[layer].squeeze().shape)
            if layer_dim == 1:
                obj_activations[image_name][layer,'unit_level'] = layer_activations[layer].squeeze()
                obj_activations[image_name][layer,'feature_means'] = layer_activations[layer].squeeze()
            else: 
                obj_activations[image_name][layer,'unit_level'] = layer_activations[layer].squeeze()
                obj_activations[image_name][layer,'feature_means'] = layer_activations[layer].squeeze().mean(axis=(1,2))
    # If you want to combine the patterns in any way. 

    for comb_name,comb_stim,comb_func,comb_func_args,unpack_args in combination_list:
        obj_activations[comb_name] = OrderedDict()
        # For different layers and activation types.
        keys = list(obj_activations[comb_stim[0]].keys())

        for key in keys:
            stim_activations = [obj_activations[stim][key] for stim in comb_stim]
            if unpack_args:
                obj_activations[comb_name][key] = comb_func(*stim_activations,**comb_func_args)
            else:
                obj_activations[comb_name][key] = comb_func(stim_activations,**comb_func_args)

    # Now make the RDM.

    print("Making RDMs...")
          
    layer_list = [('original_rgb',0)] + list(layer_activations.keys())

    for layer,dm in it.product(layer_list,dissim_metrics):
          
        layer_name = layer[0]
                    
        # Check if the layer is one of the specified layers. If it's a model with only specified layers, check 
        # if the current layer is in it or not. 

        if model_name in which_layers:
            model_layer_list = which_layers[model_name]
            if (layer not in model_layer_list) and (layer_name not in model_layer_list) and (layer_name != 'original_rgb'):
                continue
        print(f"\n\t{layer}")
          
        activation_types = ['unit_level','feature_means']

        dissim_func = dissim_defs[dm]

        for activation_type,ks in it.product(activation_types,kernel_sets):
          
            print(f"\t\tKernel set {ks[0]}, activation {activation_type}")
                    
            if (layer,activation_type) not in obj_activations[list(obj_activations.keys())[0]]:
                continue
          
            sample_image = obj_activations[list(obj_activations.keys())[0]][layer,activation_type]
          
            ks_name,ks_dict = ks
            if ks_dict == 'all':
                kernel_inds = list(range(sample_image.shape[0]))
            elif (model_name,layer_name) in ks_dict:
                kernel_inds = ks_dict[(model_name,layer_name)]
            else:
                print(f"*****Kernels not specified for {model_name},{layer}; using all kernels.*****")
                kernel_inds = list(range(sample_image.shape[0])) 
          
            rdm = np.empty((len(entry_names),len(entry_names))) * np.nan
            num_pairs = len(entry_names)*len(entry_names)/2
            pair_num = 0
            next_percent = 0
            for (i1,im1) in enumerate(entry_names):
                for (i2,im2) in enumerate(entry_names):
                    if i2<i1:
                        continue
                    if verbose:
                        pair_num += 1
                        if (pair_num/num_pairs)*100 > next_percent:
                            print(f"{next_percent}%",end=' ',flush=True)
                            next_percent=next_percent+1    
          
                    pattern1 = obj_activations[im1][layer,activation_type]
                    pattern2 = obj_activations[im2][layer,activation_type]
                    pattern1_final = pattern1[kernel_inds,...].flatten().astype(float)
                    pattern2_final = pattern2[kernel_inds,...].flatten().astype(float)     
                    dissim = dissim_func(pattern1_final,pattern2_final)
                    rdm[i1,i2] = dissim
                    rdm[i2,i1] = dissim
                    
            trash,dissim_rankings = get_extreme_dissims_total(rdm,'all')

            rdms_dict['df_name'].append(rdm_name)
            rdms_dict['model'].append(model_name)
            rdms_dict['layer'].append(layer_name)
            rdms_dict['layer_num'].append(layer[1])
            rdms_dict['layer_label'].append(layer)
            rdms_dict['dissim_metric'].append(dm)
            rdms_dict['activation_type'].append(activation_type)
            rdms_dict['matrix'].append(rdm)
            rdms_dict['entry_keys'].append(tuple(entry_names))
            rdms_dict['entry_labels'].append(tuple([{entry_types[i]:entry[i] for i in range(len(entry_types))} for entry in entry_names]))
            rdms_dict['kernel_set_name'].append(ks_name) 
            rdms_dict['kernel_inds'].append(tuple(kernel_inds))
            rdms_dict['perm'].append(perm)
            rdms_dict['dissim_rankings'].append(tuple(dissim_rankings))
            for label in image_set_labels:
                rdms_dict[label].append(image_set_labels[label])
            if debug_check:
                print(pd.DataFrame.from_dict(rdms_dict))
                ipdb.set_trace()
                debug_check=False # Only check the first one, then continue. 
    del(obj_activations)
    rdms_df = pd.DataFrame.from_dict(rdms_dict)
    return rdms_df    
          
def get_rdms(image_sets,
             models_dict,
             out_fname,
             rdm_name='default_name',
             which_layers={},
             dissim_metrics=['corr'],
             kernel_sets=[('all','all')],
             num_perms = 0,
             num_cores = 'na',
             verbose=False,
             debug_check=False):
    
    '''
    This function takes in a list of models, layer types, images, and dissim_metrics, and images,
    and calculates the RDMs for those images across all the other parameters. Spits out an
    ordered dictionary with all the information. If well-written, this will vastly speed up
    everything else. The images must in a dict or ordered-dict format, where each key
    is the image name, and each value is the image as a PIL Image variable. 
    
    4/29/19 change: you can put in a list of "combination patterns" (if you want to get the superposition of
    several patterns, e.g. for the tessellation/spiral stimuli). Each element in the list is a tuple,
    constisting of (combination_name,[list of names of patterns to combine],agg_func). This will become one 
    of the patterns that is then put through the RDM. 
    
    5/28/19 change: can specify which kernels you want to use. This will be a dictionary with (model,layer) as key,
    and a list of indices of the desired kernels as value. Of course, leave it blank if you want all the kernels. 
    
    5/31/19 change: now returns a pandas array of RDMs, to make it easier to filter and search by different keys
    while remaining backwards compatible as changes accumulate. 
    
    Also added an option to permute the data, put in a non-zero number if you want to scramble the rows and columns. 
    Not implemented yet, not sure what it even means, but there's a pandas field for it just in case. 
    
    Also, the images can EITHER be PIL images, or filenames for the images (latter is better to save RAM)
    
    6/13 change: can run the image sets on multiple cores to speed things up. Leave as 'na' to not use
    parallel processing. 
    
    6/17 change: can specify which layers of each model to extract. Empty dictionary means extract all of them. 
    '''
    
    print("*****Computing RDMs*****")          
          
    if num_cores=='na':
        rdm_df_list = []
        for image_set,model_name in it.product(image_sets,models_dict):   
            new_df = get_image_set_rdm(image_set,model_name,models_dict,rdm_name,which_layers,
                                       dissim_metrics,kernel_sets,num_perms,verbose,debug_check)
            rdm_df_list.append(new_df)
    else:
        #rdm_df_list = Parallel(n_jobs=num_cores)(delayed(get_image_set_rdm)(image_set,model_name,models,
                                                                            #rdm_name,which_layer_types,
                                       #dissim_metrics,kernel_sets,num_perms,debug_check=False) for image_set,model_name in it.product(image_sets,models))
        pool = multiprocessing.Pool(processes=num_cores)
        rdm_df_list = [pool.apply_async(get_image_set_rdm,args=(image_set,model_name,models_dict,rdm_name,which_layers,
                                       dissim_metrics,kernel_sets,num_perms,verbose,debug_check)) for image_set,model_name in it.product(image_sets,models_dict)]
        rdm_df_list = [p.get() for p in rdm_df_list]        
          
    rdms_df = pd.concat(rdm_df_list)
    rdms_df = rdms_df.sort_values(by=list(image_sets[0][0].keys())+['model'])
    pickle.dump(rdms_df,open(out_fname,"wb"))
    return rdms_df

def get_meta_rdms(rdm_df,
                  out_fname,
                  entry_variables,
                  entry_var_subsets = [('all','all')],
                  grouping_variables='rest',
                  df_name='meta_rdm',
                  num_perms=0,
                  dissim_metrics=['corr']):
    '''
    Takes in a df of rdms, and computes meta-rdms over them. Need to specify what the entry variables will be
    (that is, the cells of the meta-rdms), and what the grouping variables will be. Do any necessary
    subsetting of the rdm_df outside the function, hard-coding it would be a bitch. 
    
    The function contains some sanity checks to make sure nothing goes awry (e.g., that there's no repeats
    of an entry).
    '''
    rdm_df = copy.deepcopy(rdm_df.copy(deep=True))
    rdm_df.columns = list(copy.deepcopy(rdm_df.columns.values))
              
    perm_list = ['orig_data'] + list(range(1,num_perms+1))
              
    meta_rdm_dict = defaultdict(lambda:[])
    if grouping_variables=='rest':
        not_group_vars = entry_variables + ['matrix','entry_keys','entry_labels']
        grouping_variables = [var for var in rdm_df.columns.values if var not in not_group_vars] 
        grouping_variables = [var for var in grouping_variables if 'entry_labels' not in var]
    
    # Tweak the grouping variables to ensure no overlap with the meta-variable names. 
    # (Since both the meta-rdm and the sub-rdm have their own dissim metrics, permutations, etc.)
              
    for i,var in enumerate(grouping_variables):
        if ('perm' in var) or ('df_name' in var) or ('dissim_metric' in var) or ('entry_keys_sub' in var) or ('entry_labels_sub' in var) or ('entry_var_subset' in var):
              grouping_variables[i] = grouping_variables[i]+'_sub'
    for i,var in enumerate(rdm_df.columns.values):
        if ('perm' in var) or ('df_name' in var) or ('dissim_metric' in var) or ('entry_keys_sub' in var) or ('entry_labels_sub' in var) or ('entry_var_subset' in var):
              new_columns = rdm_df.columns.values
              new_columns[i] = new_columns[i]+'_sub'
              rdm_df.columns = new_columns
              
    rdm_groups = rdm_df.groupby(grouping_variables,as_index=False)
              
    for group_labels,rdm_group in rdm_groups:
              
        print(f"Getting meta-rdm for {','.join([g for g in group_labels if type(g)==str])}")
                      
        # Check for no repeat entries.
              
        rdm_group.loc[:,'entry_key'] = rdm_group.apply(lambda row:tuple([row[col] for col in entry_variables]),axis=1)
        rdm_group.loc[:,'entry_label'] = rdm_group.apply(lambda row:OrderedDict({col:row[col] for col in entry_variables}),axis=1)
        
        if len(rdm_group['entry_key'].unique()) != len(rdm_group):
            raise Exception("There seem to be repeat entries going into the RDM--"
                            "did you specify the right grouping and entry variables?")
              
        for entry_var_subset_name,entry_var_subset in entry_var_subsets:
            if entry_var_subset_name != 'all':
                rdm_group_subset = rdm_group.loc[rdm_group[entry_key].isin(entry_var_subset),:]
            else:
                rdm_group_subset = rdm_group.copy(deep=True)
              
            num_entries = len(rdm_group_subset)
            num_sub_entries = rdm_group_subset['matrix'].iloc[0].shape[0]
        
            for dm,perm in it.product(dissim_metrics,perm_list):
                dissim_func = dissim_defs[dm]
                inds_dict = {}
                for entry_key in rdm_group_subset['entry_key'].unique():
                    inds = list(range(num_sub_entries))
                    if perm != 'orig_data':
                        random.shuffle(inds)
                    inds_dict[entry_key] = inds
                group_rdm = np.empty((num_entries,num_entries)) * np.nan
                for i,j in it.product(range(num_entries),range(num_entries)):
                    if rdm_group_subset['entry_keys'].iloc[i] != rdm_group_subset['entry_keys'].iloc[j]:
                        raise Exception("The sub-RDMs have different entries and don't correspond. Double check that they match.")
              
                    inds1 = inds_dict[rdm_group_subset['entry_key'].iloc[i]]
                    inds2 = inds_dict[rdm_group_subset['entry_key'].iloc[j]]
              
                    rdm1 = rdm_group_subset['matrix'].iloc[i][np.ix_(inds1,inds1)]
                    rdm2 = rdm_group_subset['matrix'].iloc[j][np.ix_(inds2,inds2)]
              
                    # Remove any rows of all nans. 
              
                    which_entries = []
                    for r in range(rdm1.shape[0]):
                        if (np.sum(np.isnan(rdm1[r,:]))<rdm1.shape[0]) and ((np.sum(np.isnan(rdm2[r,:]))<rdm1.shape[0])):
                            which_entries.append(r)
                    num_rows = len(which_entries)
              
                    rdm1 = rdm1[np.ix_(which_entries,which_entries)]
                    rdm2 = rdm2[np.ix_(which_entries,which_entries)]
                    
              
                    rdm1_vals = rdm1[np.triu_indices(num_rows,k=1)].flatten()
                    rdm2_vals = rdm2[np.triu_indices(num_rows,k=1)].flatten()
                    new_val = dissim_func(rdm1_vals,rdm2_vals)
                    group_rdm[i,j] = new_val
                    
                
                trash,dissim_rankings = get_extreme_dissims_total(group_rdm,'all')
              
                meta_rdm_dict['entry_keys'].append(tuple(rdm_group_subset['entry_key']))    
                meta_rdm_dict['entry_labels'].append(tuple(rdm_group_subset['entry_label']))
                meta_rdm_dict['matrix'].append(group_rdm)
                meta_rdm_dict['dissim_metric'].append(dm)
                meta_rdm_dict['df_name'].append(df_name)
                meta_rdm_dict['perm'].append(perm)
                meta_rdm_dict['entry_var_subset'].append(entry_var_subset_name)
                meta_rdm_dict['dissim_rankings'].append(tuple(dissim_rankings))
                for col in rdm_group_subset.columns.values:
                    if ('_sub_sub') in col:
                        meta_rdm_dict[col].append(rdm_group_subset[col].iloc[0])
                meta_rdm_dict['entry_keys_sub'].append(rdm_group_subset['entry_keys'].iloc[0])
                meta_rdm_dict['entry_labels_sub'].append(rdm_group_subset['entry_labels'].iloc[0])
                for i,label in enumerate(group_labels):
                    meta_rdm_dict[grouping_variables[i]].append(label)
            
    meta_rdm_df = pd.DataFrame.from_dict(meta_rdm_dict)
    meta_rdm_df.sort_index(axis=1, inplace=True)
    pickle.dump(meta_rdm_df,open(out_fname,"wb"))
    return meta_rdm_df   
              
def subset_rdm_df(df,entry_keys):
    '''
    Extract subset of the input RDM, using only the given entry keys.
    '''
    new_matrices_list = []
    new_keys_list = []
    new_labels_list = []
              
    for r in range(len(df)):
        row = df.iloc[r,:]
        row_keys = row['entry_keys']
        row_labels = row['entry_labels']
        matrix = row['matrix']
        which_inds = []
        new_keys = []
        new_labels = []
        for k,key in enumerate(row_keys):
            if key in entry_keys:
                which_inds.append(k)
                new_keys.append(key)
                new_labels.append(row_labels[k])
        matrix_subset = matrix[np.ix_(which_inds,which_inds)]
        new_matrices_list.append(matrix_subset)
        new_keys_list.append(new_keys)
        new_labels_list.append(new_labels)
    df['matrix'] = new_matrices_list
    df['entry_keys'] = new_keys_list
    df['entry_labels'] = new_labels_list
    return df
              

def compute_tolerances(input_df):
    '''
    This function takes in an RDM dataframe, and goes through and calculates the tolerance for every 
    RDM in the dataframe. To work, the entries of the dataframe must have two keys (so two variables).
    It will calculate the tolerance of each variable with respect to the other, and then add this
    as new columns to the dataframe. Tolerance is defined with this formula from Yaoda:
    
    The averaged distance between the same object category across the two formats (d.s-object/d-format) and the averaged distance between different object categories within the same format (d.d-object/s-format). We then constructed a tolerance index (TI) as:

    TI = ( d.d-object/s-format - d.s-object/d-format) / (d.d-object/s-format + d.s-object/d-format)
    
    A completely transformation-invariant representation would have a d.s-object/d-format of “0” and a TI of “1”, whereas a representation of transformation with no object categories would have a d.d-object/s-format of “0” and a TI of “-1”. A TI of “0” means that an object category is equally similar to itself in the other format as it is to the other categories in the same format.”
    '''
              
    tolerance_vals = defaultdict(lambda:[])
    for i,row in input_df.iterrows():
        matrix = row['matrix']
        labels = row['entry_labels']
        keys = row['entry_keys']
        # Do it in both possible directions. 
        variable_list = list(labels[0].keys()) 
        for v in range(len(variable_list)):
            main_var = variable_list[v]
            comp_var = [variable_list[var] for var in range(len(variable_list)) if var!=v][0]
            main_var_vals = []
            comp_var_vals = []
            for label in labels:
                if label[main_var] not in main_var_vals:
                    main_var_vals.append(label[main_var])
                if label[comp_var] not in comp_var_vals:
                    comp_var_vals.append(label[comp_var])
              
            # Now assemble the required values: mean distance between the main val held constant
            # across different values of the comp val, and vice versa.
            
            main_var_same_vals = []
            main_var_diff_vals = []
            
            for main_var_val1,comp_var_val1,main_var_val2,comp_var_val2 in it.product(main_var_vals,comp_var_vals,main_var_vals,comp_var_vals):
              
                # Ignore if both vals the same, or both vals different. 
                if ((main_var_val1==main_var_val2 and comp_var_val1==comp_var_val2) or
                   (main_var_val1!=main_var_val2 and comp_var_val1!=comp_var_val2)):
                    continue
              
                # Pull out the indices. 
                for ind in range(len(labels)):
                    label = labels[ind]
                    if label[main_var]==main_var_val1 and label[comp_var]==comp_var_val1:
                        ind1 = ind
                    if label[main_var]==main_var_val2 and label[comp_var]==comp_var_val2:
                        ind2 = ind
              
                if main_var_val1==main_var_val2:
                    main_var_same_vals.append(matrix[ind1,ind2])
                elif comp_var_val1==comp_var_val2:
                    main_var_diff_vals.append(matrix[ind1,ind2])
            main_var_diff_mean = np.mean(main_var_diff_vals)
            main_var_same_mean = np.mean(main_var_same_vals)
            tolerance_index = (main_var_diff_mean - main_var_same_mean)/(main_var_diff_mean + main_var_same_mean)
            tolerance_vals[comp_var].append(tolerance_index)
    for val in tolerance_vals:
        input_df[val+'_tolerance'] = tolerance_vals[val]
    return input_df
            
            
            
            
        
        
              
def my_mds(rdm_df,
           dims=2,
           plot_opts={}, 
           plot_limits = (-.1,1.1),
           other_dot_opts={},
           cond_labels={},
           label_offset=.02,
           label_opts={},
           plot_dots=True,
           cond_colors={},
           cond_pics={},
           pic_size=.1,
           alphatize_pics=True,
           alpha_target = (255,255,255), 
           alpha_range=35, 
           pic_offset = (0,0),
           line_trajectories=[],
           show_stress=True, 
           plot_style='classic',
           fig_title='MDSPlot',
           fig_size=7, 
           show=True, 
           save_opts='na'):
    '''
    Takes an RDM as input, and plots the resulting MDS plot as a scatterplot. 
    
    rdm: nxn (symmetric) matrix containing the dissimilarities
    dims: usually either 2 or 3: how many dimensions to use
    cond_names: if you want there to be labels by the dots, include this (same order as RDM) as a list
    cond_pics: if you want to plot pictures on top of (instead of) the scatterplot dots, include a list
        here of either Image files or filenames. 
    plot_opts: if you wish to include any other keyword options into the matplotlib scatterplot options,
        do it here. E.g., if you wish to include colors 
    other_dot_opts: 
    show: True if you want to immediately show the plot, otherwise false.
    save_opts: na if you don't want to save the picture, else include the save options (e.g., path, file type)
    line_adj: if you wish to draw lines between the points on the MDS plot, include the adjacency matrix here,
        with zeros for non-edges and other integers for other kinds of edges.
    line_opts: also if you want to draw lines, then include a dictionary where each key is the kind of edge
        (integer as above), and each value is another dictionary containing the line options. 
    
    Notes: doing pictures on the 3D plots isn't implemented yet 
    '''
    plt.figure() 
              
    if len(rdm_df) != 1:
        raise Exception("Wrong number of rows in the input dataframe. Make sure it's just one.")          
              
    rdm = rdm_df['matrix'].iloc[0]
    entry_keys = rdm_df['entry_keys'].iloc[0]
              
    if len(cond_colors)>0:
        dot_colors = ['black']*len(entry_keys)
        for k in range(len(entry_keys)):
            if entry_keys[k] in cond_colors:
                dot_colors[k] = cond_colors[entry_keys[k]]
        plot_opts['c'] = dot_colors
    
    # If there's other ways you want to change the dots.           
              
    for opt in other_dot_opts:
        opt_dict = other_dot_opts[opt]
        opt_list = [opt_dict[key] for key in entry_keys]
        plot_opts[opt] = opt_list
              
    mpl.style.use(plot_style)
    
    # Return error if not implemented yet:
    
    if ((dims==3) and (len(cond_pics)>0)):
        raise Exception("Can't plot pictures on 3D plot yet :(")
        return 0
    
    # Get the MDS solution and rescale the dimensions to lie in interval [0,1] for all coordinates
        
    mds = manifold.MDS(n_components=dims, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)    
    fit = mds.fit(rdm)
    pos = fit.embedding_
    pos = (pos-np.min(pos))/(np.max(pos)-np.min(pos))
        
    if dims==2:
        plot = plt.figure(figsize=(fig_size,fig_size))
        ax = plot.add_subplot(111)
        ax.set_xlim(*plot_limits)
        ax.set_ylim(*plot_limits)
        
        for segment_list,line_opts in line_trajectories:   
            segment_list = [entry for entry in segment_list if entry in entry_keys]
            for s in range(1,len(segment_list)):
                origin_ind = entry_keys.index(segment_list[s-1])
                dest_ind = entry_keys.index(segment_list[s])
                ax.plot([pos[origin_ind,0],pos[dest_ind,0]],[pos[origin_ind,1],pos[dest_ind,1]], 
                        linestyle='-',zorder=1, **line_opts)
        
        if plot_dots:
            ax.scatter(x=pos[:,0],y=pos[:,1],**plot_opts,zorder=2)
              
        for entry in cond_pics:
            pic = cond_pics[entry]
            ind = entry_keys.index(entry)
            x,y = pos[ind,0]+pic_offset[0],pos[ind,1]+pic_offset[1]
            if type(pic)==str:
                pic = Image.open(pic)
            if alphatize_pics: 
                pic = alphatize_image(pic,alpha_target,alpha_range)
            bb = Bbox.from_bounds(x-pic_size/2,y-pic_size/2,pic_size,pic_size)  
            bb2 = TransformedBbox(bb,ax.transData)
            bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)
            bbox_image.set_data(pic)
            artist = ax.add_artist(bbox_image)
            artist.set_zorder(3)
        
        for entry in cond_labels:
            label = cond_labels[entry]
            ind = entry_keys.index(entry)
            ax.annotate(label,(pos[ind,0]-label_offset,pos[ind,1]+label_offset),**label_opts,zorder=4)
        
    elif dims==3: # This is not fully featured yet. 
        plot = plt.figure(figsize=(fig_size,fig_size))
        ax = Axes3D(plot)
        ax.set_xlim(*plot_limits)
        ax.set_ylim(*plot_limits)
        ax.set_zlim(*plot_limits)
        ax.scatter(xs=pos[:,0],ys=pos[:,1],zs=pos[:,2],**plot_opts)

        for entry in cond_labels:
            label = cond_labels[entry]
            ind = entry_keys.index(entry)
            ax.text(pos[ind,0]+label_offset,pos[ind,1]+label_offset,pos[ind,2]+label_offset,condname,**label_opts)
        
    if show_stress:
        fig_title += f"\nstress = {fit.stress_}"
    ax.set_title(fig_title)
    if save_opts != 'na':
        plot.savefig(**save_opts)
    print(f"Stress = {fit.stress_}")
    return(fit.stress_)

def get_layer_activation(module,input_,output_):
    layer_activations[module.layer_name] = output_.cpu().detach().numpy()  

def preprocess_image(input_image):
    '''
    Run an image through this before putting it into the network. 
    '''
    
    output_image = transform(input_image).unsqueeze(0)
    return output_image

def index_nested(the_item,indices):
    for ind in indices:
        if type(ind)!=tuple:
            the_item = the_item[ind]
        else:
            the_item = getattr(the_item,ind[1])
    return the_item

def combine_csvs(path,keys=[]):
    '''
    Utility function that takes in a list of filenames, and loads them all and combines them into a single pandas dataframe.
    Alternatively, can take in a filename for a directory, and it'll combine all the dataframes in that directory
    into one. 
    '''
    
    if type(path)==str:
        filepaths = os.listdir(path)
        filepaths = [opj(path,f) for f in filepaths if ('.csv' in f) or ('.p' in f)]
    else:
        filepaths = path
    
    if len(keys)>0:
        filepaths = [f for f in filepaths if any([key in f for key in keys])]
    
    df_list = []
    for f in filepaths:
        if '.csv' in f:
            df_list.append(pd.read_csv(f,index_col=0))
        elif '.p' in f:
            df_list.append(pickle.load(open(f,'rb')))
    out_df = pd.concat(df_list)
    return(out_df)

def get_svm_acc(bin1_patterns,bin2_patterns):
    '''
    Put in the list of patterns for each bin, it'll spit out the accuracy from leave-one-sample-out
    cross-validated SVM. 
    '''
    num_samples = len(bin1_patterns)
    total_acc = []
    for i in range(num_samples):
        bin1_test = np.array(bin1_patterns[i])
        bin2_test = np.array(bin2_patterns[i])
        bin1_train = np.array([bin1_patterns[x] for x in range(len(bin1_patterns)) if x!=i])
        bin2_train = np.array([bin2_patterns[x] for x in range(len(bin2_patterns)) if x!=i])
        train_data = np.vstack([bin1_train,bin2_train])
        test_data = np.vstack([bin1_test,bin2_test])
        train_labels = [0]*(num_samples-1)+[1]*(num_samples-1)
        test_labels = [0,1]
        clf = svm.SVC(kernel='linear',C=1)
        clf.fit(train_data,train_labels)
        label_prediction = clf.predict(test_data)
        chunk_acc = list((test_labels==label_prediction).astype(int))
        total_acc.extend(chunk_acc)
    return np.mean(total_acc)
        
def my_2way_anova(factor1,factor2,dv,df):
    '''
    Faster calculation of the eta-squared values of an ANOVA so it doesn't take forever and ever.
    Spits out eta-squared for factor1, factor2, interaction.
    
    Follows http://www.miracosta.edu/Home/rmorrissette/Chapter14.htm
    '''
        
    grand_mean = df[dv].mean()
    grand_sum = df[dv].sum()
    squared_vals = np.array(df[dv]*df[dv])
    f1_vals = df[factor1].unique()
    f2_vals = df[factor2].unique()
    n_tot = len(df)
    n_fac1 = len(df[factor1].unique())
    n_fac2 = len(df[factor2].unique())
    
    ss_tot = np.sum(squared_vals)-(grand_sum**2)/n_tot
    ss_fac1 = 0
    for f1 in f1_vals:
        df_subset = df[df[factor1]==f1]
        ss_fac1 += np.sum(df_subset[dv])**2
    ss_fac1 = ss_fac1/n_fac2 - (grand_sum**2)/n_tot
    
    ss_fac2 = 0
    for f2 in f2_vals:
        df_subset = df[df[factor2]==f2]
        ss_fac2 += np.sum(df_subset[dv])**2
    ss_fac2 = ss_fac2/n_fac1 - (grand_sum**2)/n_tot  
    
    ss_interact = ss_tot-ss_fac1-ss_fac2
    
    eta2_fac1 = ss_fac1/ss_tot
    eta2_fac2 = ss_fac2/ss_tot
    eta2_interact = ss_interact/ss_tot
    
    return eta2_fac1,eta2_fac2,eta2_interact
              
def curve_fit_with_metrics(f,xdata,ydata,param_labels,init_dict={},bound_dict={}):
    '''
    Like curve fit, but it also returns the fit metrics. 
    '''
    n = len(ydata)
    k = len(xdata[0])
    
    init_guesses = []
    lower_bounds = []
    upper_bounds = []
    for p,label in enumerate(param_labels):
        if label in init_dict:
            init_guesses.append(init_dict[label])
        else:
            init_guesses.append(1)
        if label in bound_dict:
            lower_bounds.append(bound_dict[label][0])
            upper_bounds.append(bound_dict[label][1])
        else:
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
    
    ydata = np.array(ydata)
    popt,pcov = curve_fit(f,xdata,ydata,p0=init_guesses,bounds=(lower_bounds,upper_bounds))
    predicted_values = f(xdata,*popt)
    ss_res = np.sum((ydata-predicted_values)**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1-ss_res/ss_tot
    
    aic = -2*np.log(ss_res/n) + 2*k + (2*k*(k+1))/(n-k-1)
    
    out_dict = OrderedDict({})
    for p in range(len(param_labels)):
        out_dict[param_labels[p]] = popt[p]
    out_dict['r_squared'] = r_squared
    out_dict['aic'] = aic
    return(out_dict)

def get_cov_matrix_2d(m11,m12,m22):
    '''
    Where the matrix is [[m11,m12],[m12,m22]]
    '''
    cv11 = m11**2+m12**2
    cv12 = m11*m12+m22*m12
    cv22 = m22**2+m12**2
    return cv11,cv12,cv22

def get_cov_matrix_3d(m11,m12,m13,m22,m23,m33):
    cv11 = m11**2+m12**2+m13**2
    cv12 = m11*m12 + m12*m22 + m13*m23
    cv13 = m11*m13 + m12*m23 + m13*m33
    cv22 = m12**2 + m22**2 + m23**2
    cv23 = m12*m13 + m22*m23 + m23*m33
    cv33 = m13**2+m23**2+m33**2
    return cv11,cv12,cv13,cv22,cv23,cv33
              
def rgb_to_luv_polar(df):
    '''
    Put in rgb, puts out the polar luv coordinates lum,sat,hue. 
    '''
    
    rgb = sRGBColor(df['r'],df['g'],df['b'],is_upscaled=True)
    lum,sat,hue = convert_color(rgb,LCHuvColor).get_value_tuple()
    df['l']=lum
    df['c']=sat
    df['h']=hue
    
    return df

def filt_kernels(kernel_val_df,layer,model,direction,amount,amount_type,filt_var):
    
    layer_kernel_vals = kernel_val_df[(kernel_val_df['layer']==layer) &
                                      (kernel_val_df['model']==model)]
    layer_kernel_vals = layer_kernel_vals.dropna()
    if amount_type == 'perc':
        num_kernels = int(np.round(len(layer_kernel_vals['kernel'].unique())*amount))
    elif amount_type == 'count':
        num_kernels = amount
    if direction == 'top':
        cutoff_val = np.sort(layer_kernel_vals[filt_var])[-num_kernels]
        which_kernels = layer_kernel_vals.loc[layer_kernel_vals[filt_var]>=cutoff_val,'kernel']                     
    elif direction == 'bot':
        cutoff_val = np.sort(layer_kernel_vals[filt_var])[num_kernels]
        which_kernels = layer_kernel_vals.loc[layer_kernel_vals[filt_var]<=cutoff_val,'kernel']   
    which_kernels = which_kernels[0:num_kernels]

    return(which_kernels)
              
def save_activations_for_stimset_for_model(stim_set,model_desc,activation_types,out_dir):
    '''
    Wrapper function to allow this whole thing to be parallelized nicely. 
    Stim_set is tuple consisting of (stim_set,stim_subset,image_dict_list), where each image-dict contains the desciptors for each image along with the filepath of the image. Model_desc contains the model name,
    the model url if you want to load in different pre-trained weights, and a list
    of the layers to be included in the model. Out-dir is the directory where you want to drop everything.
    Activation_types is the list of activation types that you want. 
    '''
                  
    model_name,base_model,model_url,model_layers = model_desc
    model = prepare_model(base_model,model_url)
              
    stim_set,stim_subset,image_list = stim_set
    
    print(f"{stim_set},{model_name}:")
    
    for image in image_list:
    
        image_path = image['path']
        image_labels = [label for label in image.keys() if label!='path']
        print(','.join(f'{lab} {image[lab]}' for lab in image_labels))
        layer_activations = get_model_activations_for_object(image_path,model)
        for layer_label in layer_activations: 
            if model_layers!='all' and (layer_label not in model_layers):
                continue
            layer = layer_activations[layer_label]
            layer_name = layer_label[0]
            layer_dim = len(layer.squeeze().shape)
            if ('conv' in layer_name) or ('pool' in layer_name) or (layer_dim>1):
                for activation_type in activation_types:
                    if activation_type == 'feature_means':
                        num_kernels = layer.shape[1]
                        for k in range(num_kernels):
                            kernel_activations_dict = defaultdict(lambda:[])
                            kernel_mean = layer[:,k,:,:].mean()
                            for label in image_labels:
                                kernel_activations_dict[label].append(image[label])
                            kernel_activations_dict['model'].append(model_name)
                            kernel_activations_dict['layer_label'].append(layer_label)
                            kernel_activations_dict['layer'].append(layer_name)
                            kernel_activations_dict['kernel_ind'].append(k)
                            kernel_activations_dict['kernel_x'].append('mean')
                            kernel_activations_dict['kernel_y'].append('mean')
                            kernel_activations_dict['activation'].append(kernel_mean)
                            kernel_activations_dict['activation_type'].append(activation_type)
                            kernel_activations_dict['image_set'].append(stim_set)
                            kernel_activations_dict['image_subset_activation'].append(stim_subset)
                            kernel_activations = pd.DataFrame.from_dict(kernel_activations_dict)
                            out_path = opj(out_dir,f"{model_name}_{layer_name}_{activation_type}_k{k}_{stim_set}_{stim_subset}.csv")
                            create_or_append_csv(kernel_activations,out_path)
                    elif activation_type == 'unit_level':
                        num_kernels = layer.shape[1]
                        for k in range(num_kernels):
                            kernel_activations_dict = defaultdict(lambda:[])
                            for i,j in it.product(range(layer.shape[2]),range(layer.shape[3])):
                                for label in image_labels:
                                    kernel_activations_dict[label].append(image[label])
                                kernel_activations_dict['model'].append(model_name)
                                kernel_activations_dict['layer_label'].append(layer_label)
                                kernel_activations_dict['layer'].append(layer_name)
                                kernel_activations_dict['kernel_ind'].append(k)
                                kernel_activations_dict['kernel_x'].append(i)
                                kernel_activations_dict['kernel_y'].append(j)
                                kernel_activations_dict['activation'].append(layer[0,k,i,j])
                                kernel_activations_dict['activation_type'].append(activation_type)
                                kernel_activations_dict['image_set'].append(stim_set)
                                kernel_activations_dict['image_subset_activation'].append(stim_subset)
                            kernel_activations = pd.DataFrame.from_dict(kernel_activations_dict)
                            out_path = opj(out_dir,f"{model_name}_{layer_name}_{activation_type}_k{k}_{stim_set}_{stim_subset}.csv")
                            create_or_append_csv(kernel_activations,out_path)

            elif 'fc' in layer_name or (layer_dim==1):
                kernel_activations_dict = defaultdict(lambda:[])
                for ind in range(layer.shape[1]):       
                    kernel_mean = layer[0,ind]
                    for label in image_labels:
                        kernel_activations_dict[label].append(image[label])
                    kernel_activations_dict['model'].append(model_name)
                    kernel_activations_dict['layer_label'].append(layer_label)
                    kernel_activations_dict['layer'].append(layer_name)
                    kernel_activations_dict['kernel_ind'].append(ind)
                    kernel_activations_dict['kernel_x'].append('na')
                    kernel_activations_dict['kernel_y'].append('na')
                    kernel_activations_dict['activation'].append(kernel_mean)
                    kernel_activations_dict['activation_type'].append('no_space')
                    kernel_activations_dict['image_set'].append(stim_set)
                    kernel_activations_dict['image_subset_activation'].append(stim_subset)
                kernel_activations = pd.DataFrame.from_dict(kernel_activations_dict)
                out_path = opj(out_dir,f"{model_name}_{layer_name}_fc_kall_{stim_set}_{stim_subset}.csv")
                create_or_append_csv(kernel_activations,out_path)
        # Garbage collection just in case:
        layer_activations.clear()
        del layer_activations
              
def compute_etas_for_model_and_imageset_fast(image_set,model_desc,which_layer_dict):
    '''
    This will (relatively) quickly compute the etas for color, shape, and their interaction, 
    by using the "computational" sum of squares formulas as an alternative to saving ALL the activations.
    This should save a ton of space and time. 
    
    Inputs:
    image_set: tuple consisting of (image_set_labels,stim_dict,entry_types), same format as get_rdms.
    That is, image_set_labels is a dictionary describing the image set that'll be used in the output
    dataframe, stim_dict is a dictionary of the stimuli where each key is the value of the two IVs,
    and entry_types gives the variable names of the two IVs. 
    
    model_desc: tuple consisting of the desired model label, the base model name,
    and the URL for loading different weights, OR 'na' if default weights are desired. 
    
    which_layer_dict: dictionary of lists specifying which layers to use for each model, 
    can be either just the layer name or the tuple (e.g. ('conv1',1)), or 'all' if all layers
    are desired. 
    '''
    
    image_set_labels,stim_dict,entry_types = image_set
    iv1_name,iv2_name = entry_types
    model_name,base_model,model_url = model_desc
    
    image_set_display = ', '.join(list(image_set_labels.values()))
    print(f"Calculating ANOVAs for {model_name}, image set {image_set_display}") 
    
    if model_name not in which_layer_dict:
        raise Exception("Please specify which layers are desired.")
    
    which_layers = which_layer_dict[model_name]
    model = prepare_model(base_model,model_url)
    
    # Get the different values of the two IVs and count the ns.          
              
    iv1_vals = []
    iv2_vals = []
    
    for (iv1_val,iv2_val) in stim_dict:
        if iv1_val not in iv1_vals:
            iv1_vals.append(iv1_val)
        if iv2_val not in iv2_vals:
            iv2_vals.append(iv2_val)
              
    n_iv1 = len(iv1_vals)
    n_iv2 = len(iv2_vals)
    
    n_total = n_iv1*n_iv2
    
    # Here's how it'll be: there'll be a dictionary of several numpy arrays for tallying 
    # the sum-of-squares statistics for EVERY unit. We need:
    # sum(X^2)
    # sum(X)^2
    # sum(each row)^2
    # sum(each column)^2
    
    # These are the ingredients of the-sum-of-squares formulas, and EACH needs to be tabulated
    # for EVERY unit. 
    
    # Run a sample image through to get the proper dimensions, and initialize the different variables. 
    
    ss_dict = OrderedDict({})
    sample_activations = get_model_activations_for_object(stim_dict[(iv1_vals[0],iv2_vals[0])],model)
    for layer in sample_activations:
        if (layer not in which_layers) and (layer[0] not in which_layers):
            continue
        ss_dict[layer] = OrderedDict({})
        layer_dim = np.squeeze(sample_activations[layer]).shape
        ss_dict[layer]["sum_x"] = np.zeros(layer_dim)
        ss_dict[layer]["sum_x_squared"] = np.zeros(layer_dim)
        ss_dict[layer]["row_sums"] = OrderedDict({})
        ss_dict[layer]["col_sums"] = OrderedDict({})
        for iv1_val in iv1_vals:
            ss_dict[layer]["row_sums"][iv1_val] = np.zeros(layer_dim)
        for iv2_val in iv2_vals:
            ss_dict[layer]["col_sums"][iv2_val] = np.zeros(layer_dim)
    
    # Now, we can crawl through each image, and for each one, increment all the relevant sums. 
    
    print("Tallying sum-of-squares values...")
    for (iv1_val,iv2_val) in stim_dict:
        print(f"\tStimulus {iv1_val},{iv2_val}")
        stim_path = stim_dict[(iv1_val,iv2_val)]
        stim_activations = get_model_activations_for_object(stim_path,model)
        for layer in stim_activations:
            if (layer not in which_layers) and (layer[0] not in which_layers):
                continue
            layer_activations = np.squeeze(stim_activations[layer])
            layer_activations_squared = layer_activations**2
            ss_dict[layer]["sum_x"] += layer_activations
            ss_dict[layer]["sum_x_squared"] += layer_activations_squared
            ss_dict[layer]['row_sums'][iv1_val] += layer_activations
            ss_dict[layer]['col_sums'][iv2_val] += layer_activations
    
    # Now, compute the row and column sums.
    
    output_dict = defaultdict(lambda:[]) 
              
    for layer in ss_dict:
        layer_num_dims = len(np.squeeze(sample_activations[layer]).shape)
              
        ss_dict[layer]['ss_total'] = ss_dict[layer]['sum_x_squared']-(ss_dict[layer]['sum_x']**2)/n_total
            
        ss_dict[layer]['ss_between'] = ss_dict[layer]['ss_total'] # since ss_within = 0
              
        # Tricky part is getting ss for the two variables.       
              
        ss_dict[layer]['ss_var1'] = 0
        for iv1_val in ss_dict[layer]['row_sums']:
            ss_dict[layer]['ss_var1'] += (ss_dict[layer]['row_sums'][iv1_val]**2)/n_iv2
        ss_dict[layer]['ss_var1'] = ss_dict[layer]['ss_var1']-(ss_dict[layer]['sum_x']**2)/n_total
        
        ss_dict[layer]['ss_var2'] = 0
        for iv2_val in ss_dict[layer]['col_sums']:
            ss_dict[layer]['ss_var2'] += (ss_dict[layer]['col_sums'][iv2_val]**2)/n_iv1
        ss_dict[layer]['ss_var2'] = ss_dict[layer]['ss_var2']-(ss_dict[layer]['sum_x']**2)/n_total
              
        ss_dict[layer]['ss_interaction'] = ss_dict[layer]['ss_between']-ss_dict[layer]['ss_var1']-ss_dict[layer]['ss_var2']
        
        ss_dict[layer]['eta2_var1'] = ss_dict[layer]['ss_var1']/ss_dict[layer]['ss_total']
        ss_dict[layer]['eta2_var1'][np.isinf(ss_dict[layer]['eta2_var1'])] = np.nan 
              
        ss_dict[layer]['eta2_var2'] = ss_dict[layer]['ss_var2']/ss_dict[layer]['ss_total']
        ss_dict[layer]['eta2_var2'][np.isinf(ss_dict[layer]['eta2_var2'])] = np.nan 
              
        ss_dict[layer]['eta2_interaction'] = ss_dict[layer]['ss_interaction']/ss_dict[layer]['ss_total']
        ss_dict[layer]['eta2_interaction'][np.isinf(ss_dict[layer]['eta2_interaction'])] = np.nan

        if layer_num_dims >1:
            ss_dict[layer]['eta2_var1_kernelmeans'] = np.nanmean(ss_dict[layer]['eta2_var1'],axis=(1,2))
            ss_dict[layer]['eta2_var1_kernelmeans'] = ss_dict[layer]['eta2_var1_kernelmeans'][~np.isnan(ss_dict[layer]['eta2_var1_kernelmeans'])]
            ss_dict[layer]['eta2_var2_kernelmeans'] = np.nanmean(ss_dict[layer]['eta2_var2'],axis=(1,2))
            ss_dict[layer]['eta2_var2_kernelmeans'] = ss_dict[layer]['eta2_var2_kernelmeans'][~np.isnan(ss_dict[layer]['eta2_var2_kernelmeans'])]   
            ss_dict[layer]['eta2_interaction_kernelmeans'] = np.nanmean(ss_dict[layer]['eta2_interaction'],axis=(1,2))
            ss_dict[layer]['eta2_interaction_kernelmeans'] = ss_dict[layer]['eta2_interaction_kernelmeans'][~np.isnan(ss_dict[layer]['eta2_interaction_kernelmeans'])]
        else:
            ss_dict[layer]['eta2_var1_kernelmeans'] =  ss_dict[layer]['eta2_var1']
            ss_dict[layer]['eta2_var2_kernelmeans'] =  ss_dict[layer]['eta2_var2']
            ss_dict[layer]['eta2_interaction_kernelmeans'] =  ss_dict[layer]['eta2_interaction']  
                  
        # Now save to output. Have it in a few different formats for convenience: mean etas of all units, 
        # stdevs both of ALL units and of the kernelwise means. 
              
        for image_set_label in image_set_labels:
            output_dict[image_set_label] = image_set_labels[image_set_label]
        output_dict['model'].append(model_name)    
        output_dict['layer_label'].append(layer)
        output_dict['layer'].append(layer[0])          
        output_dict[f"{iv1_name}_eta2_mean"].append(np.nanmean(ss_dict[layer]['eta2_var1']))
        output_dict[f"{iv1_name}_eta2_stdev"].append(np.nanstd(ss_dict[layer]['eta2_var1']))
        output_dict[f"{iv1_name}_eta2_stdev_kernelmeans"].append(np.nanstd(ss_dict[layer]['eta2_var1_kernelmeans']))
        output_dict[f"{iv2_name}_eta2_mean"].append(np.nanmean(ss_dict[layer]['eta2_var2']))
        output_dict[f"{iv2_name}_eta2_stdev"].append(np.nanstd(ss_dict[layer]['eta2_var2']))
        output_dict[f"{iv2_name}_eta2_stdev_kernelmeans"].append(np.nanstd(ss_dict[layer]['eta2_var2_kernelmeans'])) 
        output_dict[f"interaction_eta2_mean"].append(np.nanmean(ss_dict[layer]['eta2_interaction']))
        output_dict[f"interaction_eta2_stdev"].append(np.nanstd(ss_dict[layer]['eta2_interaction']))
        output_dict[f"interaction_eta2_stdev_kernelmeans"].append(np.nanstd(ss_dict[layer]['eta2_interaction_kernelmeans'])) 
    
    output_df = pd.DataFrame.from_dict(output_dict)
    return(output_df,ss_dict)
              
def mydelete(filename):
    """
    If a file, deletes it. If a directory, deletes it and all subfolders.
    If non-existent, does nothing and does no errors.
    """

    if os.path.exists(filename):
         try:
             if os.path.isdir(filename):
                 # delete folder
                 shutil.rmtree(filename)
                 return
             else:
                 # delete file
                 os.remove(filename)
                 return
         except:
             return
    else:
         return
              
def inds2cols(df):
    """
    Takes in a dataframe or series with indices and converts them all to columns.
    """

    # make it a dataframe if not already:

    if isinstance(df,pd.core.series.Series):
        df = df.to_frame()

    num_inds = len(df.index.names)

    for i in range(0,num_inds):
        df.reset_index(level=0,inplace=True)

    # This operation reverses the column order, so need to reverse it back:

    cols = df.columns.tolist()
    cols = cols[num_inds-1::-1] + cols[num_inds:]
    df = df[cols]
    return df
              
def my_agg(df,group_vars,data_vars,agg_funcs):
    '''
    Put in a dataframe, the variables you want to group by, the
    data variables, and the aggregation function, and spits out the
    data after applying the split/apply/combine operation.
    Keeps everything as columns rather than indices.
    Also, if you provide multiple aggregation functions,
    makes it into strings rather than tuples (e.g., x_mean rather than (x,mean))

    If you want data variables to be all non-grouping variables,
    put in 'rem' as the data_vars argument. Vice versa for the other way
    around.
    '''

    if data_vars == 'rem':
        data_vars = []
        col_names = df.columns.values
        for col in col_names:
            if not col in group_vars:
                data_vars.append(col)

    if group_vars == 'rem':
        group_vars = []
        col_names = df.columns.values
        for col in col_names:
            if not col in data_vars:
                group_vars.append(col)

    groups = df.groupby(group_vars)
    agg_df = groups[data_vars].agg(agg_funcs)
    if type(agg_funcs)==list:
        agg_df.columns = agg_df.columns.map('_'.join)
    agg_df = inds2cols(agg_df)
    return agg_df
              
def df_filtsort(df,sortlist):
    '''
    Takes in a Pandas dataframe, and a tuple or list of tuples to sort by; in each tuple, the first value
    is the column name, and the second value is a list specifying the custom sort order. The function will only
    include values in the list, in the desired order. If multiple tuples, it'll sort by all of them.
    '''
  
    # filter it first
 
    if type(sortlist)==tuple:
        sortlist = [sortlist]
    
    for (sortcol,sort_order) in sortlist:
        df_out = df[df[sortcol].isin(sort_order)] 

    dummy_col_names = []  
    for ind,(sortcol,sort_order) in enumerate(sortlist):
        recode_dict = {name:num for num,name in enumerate(sort_order)}
        df_out.loc[:,'dummycol'+str(ind)] = df_out[sortcol].replace(recode_dict)
        dummy_col_names.append('dummycol'+str(ind))
    
    df_out = df_out.sort_values(by=dummy_col_names)
    df_out = df_out.drop(dummy_col_names,axis=1)
    return df_out


def my_pivot(df,index_cols,var_cols,value_cols):
    '''
    Functionally identical to the default Pandas pivot function, but allows you to have multiple 
    values columns, and automatically flattens them so no annoying multi-index bullshit
    '''
    
    if type(value_cols) != list:
        value_cols = [value_cols]
    if type(var_cols) != list:
        var_cols = [var_cols]
    if type(index_cols) != list:
        index_cols = [index_cols]
        
    sub_dfs = []
        
    for value_col in value_cols:
        for var_col in var_cols:
            new_df = df.pivot_table(value_col,index_cols,var_cols)
            if len(value_cols)>1:
                colname_dict = {val:val+'_'+value_col for val in list(df[var_col].unique())}
                new_df = new_df.rename(columns=colname_dict)
            new_df = new_df.reset_index()
            new_df.columns.name = ''
            sub_dfs.append(new_df)
    
    # Now merge all the dfs together. Start with the first one, then tack on columns from the others. 
  
    out_df = sub_dfs[0]
    
    for ind in range(1,len(sub_dfs)):
        new_cols = [col for col in sub_dfs[ind].columns.values if col not in out_df.columns.values]
        for new_col in new_cols:
            out_df[new_col] = sub_dfs[ind][new_col]
    out_df = out_df.reset_index(drop=True)       
    return out_df
              
def df_subset(df,subset_list):
    '''
    Takes subset of a dataframe based on indicated columns. Give a list of tuples, where first element of each tuple
    is the column name, and the second element is which elements to keep. If it's a list, then it keeps elements
    in that list, if it's a single value, it'll keep elements equal to that value. This should save some ink.
    
    6/25/19 addition: now, if instead of a tuple you just put a single string, it'll treat that string 
    as both the variable name and the value of the variable... save even more ink. 
    '''
    
    for col in subset_list:
        if type(col)==tuple:
            if type(col[1])==list:
                df = df[df[col[0]].isin(col[1])]
            else:
                df = df[df[col[0]]==col[1]]    
        elif type(col)==str:
            var_value = eval(f"{col}")
            if type(var_value)==list:
                df = df[df[col.isin(var_value)]]
            else:
                df = df[df[col]==var_value] 
    return df   

def pickle_read(fname):
    '''
    Convenience function to read a pickle. 
    '''
    
    output = pickle.load(open(fname,'rb'))
    return output

def pickle_dump(obj,fname):
    '''
    Convenience function to write a pickle. 
    '''
    
    pickle.dump(obj,open(fname,"wb"))
    
def my_parallel_process(func,arg_dict,num_cores,method='pathos',make_combinations=True):
    '''
    Wrapper for parallel processing so I don't have to keep repeating the boilerplate nonsense. 
    Arg_dict will be a dictionary of the desired arguments: either single arguments, or lists if you want
    to iterate through (these would be the things you want to parallelize over). The basic logic
    of the function is that it'll unpack the inputs into one dictionary for each input set, then
    do all these in parallel. 
    '''
    
    # Depending on which method is used, different input structure is needed. 
    
    if make_combinations:
        for arg in arg_dict:
            if type(arg_dict[arg]) != list:
                arg_dict[arg] = [arg_dict[arg]] 
        arg_dict_list = [dict(zip(arg_dict.keys(), values)) for values in it.product(*arg_dict.values())]
    else:
        arg_dict_list = arg_dict
    
    if method=='pathos':
        pool = pathos.multiprocessing.ProcessPool(nodes=num_cores)
        worker_list = [pool.apipe(func,**kwd_dict) for kwd_dict in arg_dict_list]
        results = [p.get() for p in worker_list]
        pool.close()
        pool.join()
        pool.clear()
    
    if method=='multiprocessing':   
        pool = multiprocessing.Pool(processes=num_cores)
        worker_list = [pool.apply_async(func,kwds=kwd_dict) for kwd_dict in arg_dict_list]
        results = [p.get() for p in worker_list]
        pool.close()
        pool.join()
    
    return results

def create_or_append_csv(df,path):
    ''' 
    If the path already exists, append the dataframe to it, else make a new one.
    '''
    
    if not os.path.exists(path):
        df.to_csv(path)
    else:
        df.to_csv(path,mode='a',header=False)  

def stderr(array):
    output = np.std(array)/(float(len(array))**.5)
    return output

def update_list_dict(list_dict,**kwargs):
    '''
    Convenience function to append a bunch of things to a dictionary of lists (that will typically
    become a Pandas dataframe later). Just put in all the desired items as keyword arguments. 
    '''
    
    for key in kwargs:
        list_dict[key].append(kwargs[key])
    return list_dict

def get_upper_triang(matrix):
    n = matrix.shape[0]
    inds = np.triu_indices(n,1)
    vals = matrix[inds]
    return vals

def correlate_upper_triang(m1,m2,func=lambda x,y:np.corrcoef(x,y)[0,1]):
    out_val = func(get_upper_triang(m1),get_upper_triang(m2))
    return out_val