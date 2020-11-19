''' 

Function library for using PyTorch to run representational similarity analysis on convolutional neural networks.

This library contains a collection of functions for easily conducting representational similarity analysis (RSA) and related 
analyses on convolutional neural networks (CNNs), serving as a wrapper for various PyTorch functions. There are also 
many utility functions to improve quality of life. 

'''

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import model_zoo
import torchvision
import numpy as np
from os.path import join as opj
from collections import OrderedDict
from collections import defaultdict
import ipdb
import os 
from PIL import Image, ImageOps
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
import pandas as pd
from sklearn import svm
from joblib import Parallel, delayed
import multiprocessing
import sys
from importlib import reload
import time
import statsmodels.api as sm
from scipy import math
from scipy import stats
import cornet
import pathos

###################################################
################ UTILITY FUNCTIONS ################
###################################################

def parallel_process(func,arg_dict,num_cores,method='pathos',make_combinations=False):
    '''
    Wrapper to streamline running a function many times in parallel. 
    
    Utility function to run a function many times in parallel, with many different combinations of arguments. 
    Provide the function, along with a dictionary specifying the arguments to feed into the function. Can run either with 
    the pathos or multiprocessing package. Also contains a flag, make_combinations, to do all combinations of input
    arguments if desired.
    
    Args:
        func: 
            The function you want to run many times.
        arg_dict: 
            The structure depends on the make_combinations flag. If this flag is set to false, input a list of dictionaries, 
            where each dictionary contains a set of arguments for one call of the function (i.e., you manually specify each set
            of arguments for each function call); and each key is an argument name, with the corresponding value being an argument value.
            If make_combinations is set to false, input a single dictionary of lists, where each key is an argument name, and each value is a list 
            of values to put in for that argument. In the latter case, the function will run every possible permutation of argument values
            from the different lists.
        num_cores:
            The number of cores to run in parallel
        method:
            Which Python module to use for the multiprocessing; either pathos or multiprocessing. 
        make_combinations:
            Whether arg_dict is a list of dictionaries containing each set of arguments to put in (in this case put False),
            or a single dictionary of lists (in this case put True) 
    
    Returns:
        List containing the results of all calls to the function.
        
    Examples:
        >>> def mult(x,y):
                return x*y
    
        >>> parallel_process(func = mult,
                             arg_dict = [{'x':2,'y':4},{'x':5,'y':3}],
                             num_cores = 2,
                             method = 'pathos',
                             make_combinations = False)
            [8,15]
            
        >>> parallel_process(func = mult,
                             arg_dict = {'x':[2,1,3],'y':[6,2,4]},
                             num_cores=2,
                             method='pathos',
                             make_combinations=True)
            [12, 4, 8, 6, 2, 4, 18, 6, 12]
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

def corr_dissim(x,y):
    """
    Convenience function for computing 1-r for two vectors (to turn correlation into a dissimilarity metric). 
    
    Function that returns 1-r for two vectors x and y. 

    Args:
        x: First vector (numpy array)
        y: Second vector (numpy array)

    Returns:
        1-r 
    """
    
    if np.max(np.abs(x-y))==0:
        r = 0
    elif (np.sum(np.abs(x))==0) or (np.sum(np.abs(y))==0):
        r = np.nan
    else:
        r = 1-np.corrcoef(x,y)[0,1]
        
    return r

# Definition of the default dissimilarity functions. 

dissim_defs = {'corr':corr_dissim,
               'euclidean_distance':lambda x,y:np.linalg.norm(x-y)}

def stderr(array):
    '''
    Convenience function that computes the standard error of an array (STD/sqrt(n))
    
    Args:
        array: 1D numpy array
        
    Returns:
        The standard error of the array.
    
    '''
    output = np.std(array)/(float(len(array))**.5)
    return output

def almost_equals(x,y,thresh=5):
    '''
    Convenience function for determining whether two 1D arrays (x and y) are approximately equal to each other (within a range of thresh for each element)
    
    Args:
        x: 1D numpy array to compare to y
        y: 1D numpy array to compare to x
        thresh: how far apart any element of x can be from the corresponding element of y
        
    Returns: 
        True if each element of x is within thresh of each corresponding element of y, false otherwise. 
    
    '''
    
    return all(abs(a[i]-b[i])<thresh for i in range(len(a)))

def filedelete(fname):
    """
    Convenience function to delete files and directories without fuss. Deletes the file or directory if it exists, does nothing otherwise.
    
    Args:
        fname: path of the file or directory
        
    Returns:
        Returns nothing. 
    """

    if os.path.exists(fname):
         try:
             if os.path.isdir(fname):
                 # delete folder
                 shutil.rmtree(fname)
                 return
             else:
                 # delete file
                 os.remove(fname)
                 return
         except:
             return
    else:
         return
        
def pickle_read(fname):
    '''
    Convenience function to read in a pickle file.
    
    Args:
        fname: file path of the pickle file
        
    Returns:
        The Python object stored in the pickle file. 
    '''
    
    output = pickle.load(open(fname,'rb'))
    return output

def pickle_dump(obj,fname):
    '''
    Convenience function to dump a Python object to a pickle file.
    
    Args:
        obj: Any pickle-able Python object.
        fname: The file path to dump the pickle file to.
        
    Returns:
        Nothing.
    '''
    
    pickle.dump(obj,open(fname,"wb"))

###################################################
################ IMAGE PREPARATION ################
###################################################

def nonwhite_pixel_mask(image,thresh=230):
    '''
    Takes in a 3D numpy array representing an image (m x n x 3), and returns a 2D boolean mask indicating every non-white pixel. 
    Useful for making a mask indicating the stimulus footprint on a white background.
    
    Args:
        image: 3D numpy array (width m, height n, 3 pixel channels)
        thresh: RGB cutoff to count as white; if all three of R, G, and B exceed this value, the pixel is counted as white.
        
    Returns:
        2D numpy array with a 0 for every white pixel and a 1 for every non-white pixel. 
    '''
    
    x,y = image.shape[0],image.shape[1]
    mask = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if all(image[i,j,:]>thresh):
                mask[i,j] = 0
            else:
                mask[i,j] = 1
    return mask 

def save_mask(mask,fname):
    '''
    Takes in a 2D numpy array with a binary mask (1s and 0s), and saves it as an RGB JPEG image.
    
    Args:
        mask: 2D numpy array
        fname: Filename to save the image. 
        
    Returns:
        The Image object (in addition to saving it at specified filename)
    '''
    
    new_mask = Image.fromarray((np.dstack([mask,mask,mask])*255).astype('uint8'),'RGB')
    new_mask.save(fname,format='jpeg')
    return(new_mask)
    
def shrink_image(input_image,shrink_ratio,pad_color=(255,255,255),output_size=(224,224)):
    '''
    Put in a PIL image, and it'll shrink the image, while keeping the same resolution as original image 
    and padding the sides with the desired color. 
    
    Args:
        input_image: PIL image you want to shrink. 
        shrink_ratio: how much to shrink each dimension of the image by (e.g., .5 to halve the length and width)
        pad_color: RGB tuple indicating the color to fill the margins of the image with
        output_size: desired dimensions of the output image
        
    Returns:
        PIL image, shrunk to desired dimensions.
    '''
    
    orig_size = input_image.size
    pad_amount = int(round(orig_size[0]*(1-shrink_ratio)/(2*shrink_ratio)))
    new_size = (orig_size[0]+pad_amount*2,orig_size[1]+pad_amount*2)
    output_image = Image.new("RGB",new_size,color=pad_color)
    output_image.paste(input_image,(pad_amount,pad_amount))
    output_image  = output_image.resize(output_size,Image.ANTIALIAS)
    return(output_image)

def alphatize_image(im,alpha_target=(255,255,255),alpha_range=5):
    '''
    Takes in a PIL image, and makes RGB values within a given range transparent. 
    Any pixel all of whose RGB values are within alpha_range of alpha_target
    will be made transparent. Used in MDS plotting function.
    
    Args:
        im: Image in PIL format
        alpha_target: RGB triplet indicating which values to make transparent.
        alpha_range: how far away a particular RGB value can be from alpha_target and still
            make it transparent. 
            
    Returns:
        PIL image with all pixels in range of alpha_target made transparent. 
    '''
    
    image_alpha = im.convert('RGBA')
    pixel_data = list(image_alpha.getdata())

    for i,pixel in enumerate(pixel_data):
        if almost_equals(pixel[:3],alpha_target,alpha_range):
            pixel_data[i] = (255,255,255,0)

    image_alpha.putdata(pixel_data)
    return image_alpha

# Standard transformation to apply to images before reading them in 

standard_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def preprocess_image(input_image,transform=standard_transform):
    '''
    Preprocesses an image to prepare it for running through a network; resizes it, 
    turns it into a tensor, standardizes the mean and SD of the pixels, 
    and pads it to be 4D, as required by many networks (one "dummy" dimension).
    
    args:
        input_image: PIL image 
        
    return: 
        Preprocessed image
        
    '''
    
    output_image = transform(input_image).unsqueeze(0)
    return output_image

###################################################
############# DEALING WITH MODELS #################
###################################################

def get_layer_activation(module,input_,output_):
    '''
    
    Utility function that is attached as a "forward hook" to each model layer to store the activations
    as a numpy array in the dictionary layer_activations. 
    
    '''
    layer_activations[module.layer_name] = output_.cpu().detach().numpy() 

def index_nested(the_item,indices):
    '''
    
    Utility function to help with retrieving model layers that are buried in many levels of indexing that 
    can be either numerical (array) or an attribute.
    
    args:
        the_item: top-level item you want to pull something out of
        indices: list of indices; each element is either a number (if that level of indexing is array-based),
            or a string if that level is indexing based on attribute. 
    
    return: 
        Whatever you've pulled out of the_item following the given indices. 
        
    Examples:
        >>> import torchvision.models as models
        >>> alexnet = models.alexnet()
        >>> index_nested(alexnet._modules,['features', 11])
        ReLU(inplace=True)
    
    '''
    for ind in indices:
        if type(ind)!=tuple:
            the_item = the_item[ind]
        else:
            the_item = getattr(the_item,ind[1])
    return the_item    
    
def fetch_layers_internal(current_layer,layer_pointers,layer_indices,layer_counter):
    '''

    Internal helper function that recursively crawls through all layers and sublayers of a network
    and pulls out their addresses for easy reference. 

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
    
    Takes in a CNN model, and returns "addresses" of all the layers to refer to them easily; this is useful
    since different CNN models can be structured in different ways, with various chunks, layers, sublayers, etc.
    
    args:
        model: a PyTorch CNN model. Currently, at least AlexNet, VGG19, ResNet-50, GoogLeNet, and CORNet-S
        are supported; no guarantee yet that others will work. 
    returns: 
        ordered dictionary where each key is a layer label of format ('relu2',3) (that is, the
        second relu layer and third layer overall), and each value is the set of indices needed
        to refer to that layer in the model. The index_nested function can then be use those 
        indices to refer to a layer of a model when needed. 
        
    Examples:
        >>> import torchvision.models as models
        >>> alexnet = models.alexnet()
        >>> fetch_layers(alexnet)
        OrderedDict([(('conv1', 1), ['features', 0]), (('relu1', 2), ['features', 1]), (('maxpool1', 3), ['features', 2]), (('conv2', 4), ['features', 3]), (('relu2', 5), ['features', 4]), (('maxpool2', 6), ['features', 5]), (('conv3', 7), ['features', 6]), (('relu3', 8), ['features', 7]), (('conv4', 9), ['features', 8]), (('relu4', 10), ['features', 9]), (('conv5', 11), ['features', 10]), (('relu5', 12), ['features', 11]), (('maxpool3', 13), ['features', 12]), (('avgpool1', 14), ['avgpool']), (('dropout1', 15), ['classifier', 0]), (('fc1', 16), ['classifier', 1]), (('relu6', 17), ['classifier', 2]), (('dropout2', 18), ['classifier', 3]), (('fc2', 19), ['classifier', 4]), (('relu7', 20), ['classifier', 5]), (('fc3', 21), ['classifier', 6])])
    
    '''
    
    layer_pointers = OrderedDict()
    layer_counter = 1
    for macro_layer in model._modules:
        layer_pointers,layer_counter = fetch_layers_internal(model._modules[macro_layer],layer_pointers,[macro_layer],layer_counter)
    return layer_pointers

def prepare_models(models_dict):
    '''
    
    Prepares the models you want to use: loads them with specified weight settings, and registers the
    forward hooks that allow you to save intermediate activations. 
    
    args: 
        models_dict: 
            Dictionary specifying the models to prepare, and the weight settings to use. 
            Each key is your internal name you wish to use for your model. Each value is a tuple
            (base_model,weight_setting), where base_model is the model to use (e.g., "alexnet"),
            and weight_setting is "trained" for the trained version of the network,
            "random" for an untrained version of the model with random weights, or a URL
            linking to the state_dict for the weights to use if you wish to use some 
            custom setting of the weights.
            
    returns:
        A dictionary where each key is your internal name of the model, and each value is the model itself. 
    
    '''
    models_prepped = OrderedDict()
    for model_name in models_dict:
        base_model,weights_url = models_dict[model_name]
        models_prepped[model_name] = prepare_model(base_model,weights_url)
    return models_prepped

def prepare_model(which_model,weights_opt='trained'):
    '''
   
    Prepares a single model to use: loads them with specified weight settings, and registers the
    forward hooks that allow you to save intermediate activations. Better to use the prepare_models
    function, since the output format is assumed by some of the other functions in this library. 
    
    args: 
        which_model: Base model to use (e.g., "alexnet")
        weights_opt: Which weights to use. Set to "trained" for trained version of the network,
            "random" for random weights, or a url linking to the state_dict for the weights
            if you wish to use some custom setting of the weights. 
            
    returns:
        The prepared model. 
   
    '''
    
    if weights_opt == 'random':
        pretrain_opt = False
    else:
        pretrain_opt = True
    
    if which_model=='googlenet':
        if pretrain_opt==True:
            model = torch.hub.load('pytorch/vision', 'googlenet', pretrained=pretrain_opt)
        else: 
            model = torch.hub.load('pytorch/vision', 'googlenet', pretrained=pretrain_opt,aux_logits=False)
            state_dict = model.state_dict()
            state_dict['fc.bias']  = state_dict['fc.bias']*0
            model.load_state_dict(state_dict)
            
    elif which_model=='cornet_z':
        model = cornet.cornet_z(pretrained=pretrain_opt,map_location='cpu')
    elif which_model=='cornet_s':
        model = cornet.cornet_s(pretrained=pretrain_opt,map_location='cpu')
    else:  
        model = getattr(models,which_model)(pretrained=pretrain_opt)
    model.eval()
    
    if (weights_opt != 'trained') and (weights_opt != 'random'):
        checkpoint = model_zoo.load_url(weights_opt,map_location='cpu')
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

def get_model_activations_for_image(fname,model):
    '''
    
    Takes in the file path of an image, and a prepared model (use the prepare_model or prepare_models function),
    runs the image through the model, and returns a dictionary containing the model activations in each layer.
    
    args:
        fname: file path of the image to run through a model
        model: a CNN model object that has had the forward hooks attached to save the intermediate activations
            (use the prepare_model or prepare_models function to do this)
    
    returns:
        Dictionary where each key is a tuple denoting the layer (e.g., (conv3,5) is the third convolutional
        layer and the fifth layer overall), and each value is a numpy array of the layer activations. 
        
    '''
    
    image = Image.open(obj_fname).convert('RGB')    
    global layer_activations 
    layer_activations = OrderedDict()
    preprocessed_image = preprocess_image(image)
    model.forward(preprocessed_image)
    return layer_activations
              
###################################################
############# DEALING WITH MATRICES ###############
###################################################

def get_upper_triang(matrix):
    '''
    Utility function that pulls out the upper triangular values of a matrix and returns them as a 1D vector.
    
    args:
        matrix: a 2D square numpy array
    returns:
        1D numpy vector with the upper triangular elements. 
    '''
    n = matrix.shape[0]
    inds = np.triu_indices(n,1)
    vals = matrix[inds]
    return vals

def correlate_upper_triang(m1,m2,func=lambda x,y:np.corrcoef(x,y)[0,1]):
    '''
    Utility function that correlates the upper triangular values of two matrices. 
    
    args:
        m1,m2: the two matrices in question
        func: the function to use; defaults to Pearson correlation, put in your own function (e.g., Spearman)
        if desired.
    returns:
        Correlation coefficient between the upper triangular values of the two matrices. 
    '''
    out_val = func(get_upper_triang(m1),get_upper_triang(m2))
    return out_val

def get_array_ranks(array):
    '''
    Put in array, spits out the same array with the entries replaced by their ranks after being sorted (in ascending order)
    Ties are sorted arbitrarily. 
    
    args: 
        array: Numpy array of any size.
    returns:
        Array where each element corresponds to the rank of the element after sorting 
        (e.g., the smallest element in the original array has value 0 in the returned array,
        the seconds-smallest have value 1, and so on)
    '''
    
    return array.ravel().argsort().argsort().reshape(array.shape)

def get_extreme_dissims_total(input_rdm,num_items='all'):
    
    '''
    Takes in a representational dissimilarity matrix and a desired number of items, and returns a subsetted
    matrix with the items that maximize the total pairwise dissimilarity among all items (more colloquially,
    the items in the original matrix that are as "different as possible" from each other)
    
    args: 
        input_rdm: representational dissimilarity matrix you wish to draw from
        num_items: how many items to draw from input_rdm
        
    returns:
        output_rdm: the subsetted representational similarity matrix with the maximally dissimilar items
        inds_to_extract: the indices of the items from the original matrix that ended up being used
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
    Takes in a representational dissimilarity matrix and a desired number of items, and returns a subsetted
    matrix with the items with a mean pairwise similarity that is as close as possible to a target value
    (e.g., a set of items with a mean pairwise similarity that is as close as possible to .7)
    
    args: 
        input_rdm: representational dissimilarity matrix you wish to draw from
        target_val: the target similarity value for the output RDM
        num_items_final: how many items to draw from input_rdm
        
    returns:
        output_rdm: The subsetted representational similarity matrix 
        inds_to_extract: The indices of the items from the original matrix that ended up being used
        mean_dissim: The actual mean dissimilarity (which will hopefully be close to the target value!)
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
    Takes in a representational dissimilarity matrix and a desired number of items, and returns a subsetted
    matrix with the items that produces a maximally UNIFORM range of similarities (some very similar, 
    some very dissimilar).
    
    args: 
        input_rdm: representational dissimilarity matrix you wish to draw from
        num_items: how many items to draw from input_rdm
        
    returns:
        output_rdm: the subsetted representational similarity matrix with the maximally uniform range of dissimilarities
        inds_to_extract: the indices of the items from the original matrix that ended up being used
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

###################################################
############# INPUT AND OUTPUT ####################
###################################################

def create_or_append_csv(df,path):
    '''
    
    Utility function that creates a CSV from a pandas dataframe if no CSV with the given filepath exists;
    else, if the CSV already exists, appends the contents of the pandas dataframe to the CSV
    at that filepath. 
    
    args: 
        df: A pandas dataframe
        path: The desired file path
        
    returns:
        Nothing
    
    '''
    
    if not os.path.exists(path):
        df.to_csv(path)
    else:
        df.to_csv(path,mode='a',header=False)
        
def combine_csvs(path,keys=[]):
    '''
    Utility function that takes in a list of filenames, and loads them all and combines them into a single pandas dataframe.
    Alternatively, can take in a path for a directory, and it'll combine all the dataframes in that directory
    into one. Can also specify "keys": requires substrings of the CSV filename for them to be included
    in the new CSV (e.g., "cat" if you only want the CSVs with "cat" in the filename)
    
    args: 
        path: Either a list of filepaths for the CSVs you want to combine, or the path of a directory with the desired CSVs.
        keys: List of substrings that must appear in a CSV's filename for it to be included.
        
    returns: 
        Pandas dataframe consisting of all the CSVs combined.
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

###################################################
############# DATAFRAME OPERATIONS ################
###################################################

def df_inds2cols(df):
    """
    
    Utility function to convert all indices in a dataframe to columns, so they can be handled in a more uniform way.
    
    Args:
        df: dataframe whose indices you want to convert
        
    Returns:
        Dataframe that now has columns where there were once indices. 
    
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

def df_agg(df,group_vars,data_vars,agg_funcs):
    '''
    
    Utility function to easily aggregate values in a Pandas dataframe (e.g., if you want to take the mean
    and standard deviation within a bunch of subgroups). There's a built-in Pandas function that does this,
    but this one is more flexible and keeps the new column names as strings rather than tuples, and
    also doesn't convert things to indices rather than columns. 
    
    Args: 
        df: the dataframe whose data you want to aggregate
        group_vars: list of variables you want to aggregate across (e.g., "city" and "gender" to create group
            averages across combinations of city and gender, collapsing across other variables). Put 'rem'
            if you want the group_vars to be all variables EXCEPT the specified data_vars.
        data_vars: the actual values you want to aggregate (e.g., height if you want the average height).
            Put 'rem' if you want the data_vars to be all variables EXCEPT the specified group_vars. 
        agg_funcs: list of functions you want to use for aggregating
        
    Returns:
        Dataframe where the data_vars have been aggregated with the agg_funcs within each value of the group_vars.
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict({'city':['New York','New York','New York','New York',
                                                 'Boston','Boston','Boston','Boston'],
                                         'gender':['Male','Male','Female','Female','Male','Male','Female','Female'],
                                         'height':[70,72,66,65,69,73,64,63],
                                         'income':[50000,100000,80000,150000,120000,90000,70000,110000]})
        >>> pfa.df_agg(df,['city','gender'],['height','income'],[np.mean,np.std])
               city  gender  height_mean  height_std  income_mean    income_std
        0    Boston  Female         63.5    0.707107        90000  28284.271247
        1    Boston    Male         71.0    2.828427       105000  21213.203436
        2  New York  Female         65.5    0.707107       115000  49497.474683
        3  New York    Male         71.0    1.414214        75000  35355.339059
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

def df_subset(df,subset_list):
    '''
    
    Takes in a Pandas dataframe and subsets it to the desired values of specified columns.
    
    Args:
        df: the dataframe you wish to subset
        subset_list: a list of tuples specifying how to subset; each tuple is (column_name,vals_to_use),
        where column_name is the name of the column, and vals_to_use is a list of which values 
        of that column to keep (or can be just a single value if you only want one)
        
    Returns:
        Dataframe that is subsetted in the desired way. 

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict({'city':['New York','New York','Seattle','Seattle',
                                                 'Seattle','Boston','Boston','Boston'],
                             'gender':['Male','Male','Female','Female','Male','Male','Female','Female'],
                             'height':[70,72,66,65,69,73,64,63],
                             'income':[50000,100000,80000,150000,120000,90000,70000,110000]})
        >>> pfa.df_subset(df,[('city',['New York','Seattle']),
                              ('gender','Male')])
               city      gender  height  income
            0  New York   Male       70   50000
            1  New York   Male       72  100000
            4   Seattle   Male       69  120000
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

def df_filtsort(df,sortlist):
    '''
    Takes in a Pandas dataframe, and both filters and sorts it based on the values of specified columns. 
    
    Args: 
        df: The dataframe you wish to filter and sort
        sortlist: A list of tuples, where each tuple is of format (column_name,sort_vals);
            column_name is the name of the column to sort by, sort_vals is the desired
            order of the values in that column. Columns will be sorted by priority
            based on how they are listed.
    Returns:
        Dataframe that is subsetted and sorted in the desired way.
        
    Examples
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict({'city':['New York','New York','New York','Seattle',
                                                 'Seattle','Seattle','Boston','Boston','Boston'],
                                         'gender':['Male','Male','Female','Female','Female','Male','Male','Female','Female'],
                                         'height':[70,72,65,66,65,69,73,64,63],
                                         'income':[50000,100000,120000,80000,150000,120000,90000,70000,110000]})
        >>> pfa.df_filtsort(df,[('city',['Seattle','New York']),('gender',['Female','Male'])])
                   city  gender  height  income
            3   Seattle  Female      66   80000
            4   Seattle  Female      65  150000
            5   Seattle    Male      69  120000
            2  New York  Female      65  120000
            0  New York    Male      70   50000
            1  New York    Male      72  100000
    '''
  
    # filter it first
 
    if type(sortlist)==tuple:
        sortlist = [sortlist]
    
    for (sortcol,sort_order) in sortlist:
        df = df[df[sortcol].isin(sort_order)] 

    dummy_col_names = []  
    for ind,(sortcol,sort_order) in enumerate(sortlist):
        recode_dict = {name:num for num,name in enumerate(sort_order)}
        df.loc[:,'dummycol'+str(ind)] = df[sortcol].replace(recode_dict)
        dummy_col_names.append('dummycol'+str(ind))
    
    df = df.sort_values(by=dummy_col_names)
    df = df.drop(dummy_col_names,axis=1)
    return df

def df_pivot(df,index_cols,var_cols,value_cols,aggfunc='mean'):
    '''
    Functionally identical to the default Pandas pivot function, but allows you to have multiple 
    values columns, and automatically converts indices to columns. 
    
    Args:
        df: the df to turn into a pivot table
        index_cols: list of column names to keep as index columns
        var_cols: list of columns to pivot
        value_cols: list of value columns
        aggfunc: how to aggregate values when there are multiple values per cell
        
    Returns:
        Dataframe converted to pivot table.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict({'city':['New York','New York','New York','Seattle',
                                                 'Seattle','Seattle','Boston','Boston','Boston'],
                                         'gender':['Male','Male','Female','Female','Female','Male','Male','Female','Female'],
                                         'height':[70,72,65,66,65,69,73,64,63],
                                         'income':[50000,100000,120000,80000,150000,120000,90000,70000,110000]})
        >>> pfa.df_pivot(df,['city'],['gender'],['height','income'])
                           city  Female_height  Male_height  Female_income  Male_income
            0    Boston           63.5         73.0          90000        90000
            1  New York           65.0         71.0         120000        75000
            2   Seattle           65.5         69.0         115000       120000
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
            new_df = df.pivot_table(value_col,index_cols,var_cols,aggfunc)
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

###########################################
################# RSA #####################
###########################################

def package_image_sets(image_list,entry_vars,grouping_vars):
    '''
    
    This is a convenience function for preparing image sets for representational similarity analysis. 
    The use case is that you want to make multiple RDMs, where certain variables vary WITHIN the entries
    of each RDM (entry_vars), and some variables vary BETWEEN RDMs (grouping_vars). 
    So, you give it a list of images, specifying the entry_vars and grouping_vars for each image,
    and this function returns a stack of image sets that can then be fed into the get_rdms function.
    
    Args:
        image_list: a list of dictionaries, where each dictionary contains information about one image.
            Each dictionary must AT LEAST contain a key called "path", whose value is the file path 
            of the image. The other keys and values correspond to the other variables and values
            associated with the image. 
        entry_vars: list of variable names that vary across the entries of each RDM.
        grouping vars: list of variable names that vary BETWEEN RDMs. 
        
    Returns:
        List of image sets, where each image set is intended to be turned into an RDM; 
        each image set is represented by a tuple that contains:
            (
            Dictionary of grouping variables for that image set 
            Dictionary of images for that image set; each key is the value of an entry_var, each value is the
                path for that image
            Tuple of the names of the entry variables
            Blank list (which can specify ways of combining activation patterns from images if desired)
            )
        This output structure can be directly fed into the get_rdms function to make an RDM for each 
        image set. 
        
    Examples:
        >>> image_list = [{'color':'red','shape':'square','path':'red_square.jpg'},
                          {'color':'blue','shape':'square','path':'blue_square.jpg'},
                          {'color':'red','shape':'circle','path':'red_circle.jpg'},
                          {'color':'blue','shape':'circle','path':'blue_square.jpg'}]
        >>> pfa.package_image_sets(image_list,['shape'],['color'])
            [(OrderedDict([('color', 'red')]),
              OrderedDict([(('square',), 'red_square.jpg'),
                           (('circle',), 'red_circle.jpg')]),
              ('shape',),
              []),
              
             (OrderedDict([('color', 'blue')]),
              OrderedDict([(('square',), 'blue_square.jpg'),
                           (('circle',), 'blue_square.jpg')]),
              ('shape',),
              [])]
        
        Each of the top-level tuples is an image set. 
 
    '''
    
    sample_image = image_list[0]
    image_keys = list(sample_image.keys())
    if 'path' not in image_keys:
        raise Exception("Need to specify the image path!")
    image_keys = [key for key in image_keys if key!='path']
    if len(image_keys) != len(entry_vars)+len(grouping_vars):
        raise Exception("Make sure the image variables match the entry/grouping variables.")
    image_sets = []
    for image in image_list:
        
        # Check that the image set exists:
        
        image_set_dict = OrderedDict()
        for key in image:
            if key in grouping_vars:
                image_set_dict[key] = image[key]
        image_set_exists = False
        for i,image_set in enumerate(image_sets):
            if image_set[0]==image_set_dict:
                image_set_exists=True
                which_image_set = i
        if not image_set_exists:
            new_image_set = []
            new_image_set.append(image_set_dict)
            new_image_set.append(OrderedDict())
            new_image_set.append(tuple(entry_vars))
            new_image_set.append([])
            new_image_set = tuple(new_image_set)
            image_sets.append(new_image_set)
            which_image_set = -1
        image_key = []
        for key in image:
            if key in entry_vars:
                image_key.append(image[key])
        image_key = tuple(image_key)
        image_sets[which_image_set][1][image_key] = image['path']
    return(image_sets)

def get_image_set_rdm(image_set,
                      model_name,
                      models_dict,
                      out_fname,
                      rdm_name='rdm_default_name',
                      which_layers={},
                      dissim_metrics=['corr'],
                      kernel_sets=('all','all'),
                      num_perms=0,
                      append=True,
                      verbose=False,
                      debug_check=False):
    '''
    
    Internal helper function for the get_rdms function in order to enable parallel processing. 
    See get_rdms function for full documentation. 
    
    '''
    
    image_set_labels,images,entry_types,combination_list = image_set
    if append and os.path.exists(out_fname):
        append = True
        existing_df = ut.pickle_read(out_fname)
        subset_opts = [(var,image_set_labels[var]) for var in image_set_labels]
        existing_df = ut.df_subset(existing_df,[('model',model_name)])
        existing_df = ut.df_subset(existing_df,subset_opts)
           
    else:
        append = False
        
    print(f"{','.join(list(image_set_labels.values()))}")
    image_names = list(images.keys())
    combination_names = [comb[0] for comb in combination_list]
    entry_names = image_names + [comb[0] for comb in combination_list] # labels of the rows and columns

    # If you want to do different color spaces, fill this in someday. Fill in funcs that take in RGB and spit out desired color space. 
    color_space_funcs = {}
    color_space_list = []
          
    rdms_dict = defaultdict(lambda:[])
    perm_list = ['orig_data'] + list(range(1,num_perms+1))

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
        print(f"\n\t{layer},{dm}")
          
        activation_types = ['unit_level','feature_means']

        if dm in dissim_defs:
            dissim_func = dissim_defs[dm]
        elif type(dm)==tuple:
            dissim_func = dm[1]
            dm = dm[0]

        for activation_type,ks,which_perm in it.product(activation_types,kernel_sets,perm_list):
          
            print(f"\t\tKernel set {ks[0]}, activation {activation_type}, perm {which_perm}")
            ks_name,ks_dict = ks   
          
            perm_scramble = list(range(len(entry_names)))
            
            if which_perm != 'orig_data':
                random.shuffle(perm_scramble)
            
            perm_dict = {i:perm_scramble[i] for i in range(len(entry_names))}
          
            if (layer,activation_type) not in obj_activations[list(obj_activations.keys())[0]]:
                continue
          
            if append:
                subset_opts = [('model',model_name),('layer_label',layer),('activation_type',activation_type),('kernel_set_name',ks_name),('dissim_metric',dm),('perm',which_perm)]
                for label in image_set_labels:
                    subset_opts.append((label,image_set_labels[label]))
                df_subset = ut.df_subset(existing_df,subset_opts)
                if len(df_subset)>0:
                    print("\t\t\t Already done, skipping...")
                    continue
          
            sample_image = obj_activations[list(obj_activations.keys())[0]][layer,activation_type]
          
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
                    
                    ind1 = perm_dict[i1]
                    ind2 = perm_dict[i2]
                    ind1_im = entry_names[ind1]
                    ind2_im = entry_names[ind2]
          
                    pattern1 = obj_activations[ind1_im][layer,activation_type]
                    pattern2 = obj_activations[ind2_im][layer,activation_type]
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
            rdms_dict['entry_labels'].append(tuple([{entry_types[i]:entry[i] for i in range(len(entry))} for entry in entry_names]))
            rdms_dict['kernel_set_name'].append(ks_name) 
            rdms_dict['kernel_inds'].append(tuple(kernel_inds))
            rdms_dict['perm'].append(which_perm)
            rdms_dict['perm_scramble'].append(perm_scramble)
            rdms_dict['dissim_rankings'].append(tuple(dissim_rankings))
            for label in image_set_labels:
                rdms_dict[label].append(image_set_labels[label])
            if debug_check:
                print(pd.DataFrame.from_dict(rdms_dict))
                ipdb.set_trace()
                debug_check=False # Only check the first one, then continue. 
    del(obj_activations)
    rdms_df = pd.DataFrame.from_dict(rdms_dict)
    if append:
        rdms_df = pd.concat([rdms_df,existing_df])
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
             debug_check=False,
             append=True):
    
    '''
    
    This is the core function of this package: it takes in several (possibly many) image sets,
    creates an RDM for each one, and stores this stack of RDMs as a Pandas dataframe, which
    can then be easily fed into other functions to perform higher-order RSA (e.g., 
    to compare RDMs across layers and models), or to visualize the results. The basic idea
    is that your images can vary according to multiple variables; some of these will
    vary within an RDM (entry_vars), and some will vary between RDMs (grouping_vars). 
    The only work involved is preparing the inputs to this function; a minimal 
    example is shown at the end of this documentation, and also in the 
    accompanying readme.txt in the GitHub repo. 
    
    Args:
        image_sets: A list of image sets; an RDM will be computed separately for each one. 
            Each image_set is a tuple of the format: 
            (
                Dictionary of grouping variables for that image set 
                Dictionary of images for that image set; each key is the value of an entry_var, each value is the
                    path for that image
                Tuple of the names of the entry variables
                Blank list (which can specify ways of combining activation patterns from images if desired)
            )
        models_dict: a dictionary of models to use, where each key is your name for the model, 
            and each value is the model itself.
        out_fname: the file path to save the dataframe at, stored in Pickle format (so .p suffix)
        rdm_name: the name to give to the RDM
        which_layers: a dictionary specifying which layers from each model to use
            (e.g., {'alexnet':['conv1','fc3']}); if a model is not specified, all layers are assumed 
        dissim_metrics: a list of which dissimilarity metrics to use for the RDMs. Put 'corr' for 1-correlation; 
            put 'euclidean_distance' for euclidean distance. If you wish to use a different one,
            have that entry in the list be of format (dissim_func_name,dissim_func), such 
            as ('spearman',stats.spearmanr)
        kernel_sets: If you wish to only compute the RDMs over a subset of the kernels, 
            specify it here. This will be a list of tuples. Each tuple is of format
            (kernel_set_name,kernel_set_dict). Kernel_set_name is the name of the kernel_set.
            kerkel_set_dict is a dictionary specifying which kernels to select from each
            layer; each key is of format (model,layer_name), and each value is
            the list of kernel indices to use (so, {('alexnet','conv1'):[1,5,7]}).
            By default, all kernels are used and this can be ignored. 
        num_perms:
            If you wish to do a permutation test where you shuffle the labels of your RDM entries multiple times,
            specify how many permutations to do here. 
        num_cores:
            If you wish to use parallel processing, specify how many cores here; put 'na'
            if you don't want to process in parallel
        verbose: 
            If you want it to give you the progress of each RDM as it's computed, put True,
            else put False. 
        debug_check:
            Put True if you want to stop and debug with ipdb after each RDM is computed,
            put False otherwise.
        append:
            If you want to append the output dataframe to an existing dataframe, put True
            and it'll append the results to the dataframe at that filename (and skip
            any entries that are already in that dataframe). Else, that 
            dataframe will be overwritten.
    
    Returns:
        Dataframe where each row is an RDM along with accompanying metadata. The columns of this 
        dataframe are: 
            matrix: The actual RDM, as a 2D numpy aray. 
            df_name: The name of the dataframe. Purely cosmetic
            model: The name of the model used.
            layer: The layer name (e.g., conv1)
            layer_num: Where the layer falls in the network (e.g., conv1 is 1)
            layer_label: The layer name along with where it is in the network (e.g., (conv1,1) )
            entry_keys: tuple of tuples where each tuple indicates the values of the entry variables
                for the items in the RDM. For example, if the entries of the RDM are 
                square, circle, and triangle, this would be (('square'),('circle'),('triangle')).
            entry_labels: Same as entry keys, but it indicates the name of the entry variables.
                For example, ({'shape':'square'},{'shape':'circle'},{'shape':'triangle'})
            dissim_metric: the dissim metric used in that RDM (e.g., euclidean_distance)
            activation_type: Whether the RDM was computed by averaging across space 
                for the convolutional layers ('feature_means'), versus simply
                vectorizing the spatial dimension ('unit_level'). By default
                the function does both.
            kernel_set_name: The name of the kernel set being used. 
            kernel_set_inds: The indices of the kernels that are used. 
            perm: 'orig_data' if it's not permuted, or the permutation number if you specified
                permuted data. 
            dissim_rankings: The ranking of the entries that would yield a sub-matrix
                with the highest dissimilarity. For example, (3,5,6,2) means that 
                a submatrix with entries 3 and 5 from the matrix would yield the highest
                dissimilarity for two elements, a submatrix with 3,5,6 would yield
                the highest dissimilarity for three elements, and so on. 
            Additionally, there will be columns specifying the labels for each 
                RDM, which will be whatever you specified in the input image sets. 
    
    
    Examples: 
        >>> image_list = [{'color':'red','shape':'square','path':'red_square.jpg'},
                          {'color':'blue','shape':'square','path':'blue_square.jpg'},
                          {'color':'red','shape':'circle','path':'red_circle.jpg'},
                          {'color':'blue','shape':'circle','path':'blue_square.jpg'}]
        >>> image_sets = package_image_sets(image_list,['shape'],['color']) 
        >>> models_prepped = prepare_models({'alexnet':('alexnet','trained'),
                                             'vgg19':('vgg19','random')})
        >>> get_rdms(image_sets = image_sets, # the stack of image sets
                     models_dict = models_prepped, # the desired models
                     which_layers = {'alexnet':['conv1','fc3']} # which layers to use; if not specified uses all layers
                     out_fname = os.path.join(rdm_df.p), # where to save the output
                     rdm_name = 'my_rdms', # the name of the output RDM
                     dissim_metrics = ['corr','euclidean_distance'], # the dissimilarity metrics you want to try
                     num_cores = 2 # How many cores to use for parallel processing)
        
    '''
    
    print("*****Computing RDMs*****")          
          
    if num_cores=='na':
        rdm_df_list = []
        for image_set,model_name in it.product(image_sets,models_dict):   
            new_df = get_image_set_rdm(image_set,model_name,models_dict,out_fname,rdm_name,which_layers,
                                       dissim_metrics,kernel_sets,num_perms,append,verbose,debug_check)
            rdm_df_list.append(new_df)
    else:
        #rdm_df_list = Parallel(n_jobs=num_cores)(delayed(get_image_set_rdm)(image_set,model_name,models,
                                                                            #rdm_name,which_layer_types,
                                       #dissim_metrics,kernel_sets,num_perms,debug_check=False) for image_set,model_name in it.product(image_sets,models))
        pool = multiprocessing.Pool(processes=num_cores)
        rdm_df_list = [pool.apply_async(get_image_set_rdm,args=(image_set,model_name,models_dict,out_fname,rdm_name,which_layers,
                                       dissim_metrics,kernel_sets,num_perms,append,verbose,debug_check)) for image_set,model_name in it.product(image_sets,models_dict)]
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
    
    This is the other core function of this package: it takes in a Pandas dataframe with a stack of 
    RDMs, and constructs a desired "meta-rdm" (i.e., an RDM that itself is formed by correlating 
    other RDMs). Need to specify the "entry variables" (the variables that vary WITHIN each meta-rdm)
    and the "grouping variables" (the variables that vary ACROSS meta-rdms). This allows you to easily
    compute higher-order similarities among layers, models, groups of images, similarity metrics--
    anything you want!
    
    Args:
        rdm_df: The input dataframe containing a stack of rdms; the output of the get_rdms function.
        out_fname: Where to store the resulting dataframe with the meta-rdms (as .p file)
        entry_variables: The variables that vary across the entries within each output meta-rdm
        entry_var_subsets: If you only want to make an RDM with a subset of the input RDM,
            specify which subsets you want to use. This will be a list of tuples, where 
            each tuple is of format (subset_name,entry_keys); subset_name is the name
            of the subset, entry_vars is a list of the entry keys you wish to use.
            For example, if you only want to include the RDMs for red and blue things,
            this would be [('subset1',(('red'),('blue')))]. By default it just does all entries. 
        grouping_variables: The variables that vary ACROSS meta-rdms.
        df_name: The name you wish to give to the resulting dataframe (just cosmetic)
        num_perms: if you wish to do a permutation test where you scramble the entries of each RDM,
            put how many such scramblings you wish to do here.
        dissim_metrics: a list of which dissimilarity metrics to use for the RDMs. Put 'corr' for 1-correlation; 
            put 'euclidean_distance' for euclidean distance. If you wish to use a different one,
            have that entry in the list be of format (dissim_func_name,dissim_func), such 
            as ('spearman',stats.spearmanr)
        entry_keys: tuple of tuples where each tuple indicates the values of the entry variables
            for the items in the RDM. For example, if the entries of the RDM are 
            square, circle, and triangle, this would be (('square'),('circle'),('triangle')).
        entry_labels: Same as entry keys, but it indicates the name of the entry variables.
            For example, ({'shape':'square'},{'shape':'circle'},{'shape':'triangle'})
        Some other columns will be the specified grouping variables (which vary across RDMs).
        Columns with the suffix "_sub" specify the values of the "sub-RDMs" that are the entries
            to the meta-rdms; this allows you to see all "layers" of a higher-order RDM
            at a glance. 
        
    
    Returns:
        Dataframe containing the stack of meta-rdms with the following columns:
            matrix: the RDM
            df_name: Name of the dataframe
            activation_type: Whether the RDM was computed by averaging across space 
                for the convolutional layers ('feature_means'), versus simply
                vectorizing the spatial dimension ('unit_level'). By default
                the function does both.
            entry_var_subset: The name of the subset of entries, if you specified one.
    
    Examples:
        >>> meta_rdm = get_meta_rdms(rdm_df,
                                     opj(project_dir,'analysis','meta_rdm_demo.p'),
                                     dissim_metrics = ['corr','euclidean_distance'],
                                     entry_variables=['color'],
                                     grouping_variables=['model','layer','activation_type','dissim_metric'],
                                     df_name='my_meta_rdm')
            
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
    # (Since both the meta-rdm and the sub-rdm have their own dissim metrics, futations, etc.)
              
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
              
        print(f"Getting meta-rdm for {','.join([str(g) for g in group_labels])}")
                      
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
                #for col in rdm_group_subset.columns.values:
                #    if ('_sub_sub') in col:
                #        meta_rdm_dict[col].append(rdm_group_subset[col].iloc[0])
                meta_rdm_dict['entry_keys_sub'].append(rdm_group_subset['entry_keys'].iloc[0])
                meta_rdm_dict['entry_labels_sub'].append(rdm_group_subset['entry_labels'].iloc[0])
                for i,label in enumerate(group_labels):
                    meta_rdm_dict[grouping_variables[i]].append(label)
            
    meta_rdm_df = pd.DataFrame.from_dict(meta_rdm_dict)
    meta_rdm_df.sort_index(axis=1, inplace=True)
    pickle.dump(meta_rdm_df,open(out_fname,"wb"))
    return meta_rdm_df   
              
###################################################
############# OPERATIONS ON RDMS ##################
###################################################

def subset_rdm_df(df,entry_keys):
    '''
    
    If you have a Pandas dataframe with a stack of RDMs (the output of get_rdms or get_meta_rdms), 
    this function allows you to extract a subset of those RDMs (i.e., if you want to only
    look at some, but not all, cells of those RDMs).
    
    Args:
        df: The dataframe with a stack of RDMs (the output of get_rdms or get_meta_rdms)
        entry_keys: List of the keys for the RDM entries to keep.
        
    Returns:
        Dataframe with stack of RDMs subsetted based on the provided entry_keys. 
        
    Example:
        >>> subset_rdm_df(df,
                          [('red'),('blue'),('green')])
    
    '''
    new_matrices_list = []
    new_keys_list = []
    new_labels_list = []
              
    for r in range(len(df)):
        row = df.iloc[r,:]
        row_keys = row['entry_keys']
        if 'entry_labels' in df.columns.values:
            row_labels = row['entry_labels']
        matrix = row['matrix']
        which_inds = []
        new_keys = []
        new_labels = []
        for k,key in enumerate(row_keys):
            if key in entry_keys:
                which_inds.append(k)
                new_keys.append(key)
                if 'entry_labels' in df.columns.values:
                    new_labels.append(row_labels[k])
        matrix_subset = matrix[np.ix_(which_inds,which_inds)]
        new_matrices_list.append(matrix_subset)
        new_keys_list.append(new_keys)
        if 'entry_labels' in df.columns.values:
            new_labels_list.append(new_labels)
    df['matrix'] = new_matrices_list
    df['entry_keys'] = new_keys_list    
    if 'entry_labels' in df.columns.values:
        df['entry_labels'] = new_labels_list
    return df
              
def my_mds(rdm_df,
           dims=2,
           plot_opts={}, 
           other_dot_opts={},
           plot_limits = (-.1,1.1),
           cond_labels={},
           label_offset=.02,
           label_opts={},
           plot_dots=True,
           cond_colors={},
           cond_pics={},
           pic_size=.1,
           alphatize_pics=True,
           add_pic_border=False,
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
    
    This is another core function of the library: it'll create an MDS plot for an RDM. 
    You must give it a single row from a dataframe containing an RDM (the output of
    the get_rdms or get_meta_rdms function). This gives the function 
    lots of helpful meta-data to make it easier to specify plotting options, 
    because then you can specify based on the name of your RDM entries 
    (which should be meaningful) how to plot each entry on the MDS plot. 
    Generally, then, things will be specified as dictionaries, where each
    key is the entry, and each value is the plotting option for that entry 
    (for example, the size/color of the dot, the path to the picture to use, etc.)
    
    Args:
        rdm_df: One row from a dataframe from get_rdms or get_meta_rdms,
            containing an RDM and the accompanying meta-data.
        dims: How many dimensions to use in the MDS plot. Either 2 or 3; 
            if 3, you can't plot pictures yet. 
        plot_opts: Dictionary of plot options for the matplotlib call,
            that you with to be uniform for ALL entries; so, if you want to set 
            all dots to a given size, set {'s':5}). 
        other_dot_opts: this is if you want to set options for each SPECIFIC
            dot (e.g., change the color or size of certain dots). The way it works is
            you give it a dictionary, where each key is the option to change
            in Matplotlib (e.g., "s" for size), and each value is a dictionary
            specifying the value of that option for each entry; in this latter
            dictionary, each key is the entry key for an entry of the RDM,
            and each value is the desired value of the option for that entry.
            For instance, {'s':{('cat'):2,('dog'):3}} sets the size 
            of the dot for the cat entry to 2, and the size of the dot for the dog entry to 3.
        plot_limits: The desired limits of the plot, as (min,max) tuple.
            Remember that output of MDS function ranges from 0 to 1, so this 
            determines how big you want margins to be.
        cond_labels: if you want a text label annotating some or all of the dots,
            put a dictionary here, where each key is the entry key for the entry you want to annotate,
            and the value is the text you want to annotate with.
        label_offset: how far away from the dot to put the label
        label_opts: dictionary with any optional keyword arguments for making the labels, e.g. the font size or style
        plot_dots: True if you want to plot the dots, False if not (e.g., if you're plotting pictures instead)
        cond_colors: dictionary specifying what color to make the dots, where each key 
            is the entry key for a given item, and each value is the color you want. 
        cond_pics: If you want to plot pictures on the MDS plot, a dictionary where each key
            is the entry key for the item to add a picture to, and each value is either
            a PIL image or a path to an image.
        pic_size: How big to make each picture.
        alphatize_pics: Whether to set the white parts of any pictures transparent.
        alpha_target: which RGB value to set transparent if alphatize_pics is True.
        alpha_range: What range of RGB values around alpha_target to set transparent
            (e.g., if parts of the image are off-white)
        pic_offset: (x,y) tuple indicating where to put the picture relative to 
            the MDS point (e.g., if you want it a little bit above a dot)
        line_trajectories: if you want to have lines connecting the points 
            (for example, if each point is a layer), this is specified here.
            You have a list, where each element specifies a given trajectory.
            Each element of this list is a tuple (trajectory_order,opt_dict)
            where trajectory order is a list of the entry keys of the items
            you want to connect with a line, in the order you want,
            and opt_dict is any keyword options for drawing the line
            (specifying thickness, color, etc.)
        show_stress: whether to indicate the stress of the MDS solution on the plot
        plot_style: which matplotlib plot style to use (default classic)
        fig_title: The title of the figure on teh plot
        fig_size: The size of the plot.
        show: whether to show the plot (as opposed to just saving it)
        save_opts: the options to put into the matplotlib call to savefig; 
            if you want to save, this should at least include the filename and file type
            (e.g., {'fname':'my_mds.pdf','format':'PDF'})
        
    Returns:
        Shows and/or saves the resulting MDS plot.

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
            if add_pic_border:
                pic = ImageOps.expand(pic,border=5,fill='black')
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

def compute_tolerances(input_df):
    '''
    
    Can generally ignore; this is explained in an upcoming paper. 
    This function takes in a dataframe with a stack of RDMs, which must have two kinds of 
    entry variables (i.e., two variables must vary across the entries of the RDM). 
    Then, it computes how "dominant" a given variable is in the joint 
    coding of the two variables. 
    
    TI = ( d.d-var1/s-var2 - d.s-var1/d-var2) / (d.d-var1/s-var2 + d.s-var1/d-var2)
    
    A value of 1 means that var1 completely dominates, a value of -1 means that var2
    completely dominantes, and a value of 0 means they are equally dominant.
    This value is added to each row (RDM) of the dataframe.
    
    Args:
        input_df: The dataframe with the RDMs to compute the dominance values for. 

    Returns:
        Same dataframe, but with a column added indicating the dominance index. 

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
              
###################################################
############ ANALYSES BESIDES RSA #################
###################################################

def run_2way_svm_analysis(image_list,var1,var2,grouping_vars,models_dict,
                     which_layers,analysis_name,save_path):
    '''
    
    This will run an SVM decoding analysis on a set of images. 
    You give it a stack of images, along with some variables describing each image, and 
    it runs the decoding analysis using leave-one-out cross-validation.
    It assumes a two-way crossed design, where you want to performing decoding
    of one variable collapsing across values of the other, and vice versa. 
    
    Args:
        image_list: list of dictionaries, one for each image. Each dict must contain 
            a key "path" whose value is the path to that image. The other keys
            are variables associated with that image. 
        var1: The first variable to decode across
        var2: The second variable to decode across 
        grouping_vars: The variables to group the analysis by; that is, you separately do the SVM 
            analysis WITHIN each possible set of values of the grouping variables.
        models_dict: Dictionary with the models to use, where each key is the name of a model,
            and each value is the model itself.
        which_layers: The layers to use from each model; a dictionary where each key is a model
            name, and each value is a list of the layers to use (e.g,. {'alexnet':['conv1','fc2']})
        save_path:
            Where to save the output dataframe with the results.
            
    Returns:
        Dataframe with the decoding accuracies 

    '''
    
    clf = svm.SVC(kernel='linear',C=1)           
              
    # Get the variable values.                   
              
    grouping_var_dict = defaultdict(lambda:[])
    var1_vals = []
    var2_vals = []
    for image in image_list:
        if image[var1] not in var1_vals:
            var1_vals.append(image[var1])
        if image[var2] not in var2_vals:
            var2_vals.append(image[var2])
        for gvar in grouping_vars:
            if image[gvar] not in grouping_var_dict[gvar]:
                   grouping_var_dict[gvar].append(image[gvar])
    
    # Now, go through each combination of grouping variables, and do the SVM analysis.
    for model_name in models_dict:
        print(f"Running {model_name}...")
        svm_dict = defaultdict(lambda:[])
        model = models_dict[model_name]
        for gvar_vals in it.product(*grouping_var_dict.values()):
            print(f"\tGroup {','.join(gvar_vals)}, getting image activations...")
            # Now, get the image activations for this combination of grouping variables.

            activations_dict = defaultdict(lambda:[])
            for image in image_list:
                image_in_group = True
                for gv in range(len(gvar_vals)):
                    if image[grouping_vars[gv]] != gvar_vals[gv]:
                        image_in_group = False
                if not image_in_group:
                    continue
                image_activations = get_model_activations_for_image(image['path'],model)
                if verbose:
                    path = image['path']
                    print(f"Loading {path}...")
                for layer in image_activations:
                    if layer not in which_layers[model_name]:
                        continue
                    activations_dict['model'].append(model_name)
                    activations_dict['layer'].append(layer)
                    activations_dict[var1].append(image[var1])
                    activations_dict[var2].append(image[var2])
                    for gv in range(len(gvar_vals)):
                        activations_dict[grouping_vars[gv]].append(gvar_vals[gv])
                    activations_dict['activations'].append(image_activations[layer].flatten())
            
            activations_df = pd.DataFrame.from_dict(activations_dict)
            print(f"\tRunning decoding, regular decoding...")  
            
            # First, do the regular decoding.
            
            for decode_var in [var1,var2]:
                other_variable = [var for var in [var1,var2] if var!=decode_var][0]
                print(f"\t\tDecoding {decode_var}, collapsing across {other_variable}...")
                decode_var_vals = activations_df[decode_var].unique()
                num_conds = len(decode_var_vals)
                for layer in activations_df['layer'].unique():
                    print(f"\t\t\tLayer {layer[0]}")
                    for v1,v2 in it.product(range(num_conds),range(num_conds)):
                        if v1>=v2:
                            continue
                        bin1_name = decode_var_vals[v1]
                        bin2_name = decode_var_vals[v2]
                        bin1_vals = list(activations_df.loc[(activations_df[decode_var]==bin1_name) &
                                                            (activations_df['layer']==layer),'activations'])
                        bin2_vals = list(activations_df.loc[(activations_df[decode_var]==bin2_name) &
                                                            (activations_df['layer']==layer),'activations'])
                        items_per_bin = len(bin1_vals)
                        all_inds = list(range(items_per_bin))
                        decoding_accs = []
                        for test_ind in range(items_per_bin):
                            bin1_test = bin1_vals[test_ind]
                            bin2_test = bin2_vals[test_ind]
                            test_vals = np.array([bin1_test,bin2_test])
                            test_labels = np.array([1,2])

                            train_inds = [ind for ind in all_inds if ind!=test_ind]
                            bin1_train = np.array([bin1_vals[ind] for ind in train_inds])
                            bin2_train = np.array([bin2_vals[ind] for ind in train_inds])
                            train_vals = np.vstack([bin1_train,bin2_train])
                            train_labels = [1]*(items_per_bin-1) + [2]*(items_per_bin-1)

                            clf.fit(train_vals,train_labels)
                            label_prediction = clf.predict(test_vals)
                            chunk_acc = list((test_labels==label_prediction).astype(int))
                            decoding_accs.extend(chunk_acc)
                        mean_acc = np.mean(decoding_accs)
                        svm_dict['analysis_name'].append(analysis_name)
                        svm_dict['model'].append(model_name)
                        svm_dict['layer_label'].append(layer)
                        svm_dict['layer'].append(layer[0])
                        svm_dict['layer_num'].append(layer[1])
                        svm_dict['decoded_variable'].append(decode_var)
                        svm_dict['other_variable'].append(other_variable)
                        for gv in range(len(gvar_vals)):
                            svm_dict[grouping_vars[gv]].append(gvar_vals[gv])
                        svm_dict['decoding_type'].append('within')
                        svm_dict['bin1'].append(bin1_name)
                        svm_dict['bin2'].append(bin2_name)
                        svm_dict['acc'].append(mean_acc)
                  
        # Add cross-decoding here later. 
        svm_df = pd.DataFrame.from_dict(svm_dict)
        create_or_append_csv(svm_df,save_path)
    return svm_df                 
                  
def compute_etas_for_model_and_imageset_fast(image_set,model_desc,which_layer_dict):
    '''
    
    This will run a two-way ANOVA analysis for a model and set of images,
    giving the eta-squared (variance explained effect size) for two factors
    and their interaction. The nice thing is that the method doesn't require
    saving the model activations for every single stimulus, which becomes
    very intractable very quickly. 
    
    Args:
        image_set: tuple consisting of (image_set_labels,stim_dict,entry_types).
            image_set_labels is a dictionary describing the image set that'll be used in the output
            dataframe, stim_dict is a dictionary of the stimuli where each key is the value of the two IVs,
            and entry_types gives the variable names of the two IVs. 
        model_desc: tuple of format (model_name,base_model,weight_opts), 
            where model_name is your internal name for the model, base_model is the model architecture
            (e.g., AlexNet), and weight_opts is either 'trained','random', or a 
            link to the state_dict for a set of weights.
        which_layer_dict: dictionary specifying which layers to analyze for each model,
            where each key is a model, and each value is a list of layers in format ('conv3',5)
            (tuple specifying e.g. 3rd Conv layer and 5th layer overall). This is REQUIRED
            for this function to save space. 
            
    Returns:
        Dataframe indicating the average and standard deviation of the eta-squared values
        in each layer. 
        
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
    sample_activations = get_model_activations_for_image(stim_dict[(iv1_vals[0],iv2_vals[0])],model)
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
        stim_activations = get_model_activations_for_image(stim_path,model)
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
