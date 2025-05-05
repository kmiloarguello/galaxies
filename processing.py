import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os 
import glob
import astroalign as aa
from astropy.io import fits
from astropy.stats import sigma_clip

import astropy
from astropy.modeling.functional_models import Sersic1D
from astropy.modeling import models, fitting



def get_best_sigma_for_master_bias(list_bias, sigma_range=None, plot_results=True):
    """
    Détermine le meilleur sigma pour le filtrage des poses de bias.
    
    Args:
        list_bias (list): Liste des chemins vers les fichiers bias.
        sigma_range (list, optional): Liste des valeurs de sigma à tester. Par défaut [0, 0.5, 1, 2, 3, 5].
        plot_results (bool, optional): Si True, affiche un graphique des résultats. Par défaut True.
        
    Returns:
        tuple: (meilleur_sigma, master_bias_list, résultats)
    """
    # Valeurs de sigma à tester
    if sigma_range is None:
        sigma_range = [0.5, 1, 2, 3, 5]
    
    # Chargement des données (une seule fois)
    print(f'Chargement de {len(list_bias)} poses de bias...')
    
    # Ouvrir le premier fichier pour obtenir la forme
    shape_bias = np.shape(fits.getdata(list_bias[0]))
    n_bias = len(list_bias)
    
    # Initialisation du tableau pour stocker toutes les poses de bias
    master_bias_list = np.zeros([n_bias, shape_bias[0], shape_bias[1]])
    
    # Chargement des données en une seule passe
    for i, bias_file in enumerate(list_bias):
        master_bias_list[i, :, :] = fits.getdata(bias_file)
    
    # Calcul de la médiane sans sigma clipping (une seule fois)
    master_bias_no_sigma = np.nanmedian(master_bias_list, axis=0)
    
    # Stockage des résultats
    results = []
    
    # Test des différentes valeurs de sigma
    for sigma in sigma_range:        
        masked_data = sigma_clip(master_bias_list, sigma=sigma, maxiters=5, axis=0, masked=True)
        master_bias = np.nanmedian(masked_data, axis=0)
        
        # Calcul des différences
        max_diff, mean_diff, percentage_changed = get_master_bias_difference_sigma(
            master_bias, master_bias_no_sigma, masked_data)
        
        # Stockage des résultats
        results.append({
            'sigma': sigma,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'percentage_changed': percentage_changed,
            'master_bias': master_bias
        })
    
    # Détermination du meilleur sigma (basé sur un compromis entre le pourcentage de pixels modifiés et la différence moyenne)
    best_result = min(results[1:], key=lambda x: abs(x['percentage_changed'] - 1.0))
    best_sigma = best_result['sigma']
    
    print(f"\nMeilleur sigma: {best_sigma}")
    
    # Visualisation des résultats si demandé
    if plot_results:
        plot_sigma_comparison(results)
    
    return best_sigma, master_bias_list, results

def plot_sigma_comparison(results):
    """
    Crée un graphique comparant l'effet des différentes valeurs de sigma.
    
    Args:
        results (list): Liste des résultats pour chaque valeur de sigma.
    """
    sigma_values = [result['sigma'] for result in results]
    max_diffs = [result['max_diff'] for result in results]
    mean_diffs = [result['mean_diff'] for result in results]
    percentages = [result['percentage_changed'] for result in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique des différences
    ax1.plot(sigma_values, max_diffs, 'o-', label='Différence maximale (ADU)')
    ax1.plot(sigma_values, mean_diffs, 's-', label='Différence moyenne (ADU)')
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('Différence (ADU)')
    ax1.set_title('Différences en fonction de sigma')
    ax1.grid(True)
    ax1.legend()
    
    # Graphique du pourcentage de pixels modifiés
    ax2.plot(sigma_values, percentages, 'o-', color='green')
    ax2.set_xlabel('Sigma')
    ax2.set_ylabel('Pixels modifiés (%)')
    ax2.set_title('Pourcentage de pixels modifiés en fonction de sigma')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def master_bias(list_bias, out_name ='', out_dir = '', overwrite = 1, sigma=5, out_master_bias_no_sigma = ''):
    #on ouvre le premier fichier de list_bias pour obtenir la forme de chaque master bias
    shape_bias =  np.shape(fits.getdata(list_bias[0]))
    n_bias = len(list_bias)
    print(f'il y a {n_bias} poses de bias dans la liste')
    # on initialise un tableau numpy à zeros qui va contenir l'ensemble des poses de bias individuelles
    master_bias_list = np.zeros([n_bias,shape_bias[0],shape_bias[1]]) 
    for b,i in zip(list_bias,range(n_bias)):
        master_bias_list[i,:,:]= fits.getdata(b)
    # le master bias contient, en chaque pixel, la mediane de la valeur de ce pixel dans chacune des poses de bias 
    master_bias_no_sigma = np.nanmedian(master_bias_list,axis=0)
    masked_data = sigma_clip(master_bias_list, sigma=sigma,maxiters=5,axis=0,masked=True)
    master_bias = np.nanmedian(masked_data, axis=0)

    # Plot des différences avec et sans sigma_clip
    # plot_master_bias_difference_sigma(master_bias, master_bias_no_sigma, masked_data)
    #sigma_list = np.array([0, 1, 2, 3, 4, 5])
    # get_master_bias_difference_sigma(master_bias, master_bias_no_sigma, masked_data)

    if len(out_name)>0:
        ## on enregistre le master bias dans un nouveau fichier fits
        hdu = fits.PrimaryHDU(master_bias)
        hdu.writeto(out_dir + out_name, overwrite = overwrite)
        print(f'le master bias a été sauvé en {out_dir + out_name}')
    if (len(out_master_bias_no_sigma)>0):
        ## on enregistre le master bias dans un nouveau fichier fits
        hdu = fits.PrimaryHDU(master_bias_no_sigma)
        hdu.writeto(out_dir + out_master_bias_no_sigma, overwrite = overwrite)
        print(f'le master bias sans sigma a été sauvé en {out_master_bias_no_sigma}')
    return master_bias, out_dir + out_name, master_bias_no_sigma

def get_master_bias_difference_sigma(master_bias, master_bias_no_sigma, masked_data):
    """
    Calcule la différence entre le master bias avec et sans sigma clipping.
    
    Args:
        master_bias (ndarray): Master bias avec sigma clipping.
        master_bias_no_sigma (ndarray): Master bias sans sigma clipping.
        masked_data (ndarray): Données masquées après sigma clipping.
    
    Returns:
        tuple: Différence maximale, moyenne et pourcentage de pixels modifiés.
    """
    difference = master_bias - master_bias_no_sigma
    max_diff = np.max(np.abs(difference))
    mean_diff = np.mean(np.abs(difference))
    percentage_changed = np.sum(masked_data.mask.any(axis=0)) / (difference.shape[0] * difference.shape[1]) * 100

    return max_diff, mean_diff, percentage_changed


def master_flat(list_flat, master_bias_name, out_name ='', out_dir = '', overwrite = 1):
    #on ouvre le premier 
    tmp = fits.getdata(list_flat[0])
    # on initialise le master flat 
    master_flat = np.zeros(np.shape(tmp)) 
    # ouvrir le master bias
    mb = fits.getdata(master_bias_name)
    for b in list_flat:
        exp = fits.open(b)[0].header['EXPOSURE']
        #tmp = (fits.getdata(b)-mb)
        tmp = np.flip(fits.getdata(b).T,axis=0)-mb
        master_flat += tmp/np.median(tmp)/len(list_flat)
    ## on normalise le resultat
    master_flat /= np.median(master_flat)
    if len(out_name)>0:
        ## on enregistre le master bias dans un nouveau fichier fits
        hdu = fits.PrimaryHDU(master_flat)
        hdu.writeto(out_dir + out_name, overwrite = overwrite)
    return master_flat


def stacking(list_files, master_bias_name, master_flat_name, out_name = '', out_dir = '', overwrite = 1):
    mb = fits.getdata(master_bias_name)
    mf = fits.getdata(master_flat_name)
    tmp = fits.getdata(list_files[0])
    image_concat = np.zeros([len(list_files),np.shape(tmp)[0],np.shape(tmp)[1]])
    for image,i in zip(list_files,range(len(list_files))):
        if (i==1):
            source = (fits.getdata(image)-mb)/mf # calibration
            source /= np.median(source) #renormalization
            #source =  np.array(fits.getdata(image), dtype="<f4")
            image_concat[i,:,:]=(source)#/mf
        elif (i>1):
            #registered_image, footprint = aa.register(np.array(fits.getdata(image), dtype="<f4"),source)   
            image = (fits.getdata(image)-mb)/mf
            image /= np.median(image)
            registered_image, footprint = aa.register(image,source, min_area=12)
            image_concat[i,:,:] = (registered_image)
    filtered_data = sigma_clip(image_concat, sigma=2, maxiters=10,axis=0,masked=False) #image_concat#

    final_image = np.nanmean(filtered_data, axis=0)
    final_image=(final_image).astype(np.float32)
    hdu = fits.PrimaryHDU(final_image)
    hdu.writeto(out_dir + out_name, overwrite=True)
    return final_image,out_dir + out_name

