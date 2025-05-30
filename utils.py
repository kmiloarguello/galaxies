import os
import warnings
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import processing as  pr
from astroquery.vizier import Vizier
import pandas as pd
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.centroids import centroid_quadratic

# --------------------
# HELPERS
# --------------------

# Gets the pixel size in arcseconds / pixel
def get_pixel_size(XPIXSZ, FOCALLEN):
    """Obtenir pixel size en arcsec/pixel."""
    return rad2arcsec(1) * (XPIXSZ / FOCALLEN)

# Gets the physical size in parsec and kpc
def get_physical_size(angular_size_arcsec, distance_mpc):
    """Calculer la taille physique en parsec et kpc à partir de la taille angulaire et de la distance."""
    angular_size_rad = angular_size_arcsec / rad2arcsec(1) # Conversion d'arcsec à radian
    distance_pc = distance_mpc * 1e6 # Conversion de Mpc à pc
    physical_size_pc = angular_size_rad * distance_pc
    physical_size_kpc = physical_size_pc / 1e3 # Conversion de pc à kpc
    return physical_size_pc, physical_size_kpc

def get_pixel_size_in_kpc(pixel_size, distance_mpc):
    distance_pc = distance_mpc * 1e6
    return pixel_size * distance_pc / rad2arcsec(1) / 1e3

def plot_initial_filter_sequence(galaxy_name = 'M83',
                                 pathname = './M83/',
                                  filter_sequence = ['Red', 'Blue', 'Green', 'HA'],
                                  image_index = 0):
    """Tracer les images initiales de chaque filtre."""

    red_files = sorted(glob.glob(f'./{pathname}/{galaxy_name}-Red-*.fit'))
    blue_files = sorted(glob.glob(f'./{pathname}/{galaxy_name}-Blue-*.fit'))
    green_files = sorted(glob.glob(f'./{pathname}/{galaxy_name}-Green-*.fit'))

    initial_files_to_plot = []

    # Add Red file if available
    if 'Red' in filter_sequence and red_files and image_index < len(red_files):
        initial_files_to_plot.append(red_files[image_index])
    
    # Add Blue file if available
    if 'Blue' in filter_sequence and blue_files and image_index < len(blue_files):
        initial_files_to_plot.append(blue_files[image_index])
    
    # Add Green file if available
    if 'Green' in filter_sequence and green_files and image_index < len(green_files):
        initial_files_to_plot.append(green_files[image_index])
    
    # Add HA file if available and requested
    if 'HA' in filter_sequence:
        ha_files = sorted(glob.glob(f'./{pathname}/{galaxy_name}-HA-*.fit'))
        if ha_files and image_index < len(ha_files):
            print('ha')
            initial_files_to_plot.append(ha_files[image_index])
    
    files_to_plot = initial_files_to_plot  # All files are already valid

    if len(files_to_plot) == 0:
        print("Aucune image à afficher.")
        return

    fig, axes = plt.subplots(2, len(files_to_plot), figsize=(15, 5), gridspec_kw={'height_ratios': [2, 1]})

    for ax_img, ax_hist, file in zip(axes[0], axes[1], files_to_plot):
        with fits.open(file) as hdulist:
            data = hdulist[0].data

        im = ax_img.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data,5) , vmax=np.percentile(data,99))
        ax_img.set_title(f'{file.split("/")[-1]}', fontsize=10)
        ax_img.axis()
        ax_img.grid(False)
        ax_img.axis('off')
        ax_img.set_anchor('C')
        cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        if ax_img == axes[0][-1]:
            cbar.set_label('ADU', fontsize=10)

        # Histogramme
        ax_hist.hist(data.flatten(), bins=1000, color='blue')
        ax_hist.loglog()
        ax_hist.grid(True)
        ax_hist.set_xlabel('ADU', fontsize=10)
        if ax_hist == axes[1][0]:
            ax_hist.set_ylabel('Nombre de pixels', fontsize=10)
    plt.tight_layout()
    plt.show()

def get_exposure_times(filenames):
    for filename in filenames:
        exptime = get_exposure_time(filename['pathname'])
        print(f"Temps de pose pour le filtre {filename['filter_name']} : {exptime} s")

def get_exposure_time(filename):
    """Obtenir le temps d'exposition à partir du nom de fichier."""
    hdul = fits.open(filename)
    offset_data_bias = hdul[0].data
    hdul.close()

    if offset_data_bias is None:
        print("Aucune image à afficher.")
        return
    
    header = hdul[0].header
    exptime = header.get('EXPTIME', None)
    if exptime is None:
        print("Le temps d'exposition n'est pas trouvé dans l'en-tête de l'image.")
        return None
    return exptime

def plot_bias_hist(filename, galaxie_name, bins=1000):
    hdul = fits.open(filename)
    offset_data_bias = hdul[0].data
    hdul.close()

    if offset_data_bias is None:
        print("Aucune image à afficher.")
        return
    
    hist_values, bin_edges = np.histogram(offset_data_bias.flatten(), bins=bins)
    peak_bin_index = np.argmax(hist_values) # Index of the peak bin
    peak_x = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2 # X value of the peak bin

    fig, (ax_im, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))

    im = ax_im.imshow(offset_data_bias, cmap='gray', origin='lower', 
                    vmin=np.percentile(offset_data_bias, 5), 
                    vmax=np.percentile(offset_data_bias, 99))
    ax_im.set_title(f'Bias {galaxie_name}', fontsize=10)
    ax_im.axis('off')
    ax_im.grid(False)
    cbar = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04)
    cbar.set_label('ADU', fontsize=10)

    ax_hist.hist(offset_data_bias.flatten(), bins=bins, color='blue')
    ax_hist.axvline(x=peak_x, color='red', linestyle='--', label=f'Pic à {peak_x:.2f} ADU')
    ax_hist.loglog()
    ax_hist.set_xlabel('ADU', fontsize=10)
    ax_hist.set_ylabel('Nombre de pixels', fontsize=10)
    ax_hist.set_title('Histogramme Bias (offset)', fontsize=10)
    ax_hist.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bias_hist_sigma_stack(files, sigma=2):
    # Load data first
    data_list = []
    titles = []
    has_sigma_list = []

    for filename, title_prefix, has_sigma in files:
        with fits.open(filename) as hdul:
            data = hdul[0].data
            data_list.append(data)
            titles.append(title_prefix + (str(sigma) if has_sigma else ""))
            has_sigma_list.append(has_sigma)

    if len(data_list) == 0:
        print("Aucune image à afficher.")
        return

    # Create figure: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Plot images and individual histograms
    for i, (data, title, has_sigma) in enumerate(zip(data_list, titles, has_sigma_list)):
        # Only plot in the first two columns
        if i < 2:  # Limit to first two columns
            ax_img = axes[0, i]
            ax_hist = axes[1, i]

            im = ax_img.imshow(data, cmap='gray', origin='lower', 
                            vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
            cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
            cbar.set_label('ADU', fontsize=10)
            ax_img.axis('off')
            ax_img.set_anchor('C')
            ax_img.set_title(title, fontsize=10)

            # Different colors for histograms
            color = 'red' if has_sigma else 'blue'
            ax_hist.hist(data.flatten(), bins=1000, color=color)
            ax_hist.loglog()
            ax_hist.set_xlabel('ADU', fontsize=10)
            ax_hist.set_ylabel('N', fontsize=10)
            ax_hist.set_title('Hist ' + ('avec' if has_sigma else 'sans') + ' sigma', fontsize=10)
            ax_hist.grid()

    # Superposed histograms (bottom right cell)
    colors = ['red', 'blue']  # Changed green to blue for consistency
    labels = [f'Avec sigma = {sigma}', 'Sans sigma']
    ax_superposed = axes[1, 2]

    for data, color, label in zip(data_list, colors, labels):
        ax_superposed.hist(data.flatten(), bins=1000, color=color, alpha=0.7, label=label)

    ax_superposed.loglog()
    ax_superposed.set_xlabel('ADU', fontsize=10)
    ax_superposed.set_ylabel('N', fontsize=10)
    ax_superposed.set_title('Superposition des histogrammes', fontsize=10)
    ax_superposed.grid()
    ax_superposed.legend()

    # Keep the top right cell empty
    axes[0, 2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_master_flat_hist(filename, galaxie_name, bins=1000):
    hdul = fits.open(filename)
    offset_data_plat = hdul[0].data
    hdul.close()

    if offset_data_plat is None:
        print("Aucune image à afficher.")
        return
    
    hist_values, bin_edges = np.histogram(offset_data_plat.flatten(), bins=bins)
    # mean of values of pixels
    mean_value = np.mean(offset_data_plat)
    print(f"Valeur moyenne des pixels: {mean_value:.2f} ADU")

    fig, (ax_im, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))

    im = ax_im.imshow(offset_data_plat, cmap='gray', origin='lower', 
                    vmin=np.percentile(offset_data_plat, 5), 
                    vmax=np.percentile(offset_data_plat, 99))
    ax_im.set_title(f'Master Flat {galaxie_name}', fontsize=10)
    ax_im.axis('off')
    ax_im.grid(False)
    cbar = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04)
    cbar.set_label('ADU', fontsize=10)

    ax_hist.hist(offset_data_plat.flatten(), bins=bins, color='blue')
    ax_hist.axvline(x=mean_value, color='red', linestyle='--', label=f'Moyenne à {mean_value:.2f} ADU')
    ax_hist.loglog()
    ax_hist.set_xlabel('ADU', fontsize=10)
    ax_hist.set_ylabel('Nombre de pixels', fontsize=10)
    ax_hist.set_title('Histogramme Master Flat', fontsize=10)
    ax_hist.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

def process_filter(props):
    filter_name = props['filter_name']
    input_folder = props['input_folder']
    bias_path = props['bias_path']
    flat_folder = props['flat_folder']
    output_folder = props['output_folder']
    galaxy_name = props['galaxy_name']
    os = props['os']
    if not filter_name or not input_folder or not bias_path or not flat_folder or not output_folder or not os:
        print("Erreur: filter_name, input_folder, bias_path, flat_folder et output_folder doivent être fournis.")
        return

    # 1. Créer la liste des fichiers pour ce filtre
    pattern = os.path.join(input_folder, f'{galaxy_name}-{filter_name}-*.fit')
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"Aucun fichier trouvé pour le filtre {filter_name} !")
        return

    print(f"Trouvé {len(file_list)} fichiers pour le filtre {filter_name}.")

    if props['override_master_filename']:
        # 1.1. Si le nom du master flat est fourni, l'utiliser
        master_flat = os.path.join(flat_folder, props['override_master_filename'])
    else:
        # 2. Trouver le master flat correspondant
        master_flat = os.path.join(flat_folder, f'master_flat--{filter_name}-T32.fits')

    if not os.path.exists(master_flat):
        print(f"Le master flat {master_flat} n'existe pas !")
        return

    # 3. Définir le nom de l'image finale
    output_name = f'{galaxy_name}_{filter_name}_final_calibrated.fits'

    # 4. Appliquer stacking (calibration + empilement)
    final_image, saved_path = pr.stacking(file_list, bias_path, master_flat, out_name=output_name, out_dir=output_folder)

    return final_image, saved_path

def get_initial_image(props):
    filter_name = props['filter_name']
    input_folder = props['input_folder']
    galaxy_name = props['galaxy_name']
    os = props['os']
    if not filter_name or not input_folder or not os:
        print("Erreur: filter_name, input_folder et os doivent être fournis.")
        return None

    # 1. Créer la liste des fichiers pour ce filtre
    pattern = os.path.join(input_folder, f'{galaxy_name}-{filter_name}-*.fit')
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"Aucun fichier trouvé pour le filtre {filter_name} !")
        return None

    # 2. Charger la première image brute
    initial_image = fits.getdata(file_list[0])

    if initial_image is None:
        print(f"Aucune image brute trouvée pour le filtre {filter_name} !")
        return None
    
    return initial_image

def plot_images_and_histograms(props):
    """
    Plots the initial and final images and their histograms for a given filter.
    
    Parameters:
        filter_name (str): The filter name (e.g., 'Red', 'Blue', 'Green', 'HA').
        color_init (str): Color for the initial image histogram.
        color_final (str): Color for the final image histogram.
    """

    filter_name = props['filter_name']
    color_init = props['color_init']
    color_final = props['color_final']
    galaxy_name = props['galaxy_name']
    os = props['os']

    if not filter_name or not color_init or not color_final or not galaxy_name or not os:
        print("Erreur: filter_name, color_init et color_final et galaxy_name et os doivent être fournis.")
        return

    # Load images
    initial_image = get_initial_image(props)
    final_image, _ = process_filter(props)

    print("--- Pose individuelle ---")
    print(f'{galaxy_name} Image brute : Moyenne : {np.mean(initial_image):.2f} ADU, Ecart-type : {np.std(initial_image):.2f}')

    print("--- Image finale stackée ---")
    print(f"{galaxy_name} Image finale: Moyenne : {np.mean(final_image):.2f} ADU, Ecart-type : {np.std(final_image):.2f}")

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Initial image
    im_init = axes[0, 0].imshow(initial_image, cmap='gray', origin='lower',
                                vmin=np.percentile(initial_image, 5),
                                vmax=np.percentile(initial_image, 99))
    cbar = fig.colorbar(im_init, ax=axes[0, 0], pad=0.04, fraction=0.046)
    cbar.set_label('ADU', fontsize=10)
    axes[0, 0].set_title(f'{galaxy_name} - Image brute - {filter_name}', fontsize=10)
    axes[0, 0].axis('off')
    
    # Final image
    im_final = axes[0, 1].imshow(final_image, cmap='gray', origin='lower',
                                 vmin=np.percentile(final_image, 5),
                                 vmax=np.percentile(final_image, 99))
    cbar = fig.colorbar(im_final, ax=axes[0, 1], pad=0.04, fraction=0.046)
    cbar.set_label('ADU', fontsize=10)
    axes[0, 1].set_title(f'{galaxy_name} - Image finale - {filter_name}', fontsize=10)
    axes[0, 1].axis('off')
    
    # Initial histogram
    axes[1, 0].hist(initial_image.flatten(), bins=1000, color=color_init)
    axes[1, 0].loglog()
    axes[1, 0].set_xlabel('ADU', fontsize=10)
    axes[1, 0].set_ylabel('N', fontsize=10)
    axes[1, 0].set_title(f'{galaxy_name} - Histogramme image brute - {filter_name}', fontsize=10)
    axes[1, 0].grid()
    
    # Final histogram
    axes[1, 1].hist(final_image.flatten(), bins=1000, color=color_final)
    axes[1, 1].loglog()
    axes[1, 1].set_xlabel('ADU', fontsize=10)
    axes[1, 1].set_ylabel('N', fontsize=10)
    axes[1, 1].set_title(f'{galaxy_name} - Histogramme image finale - {filter_name}', fontsize=10)
    axes[1, 1].grid()
    axes[1, 1].set_xlim(1e0, 1e5)
    
    plt.tight_layout()
    plt.show()

def plot_custom_images_and_histograms(props):
    """
    Plots the initial and final images and their histograms for a given filter.
    
    Parameters:
        filter_name (str): The filter name (e.g., 'Red', 'Blue', 'Green', 'HA').
        color_init (str): Color for the initial image histogram.
        color_final (str): Color for the final image histogram.
    """
    galaxy_name = props['galaxy_name']
    filter_name = props['filter_name']
    initial_filename = props['initial_filename']
    final_filename = props['final_filename']

    file_list_initial = sorted(glob.glob(initial_filename))
    initial_image = fits.getdata(file_list_initial[0])

    if initial_image is None:
        print(f"Aucune image brute trouvée pour le filtre {filter_name} !")
        return None
    
    file_list_final = sorted(glob.glob(final_filename))
    final_image = fits.getdata(file_list_final[0])

    if final_image is None:
        print(f"Aucune image brute trouvée pour le filtre {filter_name} !")
        return None
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)

    # Initial image - avec axes[0] au lieu de axes[0, 0]
    im_init = axes[0].imshow(initial_image, cmap='gray', origin='lower',
                             vmin=np.percentile(initial_image, 5),
                             vmax=np.percentile(initial_image, 99))
    cbar = fig.colorbar(im_init, ax=axes[0], pad=0.04, fraction=0.046)
    cbar.set_label('ADU', fontsize=10)
    axes[0].set_title(f'{galaxy_name} - Image brute - {filter_name}', fontsize=10)
    axes[0].axis('off')
    
    # Final image - avec axes[1] au lieu de axes[0, 1]
    im_final = axes[1].imshow(final_image, cmap='gray', origin='lower',
                              vmin=np.percentile(final_image, 5),
                              vmax=np.percentile(final_image, 99))
    cbar = fig.colorbar(im_final, ax=axes[1], pad=0.04, fraction=0.046)
    cbar.set_label('ADU', fontsize=10)
    axes[1].set_title(f'{galaxy_name} - Image calibrée - {filter_name}', fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_image_in_color(props):
    """
    Crée et affiche uniquement l'image composite RGB à partir des images filtrées.
    
    props: [
        { 'filter_name': 'Red', 'color_init': 'red', 'color_final': 'red', 'galaxy_name': 'M83', 'os': os}, 
        { 'filter_name': 'Green', 'color_init': 'green', 'color_final': 'green', 'galaxy_name': 'M83', 'os': os}, 
        { 'filter_name': 'Blue', 'color_init': 'blue', 'color_final': 'blue', 'galaxy_name': 'M83', 'os': os},
    ]
    """
    final_images = []
    filter_names = []
    galaxy_name = None
    
    for filter_props in props:
        if not filter_props:
            print("Erreur: filter_props doit être fourni.")
            continue
        filter_name = filter_props['filter_name']
        filter_names.append(filter_name)
        
        # Si l'image finale est déjà fournie dans le dictionnaire
        if 'final_filename' in filter_props and filter_props['final_filename'] is not None:
            try:
                final_image_file = glob.glob(filter_props['final_filename'])
                if not final_image_file:
                    print(f"Aucun fichier trouvé pour le chemin {filter_props['final_filename']} !")
                    continue
                    
                final_image = fits.getdata(final_image_file[0])
                if final_image is None:
                    print(f"Impossible de charger l'image pour le filtre {filter_name} !")
                    continue
                    
                galaxy_name = filter_props.get('galaxy_name', galaxy_name)
                final_images.append(final_image)
            except Exception as e:
                print(f"Erreur lors du chargement du fichier {filter_props['final_image']}: {str(e)}")
                continue
        else:
            # Sinon, utiliser process_filter pour l'obtenir
            color_init = filter_props['color_init']
            color = filter_props['color_final']
            galaxy_name = filter_props['galaxy_name']
            os = filter_props['os']
            
            if not filter_name or not color_init or not color or not galaxy_name or not os:
                print("Erreur: filter_name, color_init, color, galaxy_name et os doivent être fournis.")
                continue
                
            final_image, _ = process_filter(filter_props)
            
        final_images.append(final_image)
    
    if len(final_images) < 3:
        print("Erreur: trois filtres (Rouge, Vert, Bleu) sont nécessaires pour créer une image RGB.")
        return None
    
    ## Images seems not aligned
    # Alignement 
    centers = []
    for img in final_images:
        y_center, x_center = np.unravel_index(np.argmax(img), img.shape)
        centers.append((y_center, x_center))
    
    # Décalage
    reference_center = centers[0] # Réference
    shifts = []
    for center in centers:
        y_shift = reference_center[0] - center[0]
        x_shift = reference_center[1] - center[1]
        shifts.append((y_shift, x_shift))
    
    aligned_images = []
    for i, img in enumerate(final_images):
        y_shift, x_shift = shifts[i]
        aligned = np.zeros_like(img)
        
        # Copier les données avec le décalage approprié
        if y_shift >= 0 and x_shift >= 0:
            aligned[y_shift:, x_shift:] = img[:img.shape[0]-y_shift, :img.shape[1]-x_shift]
        elif y_shift >= 0 and x_shift < 0:
            aligned[y_shift:, :img.shape[1]+x_shift] = img[:img.shape[0]-y_shift, -x_shift:]
        elif y_shift < 0 and x_shift >= 0:
            aligned[:img.shape[0]+y_shift, x_shift:] = img[-y_shift:, :img.shape[1]-x_shift]
        else:  # y_shift < 0 and x_shift < 0
            aligned[:img.shape[0]+y_shift, :img.shape[1]+x_shift] = img[-y_shift:, -x_shift:]
        
        aligned_images.append(aligned)
    
    # Images alignées pour la suite
    shapes = [img.shape for img in aligned_images]
    min_height = min(shape[0] for shape in shapes)
    min_width = min(shape[1] for shape in shapes)
    
    # Recadrer les images
    for i in range(len(aligned_images)):
        if aligned_images[i].shape != (min_height, min_width):
            aligned_images[i] = aligned_images[i][:min_height, :min_width]
    
    # Normaliser les images pour l'affichage
    normalized_images = []
    for img in aligned_images:
        # Clip les valeurs extrêmes
        p_low, p_high = np.percentile(img, [1, 99])
        img_norm = np.clip(img, p_low, p_high)
        
        # Normalisation entre 0 et 1
        img_norm = (img_norm - p_low) / (p_high - p_low)
        normalized_images.append(img_norm)
    
    # RGB vide
    rgb_image = np.zeros((min_height, min_width, 3))
    
    # Filtre (R, G, B)
    filter_dict = {name: img for name, img in zip(filter_names, normalized_images)}
    
    if 'Red' in filter_dict:
        rgb_image[:, :, 0] = filter_dict['Red']
    if 'Green' in filter_dict:
        rgb_image[:, :, 1] = filter_dict['Green']
    if 'Blue' in filter_dict:
        rgb_image[:, :, 2] = filter_dict['Blue']
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, origin='lower')
    plt.title(f"{galaxy_name} - Image composite RGB", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def return_flux(image,rad_list,pix_siz = 1,search_center = False,rm_background = True,diagnostic = True, galaxy_name = '', filter_name = '', color = 'black'):
    """
    A function to measure the flux in apertures
    ...

    Arguments
    ----------
    image : np.array
        the 2D array that contains the photometric image
    rad_list : list
        the list of radius apertures
    pix_siz : float 
        the conversion factor from the unit of rad_list to pixels
        default = 1 (i.e no conversion)
    search_center : boolean
        if True, the centroid is determined in a box of size 300pix around the image center
        default = True
        USE WITH CARE
    rm_background : if True, the background is measured from the median of pixels, and subtracted
        default = True
    diagnostic : if True, a plot with the apertures on top of the images is shown.
    Output
    -------
    phot_table : the list of fluxes in apertures corresponding to rad_list
    bckg : the background ( = 0 if rm_background was set to True)
    
    """
    rad_list = np.array(rad_list)
    rad_pix_list = rad_list/pix_siz 
    default_center = [np.shape(image)[1]//2,np.shape(image)[0]//2]
    if (search_center == True):
        positions = (centroid_quadratic(image, xpeak=default_center[0], ypeak=default_center[1],search_boxsize = 301))
    else:
        positions = default_center

    warnings.filterwarnings('ignore')
        
    ## masking of pixels with value above the  value at the center of the galaxy
    mask = np.zeros(image.shape, dtype=bool)
    image_center = image[int(positions[1])-20:int(positions[1])+20,int(positions[0])-20:int(positions[0])+20]
    mean_cent = np.mean(image_center)
    std_cent =  np.std(image_center)
    mask[np.where((image-mean_cent)>5*std_cent)] = True
    
    ## plot diagnostic
    if (diagnostic == True):
        mm = np.mean(np.log10(image_center))
        ss = np.std(np.log10(image_center))
        image_clip = np.zeros(np.shape(image))+image
        image_clip[np.where(mask == True)] = 0
        plt.imshow(np.log10(image_clip),cmap='gray',origin='lower',vmin = mm-2*ss,vmax = mm+ss/2)
        plt.scatter(positions[0],positions[1],color=color,marker = '+',alpha=0.5)
        axes = plt.gca()
        axes.set_xlim(positions[0]-np.max(rad_pix_list)-100,positions[0]+np.max(rad_pix_list)+100)
        axes.set_ylim(positions[1]-np.max(rad_pix_list)-100,positions[1]+np.max(rad_pix_list)+100)
        axes.axis('off')
        axes.set_title(f'{galaxy_name} - {filter_name}', fontsize=10)

        for k in range(len(rad_pix_list)):
            circle = plt.Circle((positions[0], positions[1]), rad_pix_list[k], color=color,fill=False)
            axes.add_patch(circle)
    
    if (rm_background==True):
        bckg = np.median(image_clip[mask==False])
    else:
        bckg = 0
        
    
    phot_table = np.zeros(len(rad_pix_list))
    for i,aperture_radius in enumerate(rad_pix_list):
        apertures = CircularAperture(positions, r=aperture_radius)  
        phot_table[i] = aperture_photometry(image-bckg, apertures,mask=mask)['aperture_sum'].value[0]


    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return phot_table, bckg

def size_luminosity(rad_list,flux_list):
    """
    A function to measure the galaxy size and total luminosity
    ...

    Arguments
    ----------
    rad_list : list
        the list of radius apertures
    flux_list : list
        the list of flux in the above specified apertures
    Output
    -------
    lum : the total flux of the galaxy
    rad : the effective radius of the galaxy
    rad_tot : the minimum radius that encloses the total flux
    
    """
    w = np.where(flux_list == np.max(flux_list))[0]
    lum = flux_list[w]
    w = np.where(np.abs(flux_list - lum/2) == np.min(np.abs(flux_list-lum/2)))[0]
    rad = rad_list[w]
    w = np.where(np.abs(flux_list - lum) == np.min(np.abs(flux_list-lum)))[0]
    rad_tot = rad_list[w]
    return lum,rad,rad_tot

def get_image_calibrated(filename):
    if not os.path.exists(filename):
        print(f"Le fichier {filename} n'existe pas.")
        return None
    image = fits.getdata(filename)
    if image is None:
        print(f"Aucune image trouvée dans le fichier {filename}.")
        return None
    return image

def calculate_magnitudes(flux_list, distance_mpc, rad_list, r_eff, zero_point=19.05):
    """
    Calculate apparent and absolute magnitudes from flux measurements.
    
    Parameters:
    -----------
    flux_list : array-like
        Flux measurements in detector units (e.g., ADU).
    distance_mpc : float
        Distance to the object in megaparsecs.
    zero_point : float, optional
        Photometric zero point in the AB system. Default is 19.05.
  
    Returns:
    --------
    dict
        Dictionary containing:
        - 'apparent_magnitudes': AB apparent magnitudes array
        - 'absolute_magnitudes': Absolute magnitudes array
        - 'distance_modulus': The distance modulus value
    """
    
    # Calculate apparent magnitudes (AB system)
    apparent_magnitudes = -2.5 * np.log10(flux_list) + zero_point
    
    # Calculate distance modulus: μ = 5*log10(d/10pc)
    distance_pc = distance_mpc * 1e6
    distance_modulus = 5 * np.log10(distance_pc / 10)
    
    # Calculate absolute magnitudes: M = m - μ
    absolute_magnitudes = apparent_magnitudes - distance_modulus

    # Effective magnitude
    idx = np.argmin(np.abs(rad_list - r_eff))
    effective_magnitude = absolute_magnitudes[idx]
    
    return {
        'apparent_magnitudes': apparent_magnitudes,
        'absolute_magnitudes': absolute_magnitudes,
        'distance_modulus': distance_modulus,
        'effective_magnitude': effective_magnitude,
    }

def plot_radial_magnitudes(magnitudes, rad_list, rad_eff, galaxy_name, filter_name):
    plt.figure(figsize=(6, 3))
    plt.plot(rad_list, magnitudes['absolute_magnitudes'])
    plt.axvline(x=rad_eff, color='r', linestyle='--', label='Rayon effective')
    axes = plt.gca()
    axes.set_xlabel('Madius (kpc)',fontsize=10)
    axes.set_ylabel('Magnitude',fontsize=10)
    plt.title(f'{galaxy_name} - {filter_name}', fontsize=10)
    plt.legend()
    plt.show()

def get_size_luminosity_data():
    magz_ellipt = np.array([-20.194174757281555, -20.32977354179308, -20.32977354179308, -20.65275025599211, -20.725565789972695, -20.85080874544903, -20.85080874544903, -20.444337122648665, -21.069578892976335, -21.225565789972695, -20.996439813410195, -21.132038597921724, -20.68381892824636, -20.809061883722695, -20.7569580078125, -20.631715052336165, -20.861164574484224, -20.81941771275789, -20.96537232630461, -20.840129370828276, -20.96537232630461, -21.07993472201153, -21.2987060546875, -21.46537232630461, -21.18414247383192, -21.31941771275789, -21.45501649726942, -21.39255679232403, -21.27766966588289, -21.621683954035195, -21.78834904050364, -21.78834904050364, -21.569578892976335, -21.694499488015776, -21.64239561210558, -21.486083984375, -21.27766966588289, -21.85080874544903, -21.74660099362864, -21.78834904050364, -21.95501649726942, -21.95501649726942, -21.95501649726942, -22.09061409663228, -22.11132575470267, -22.13203978307039, -21.85080874544903, -21.85080874544903, -21.975728155339805, -22.121683954035195, -22.28834904050364, -22.28834904050364, -22.35080874544903, -22.50712037317961, -22.49676454414442, -22.621683954035195, -22.65307380157767, -22.840452916413835, -22.99676454414442, -23.007442733616504, -22.861489305218445, -22.663429630612864, -22.580258267597088, -22.71553350652306, -22.851133476183254, -22.50712037317961, -22.50712037317961, -22.79902960027306, -22.95533885770631, -23.080258267597088, -22.621683954035195])
    reff_ellipt = np.array([1.5036364920323924, 1.1671934913978255, 1.1671934913978255, 1.3255040793180237, 1.4527491799205718, 1.4527491799205718, 1.615067095695038, 1.860047032524925, 1.3635358901542516, 1.3537042309082132, 1.5481376418045563, 1.684890059617907, 1.808165056123628, 1.9268903570358138, 2.17246319457596, 2.3314114839265963, 2.9228700054788113, 3.3899276529316933, 2.5737792133782427, 2.449868184026514, 2.1123301042354807, 1.9134161303581592, 1.8341228941753007, 2.14218652118343, 2.17246319457596, 2.3815363388427038, 2.5201626534829913, 2.723597937648605, 2.6291149233316813, 2.432736913883709, 2.3815363388427038, 2.3815363388427038, 2.685049447067081, 2.9228700054788113, 3.1367243349937426, 3.070704829165085, 3.0279097533131134, 2.821474140561766, 2.685049447067081, 3.1141047508236963, 3.092328652792303, 2.5737792133782427, 2.5737792133782427, 2.841342888155084, 3.1141047508236963, 3.4620512778059527, 3.5364818096244384, 3.5364818096244384, 3.8219575033829973, 3.8219575033829973, 3.6893718303953866, 3.413799435390171, 3.1588130662657576, 3.413799435390171, 3.7686895196430443, 4.159561049587781, 3.8488716288238165, 4.043542366000531, 4.278912016737381, 3.9593047538547586, 4.4638989563946545, 4.495333616869345, 4.790502087631065, 4.927954770660778, 4.824236678870472, 5.325752305029238, 5.325752305029238, 5.325752305029238, 5.796192323790767, 5.715413315729389, 6.176776585219983])


    magz_disk = np.array([-20.36310703314624, -20.21197450508192, -20.322977899347695, -20.040452679384103, -20.181552923998787, -20.383495145631066, -20.46407814396238, -20.3025886017142, -20.42394782501517, -20.474110427412015, -20.60550193416262, -20.645631067961165, -20.787054858161408, -20.928155102776092, -21.110032757509103, -21.099676928473908, -20.63559878451153, -20.75663446222694, -20.70614831424454, -20.615534217612257, -20.524595390245754, -20.524595390245754, -20.524595390245754, -20.68608374734527, -20.454045860512743, -20.867637856492717, -20.948543215260923, -21.00906164669296, -21.150161891307647, -21.231068435224515, -21.231068435224515, -21.281553398058254, -21.321682531856794, -21.483172074104974, -21.39255679232403, -21.362136396389563, -21.22103615177488, -21.12006504095874, -21.160194174757283, -21.00906164669296, -20.918122819326456, -20.80711942506068, -20.83754100614381, -21.089643459875607, -21.069578892976335, -21.039158497041868, -20.90809053587682, -20.787054858161408, -20.787054858161408, -20.696116030794904, -20.726537611878033, -20.676051463895632, -20.73656989532767, -21.22103615177488, -21.422653642672937, -21.594174283222088, -21.745631542020632, -21.56407743287318, -21.634628147754853, -21.463106322057037, -21.422653642672937, -21.61423885012136, -21.715211146086165, -21.80614997345267, -21.73559807342233, -21.95728250151699, -22.14919125455097, -22.15922235285194, -22.028154391686893, -21.86666603458738, -21.685113110588592, -21.56407743287318, -21.463106322057037, -21.463106322057037, -21.24110071867415, -21.785760675819176, -21.967638330552184, -22.108737390018202, -22.108737390018202, -22.57281553398058, -22.34077764714806, -22.28025921571602, -22.15922235285194, -22.038187860285195, -21.705177677487864, -21.685113110588592, -22.007767464350728, -22.270225747117717, -22.20970968598301, -22.492232535649272, -22.42168300591626, -22.40129370828277, -22.63333396541262, -22.18964511908374, -22.22977425288228, -22.038187860285195, -21.836245638652912, -22.582849002578882, -22.804853420813107, -22.774757755612864, -21.089643459875607, -20.242071355430824, -20.020064566899272, -23.016829111043688, -20.928155102776092])
    reff_disk = np.array([0.7305077602356541, 0.8258860468725422, 1.0417030627346284, 1.0950072414950993, 1.1408388132508438, 1.11516325144177, 1.2325687301272645, 1.289980232146166, 1.38114985899269, 1.5057631132147946, 1.4518236062556378, 1.2325687301272645, 1.38114985899269, 1.485250449087927, 1.4989672952164963, 1.38114985899269, 1.0089406900425395, 1.6268350154033149, 1.8730964839668607, 1.6490620643922178, 1.7899945596527838, 1.7899945596527838, 1.9162232418039469, 2.014570682567339, 2.0796845114930074, 2.117965666275469, 1.898965655779886, 1.7336978365480198, 1.822677419699831, 1.9339202836543326, 1.9339202836543326, 1.710329558814473, 1.5263359864110422, 1.710329558814473, 1.864642815068627, 2.1472161312600977, 2.0796845114930074, 2.2266666559835975, 2.3409471647843425, 2.4722507336621242, 2.3198644974871776, 2.3515602388162273, 2.6109191108787697, 2.5874047193770977, 2.744920465374478, 3.006585275737023, 2.7325320759002114, 2.6109191108787697, 2.6109191108787697, 2.529172605875244, 2.8340540390085125, 3.1037624465301996, 3.3081235870542285, 2.552157441689805, 2.2574189545368384, 2.02370407443461, 1.9964273754111086, 2.236762217913232, 2.3948458626620193, 2.552157441689805, 2.757365388038199, 2.770270467464691, 2.5641024451171974, 2.744920465374478, 2.952242772390612, 2.4722507336621242, 2.4722507336621242, 2.2676533395096774, 2.744920465374478, 3.0476639222715436, 3.292712962660338, 3.061927662997669, 3.3533221186314845, 3.6393734926206087, 3.3685249632036176, 3.5741155029434792, 3.323121960494278, 3.061927662997669, 3.061927662997669, 3.061927662997669, 3.033908784130542, 3.399634103135108, 3.277852273207597, 3.5903193511130898, 3.9142533363542706, 4.287386473619053, 3.8435057068153298, 3.8790008901929722, 4.287386473619053, 3.931999256067118, 4.115146288849563, 4.632116684648166, 5.050045411546503, 5.213270856495423, 5.788253460981814, 6.112916207937106, 6.057862221648741, 6.112916207937106, 6.340028981774115, 7.639762327420236, 3.8261586655104707, 1.9874170888852634, 2.0894178946417523, 4.98197786625372, 5.2852701785342315])

    return magz_ellipt, reff_ellipt, magz_disk, reff_disk

def process_size_luminosity(props):
    pixel_size = props['pixel_size']
    distance_mpc = props['distance_mpc']
    galaxy_name = props['galaxy_name']
    filter_name = props['filter_name']
    zero_point = props['zero_point']
    color = props['color']

    if pixel_size is None or distance_mpc is None or zero_point is None:
        print('Function needs all the arguments')
        return

    pixel_size_kpc = get_pixel_size_in_kpc(pixel_size, distance_mpc) # kpc/pixel
    print(f"Taille du pixel : {pixel_size_kpc:.3f} kpc/pixel")

    image_calibrated = get_image_calibrated(f'./Processed/{galaxy_name}_{filter_name}_final_calibrated.fits')

    if image_calibrated is None:
        print("Aucune image à afficher.")
        return

    rad_list = np.linspace(0.1,10,10)
    flux_list, _ = return_flux(image_calibrated,rad_list,pixel_size_kpc,search_center=True,rm_background=True,diagnostic=True, galaxy_name=galaxy_name, filter_name=filter_name, color=color)
    lum,rad,rad_tot = size_luminosity(rad_list,flux_list)
    print(f"Le flux totale : {lum[0]:.2f} ADU")
    print(f"Effective radius : {rad[0]:.2f} kpc")
    print(f"Minimum radius : {rad_tot[0]:.2f} kpc")
    magnitudes = calculate_magnitudes(flux_list, distance_mpc, rad_list, rad, zero_point)
    print(f"La moyenne des magnitudes se trouve autour de {magnitudes['apparent_magnitudes'].mean():.2f} mag AB")
    print(f"Magnitude effective {magnitudes['effective_magnitude']:.2f} mag AB")
    print(f"La moyenne des magnitudes à 10pc de la source se trouve autour de {magnitudes['absolute_magnitudes'].mean():.2f} mag")
    plot_radial_magnitudes(magnitudes, rad_list, rad, galaxy_name, filter_name)

def plot_galaxies_rayon_magnitude(magz_ellipt, reff_ellipt, magz_disk, reff_disk, effective_magnitude, effective_radius, galaxy_type='elliptique', gallaxy_name='M83'):
    fig, axes = plt.subplots(1,2,figsize = (18,8),sharex=True,sharey=True)

    ## Panneau de gauche : galaxies elliptiques
    ax = axes[0]

    # Plot with error bars
    ax.scatter(magz_ellipt,reff_ellipt,s = 200,alpha=0.7,color='red')
    if galaxy_type == 'elliptique':
        ax.scatter(effective_magnitude, effective_radius, color='green', s=200, alpha=0.7, label=gallaxy_name)
    
    #ax.plot(x_fit_ell, y_fit_ell, color='darkred',
    #        label=f'Ellipticals fit: y = {coeffs_ell[0]:.2f} x + {coeffs_ell[1]:.2f}')

    ax.legend(loc = 'lower left', fontsize = 16)
    ax.set_xlim(-20,-24)
    ax.set_ylim(0.5,20)
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude (AB)',fontsize=16)
    ax.set_ylabel('rayon effectif (kpc)',fontsize=16)
    ax.set_title('Galaxies elliptiques',fontsize = 20)
    ax.grid()
    ## Panneau de droite : galaxies disques
    ax1 = axes[1]

    # ax1.plot(x_fit_disk, y_fit_disk, color='darkblue',
    #        label=f'Disks fit: y = {coeffs_disk[0]:.2f} x + {coeffs_disk[1]:.2f}')
    ax1.scatter(magz_disk,reff_disk,s = 200,alpha=0.7,color='blue')
    if galaxy_type == 'spiral':
        # ax1.errorbar(effective_magnitude, effective_radius, xerr=0.1, yerr=[0.15], color='green', s=200, alpha=0.7, label=gallaxy_name)
        ax1.scatter(effective_magnitude, effective_radius, color='green', s=200, alpha=0.7, label=gallaxy_name)

    # Add a point for the M87 galaxy effective radius (rad) and magnitude (mag_tot)
    #ax1.scatter(magnitudes['effective_magnitude'],rad, color='green', s=200, alpha=0.7, label='M83')
    ax1.legend(loc = 'upper left', fontsize = 16)
    ax1.set_xlim(-20.3,-23.7)
    ax1.set_ylim(0.5,20)
    ax1.set_yscale('log')
    ax1.set_xlabel('Magnitude (AB)',fontsize=16)
    ax1.set_title('Galaxies disques',fontsize = 20)
    ax1.grid()
    for i in range(2):
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].tick_params(axis='both', which='minor', labelsize=14)
    plt.show()

def load_data_from_gadotti():
    # Set Vizier row limit (you can change or set to -1 for all available rows)
    Vizier.ROW_LIMIT = -1

    # Define only the columns we need
    columns = [
        "MType", "MType2",  # Morphology
        "gMAGd", "rMAGd", "iMAGd",  # Disk magnitudes
        "gMAGbu", "rMAGbu", "iMAGbu",  # Bulge magnitudes
        "h", "re",  # Disk scalelength and Bulge effective radius
        "Blg/T", "D/T", "Bar/T",  # Component ratios
        "z",  # Redshift
        "(g-i)d", "(g-i)bu"  # Color indices
    ]

    # Load the table: J/MNRAS/393/1531 (Dimitri A. Gadotti 2009)
    catalog_id = "J/MNRAS/393/1531"
    vizier = Vizier(columns=columns)
    result = vizier.get_catalogs(catalog_id)

    # The main table is usually the first one
    data = result[0].to_pandas()

    # Preview the dataframe
    print("Total entries:", len(data))
    print(data.head())

    # Regroupement des galaxies selon leur type morphologique 
    # 1. Galaxies dominées par le bulbe
    bulge_dominated = data[data["MType"].isin(["elliptical", "classical"])]
    x_bulge = bulge_dominated["iMAGbu"]  # Magnitude du bulbe
    y_bulge = bulge_dominated["re"]      # Rayon effectif du bulbe

    # 2. Galaxies dominées par le disque
    disk_dominated = data[data["MType"].isin(["pseudo-bulge", "bulgeless", "barred", "unbarred"])]
    x_disk = disk_dominated["iMAGd"]    # Magnitude du disque
    y_disk = disk_dominated["h"]        # Échelle de longueur du disque

    return data, x_bulge, y_bulge, x_disk, y_disk

def plot_relations_taille_luminosite_gadotti(props):
    x_bulge = props['x_bulge']
    y_bulge = props['y_bulge']
    x_disk = props['x_disk'] 
    y_disk = props['y_disk']
    magnitudes = props['magnitudes']
    galaxy_type = props['galaxy_type']
    rad = props['rad']

    # Création de la figure comparative
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Panneau de gauche : Galaxies dominées par le bulbe
    xerr_bulge = [0.1] * len(x_bulge)
    yerr_bulge = 0.2 * y_bulge
    ax1.scatter(x_bulge,y_bulge,s = 200,alpha=0.7,color='red')
    ax1.errorbar(x_bulge, y_bulge, xerr=xerr_bulge, yerr=yerr_bulge, 
                fmt='o', ecolor='gray', alpha=0.6, markersize=4, label='Gadotti (2009)')

    ax1.set_title("Galaxies dominées par le bulbe")
    ax1.set_xlabel("Magnitude du bulbe (iMAGbu)")
    ax1.set_ylabel("Rayon effectif (kpc)")
    ax1.invert_xaxis()
    ax1.grid(True)

    # Panneau de droite : Galaxies dominées par le disque
    xerr_disk = [0.1] * len(x_disk)
    yerr_disk = 0.2 * y_disk
    ax2.scatter(x_disk,y_disk,s = 200,alpha=0.7,color='blue')
    ax2.errorbar(x_disk, y_disk, xerr=xerr_disk, yerr=yerr_disk, 
                fmt='o', ecolor='gray', alpha=0.6, markersize=4, label='Gadotti (2009)')

    # Ajout de notre point de mesure M83
    if galaxy_type == 'spiral':
        ax2.scatter(magnitudes['effective_magnitude'], rad, color='green', s=100, 
                alpha=0.9, label='M83')
        ax2.errorbar(magnitudes['effective_magnitude'], rad, xerr=[0.1], 
                    yerr=[0.2 * rad[0]], fmt='o', color='green', ecolor='green', alpha=0.9)

    ax2.set_title("Galaxies dominées par le disque")
    ax2.set_xlabel("Magnitude du disque (iMAGd)")
    ax2.invert_xaxis()
    ax2.grid(True)
    ax2.legend()

    plt.suptitle("Relations Taille-Luminosité par composante galactique", fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------
# UNITÉS D'ASTRONOMIE
# --------------------

# Parsec -> Mètres
def pc2m(pc):
    """Convertir parsec en mètres."""
    return pc * 3.085677581491367e16

# Mètres -> Parsec
def m2pc(m):
    """Convertir mètres en parsec."""
    return m / 3.085677581491367e16

# Parsec -> Années-lumière
def pc2ly(pc):
    """Convertir parsec en années-lumière."""
    return pc * 3.26156

# Années-lumière -> Parsec
def ly2pc(ly):
    """Convertir années-lumière en parsec."""
    return ly / 3.26156

# Arcseconds -> Radians
def arcsec2rad(arcsec):
    """Convertir secondes d'arc en radians."""
    return arcsec * (np.pi / (180.0 * 3600.0))

# Radians -> Arcseconds
def rad2arcsec(rad):
    """Convertir radians en secondes d'arc."""
    return rad * (180.0 * 3600.0) / np.pi

# --------------------
# ANGLES
# --------------------

# Degrés, minutes, secondes -> Degrés décimaux
def dms2deg(degrees, minutes, seconds):
    """Convertir (degrés, minutes, secondes) en degrés décimaux."""
    return degrees + minutes/60 + seconds/3600

# Degrés décimaux -> Degrés, minutes, secondes
def deg2dms(deg):
    """Convertir degrés décimaux en (degrés, minutes, secondes)."""
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m/60) * 3600
    return d, m, s

# --------------------
# LUMINOSITÉ ET PHOTOMÉTRIE
# --------------------

# Flux -> Magnitude
def flux2mag(flux, zero_point):
    """Convertir flux (en µJy ou autre) en magnitude apparente."""
    return -2.5 * np.log10(flux) + zero_point

# Magnitude -> Flux
def mag2flux(mag, zero_point):
    """Convertir magnitude apparente en flux."""
    return 10**((zero_point - mag) / 2.5)

# Module de distance
def distance_modulus(distance_pc):
    """Calculer le module de distance μ = 5 * log10(D/10 pc)."""
    return 5 * np.log10(distance_pc / 10)

# Flux apparent -> Flux absolu
def apparent_to_absolute_flux(flux_apparent, distance_pc):
    """Convertir un flux apparent en flux absolu (comme à 10 pc)."""
    d = distance_pc
    return flux_apparent * (d/10)**2

# --------------------
# AUTRES OUTILS
# --------------------

# Conversion kpc -> arcsec (besoin de la distance)
def kpc2arcsec(kpc, distance_pc):
    """Convertir kpc en arcsec à une distance donnée."""
    # 1 radian correspond à distance_pc parsec
    angle_rad = (kpc * 1e3) / pc2m(distance_pc)
    return rad2arcsec(angle_rad)

# Conversion arcsec -> kpc
def arcsec2kpc(arcsec, distance_pc):
    """Convertir arcsec en kpc à une distance donnée."""
    angle_rad = arcsec2rad(arcsec)
    return (angle_rad * pc2m(distance_pc)) / 1e3
