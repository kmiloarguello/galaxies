## 4 


offset_hdul = fits.open("offset_001.fit")
offset_data = offset_hdul[0].data

plt.hist(offset_data.flatten(), bins=100)
plt.title("Histogramme des valeurs d'une pose d'offset")
plt.xlabel("Valeurs")
plt.ylabel("Nombre de pixels")
plt.show()

# Construction de la pose maître d'offset
from processing import master_bias

offset_files = ["offset_001.fit", "offset_002.fit", ...]  # À compléter
bias_master = master_bias(offset_files)

# Sauvegarde
fits.writeto("master_bias.fits", bias_master, overwrite=True)