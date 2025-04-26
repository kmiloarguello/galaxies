import numpy as np

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
