from astroquery.vizier import Vizier
import pandas as pd

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

# Optional: Filter Elliptical vs Disk
# For example: classify "Elliptical" as MType starting with 'E', "Disk" with 'S' or 'Sb' etc.
ellipticals = data[data["MType"].str.startswith("E", na=False)]
disks = data[data["MType"].str.startswith("S", na=False)]

print("Ellipticals:", len(ellipticals))
print("Disks:", len(disks))
