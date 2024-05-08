# SOGM
The Soil Optical Generative Model (SOGM) is a soil spectra simulation model that utilizes the Denoising Diffusion Probabilistic Model (DDPM). The model generates soil reflectance spectra from text-based inputs describing soil properties and their values rather than only numerical values and labels in binary vector format, which means the model can handle variable formats for property reporting. Because the model is generative, it can simulate reasonable output spectra based on an incomplete set of available input properties, which becomes more constrained as the input property set becomes more complete. Two additional sub-models were also built to complement the SOGM: a spectral padding model that can fill in the gaps for spectra shorter than the target visible-near-infrared range (VIS-NIR; 400 to 2499 nm), and a wet soil spectra model that can estimate the effects of water content on soil reflectance spectra given the dry spectrum predicted by the SOGM.

Additionally, the SOGM is packaged with the PROSAIL model, which combines the PROSPECT-5, D, and 4SAIL models.

Citation: Lei, T., & Bailey, B. N. (2024). A text-based, generative deep learning model for soil reflectance spectrum simulation in the VIS-NIR (400-2499 nm) bands. arXiv preprint arXiv:2405.01060.

Example of input properties:

# Coarse fragments : 1 %
# Clay content : 1 %
# Silt content : 1 %
# Sand content : 1 %
# Bulk density : 1 g/cm3
# Soil organic matter : 1 g/kg
# Total carbon content : 1 g/kg
# Organic carbon content : 1 g/kg
# pH measured from CaCl2 solution : 1
# pH measured from water solution : 1
# Electrical conductivity : 1 mS/m
# CaCO3 content : 1 g/kg
# Total nitrogen content : 1 g/kg
# Extractable phosphorus content : 1 mg/kg
# Extractable potassium content : 1 mg/kg
# Cation exchange capacity : 1 cmol(+)/kg
# The primary land cover : Maize
# The primary land use : Forestry
# Percentage of stones in soil : <10 %
# Country : United States
# Province/State : Sao Paulo
# Total magnesium content : 1 mg/kg
# Total aluminium content : 1 mg/kg
# Total phosphorus content : 1 mg/kg
# Total calcium content : 1 mg/kg
# Total manganese content : 1 mg/kg
# Total iron content : 1 mg/kg
# Total zinc content : 1 mg/kg
# Total nickel content : 1 mg/kg
