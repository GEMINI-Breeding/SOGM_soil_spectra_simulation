# Description: This file contains the basic functions used for the SOGM model
import torch
import numpy as np
import json

# Function check if a string is a float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Isolate single parenthesis for input list of words
def isoparenthesis(wordlist):
    for i in range(len(wordlist)):
        # if both left and right parentheses are in the same word, split them
        if '(' in wordlist[i] and ')' in wordlist[i]:
            continue
        elif '(' in wordlist[i]:
            wordlist[i] = wordlist[i].replace('(','')
            wordlist.append('(')
        elif ')' in wordlist[i]:
            wordlist[i] = wordlist[i].replace(')','')
            wordlist.append(')')
        else:
            continue
    return wordlist

# Convert 3D tensor index to 2D tensor index
def getreshapeindices(original_indices, shape_of_original):
    index_Ar = [original_indices[0] * shape_of_original[1] + original_indices[1], original_indices[2]]
    return index_Ar

# Convert text to indices
def words2indices(propdata, maxsentences=42,maxwords=16,pad_index = 509):
    with open('lib/word2idx_brand.json', 'r') as file:
        word2idx = json.load(file)
    # convert the words into indices with float type
    propdata_idx = torch.ones(len(propdata), maxsentences, maxwords) * pad_index
    for ipropind in range(0, len(propdata)):
        iprop = propdata[ipropind]  # get the sentence list for each sample
        for isentenceind in range(len(iprop)):
            sentence_words = isoparenthesis(iprop[isentenceind].split())
            for iwordind in range(len(sentence_words)):
                word = sentence_words[iwordind]
                if isfloat(word):
                    if float(word) >= 0:
                        propdata_idx[ipropind, isentenceind, iwordind] = -float(word)
                    else:
                        print("warning: negative number: ", float(word))
                else:
                    propdata_idx[ipropind, isentenceind, iwordind] = float(word2idx[word.lower()])
    return propdata_idx


def rmse(y_true, y_pred):
    # get mask that y_true between 0 and 1
    mask = (y_true > 0) & (y_true < 1)
    # Calculate the square of the differences
    squared_errors = pow(y_true[mask] - y_pred[mask], 2)

    # Calculate the mean of the squared errors
    mean_squared_error = torch.mean(squared_errors)

    # Take the square root of the mean squared error to get the RMSE
    rmse_value = torch.sqrt(mean_squared_error)

    return rmse_value

def getRMSE(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape.")

    num_signals = tensor1.shape[0]
    rmse_values = torch.zeros(num_signals)

    for i in range(num_signals):
        rmse_values[i] = rmse(tensor1[i], tensor2[i])

    return rmse_values

def getR(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same shape.")

    num_signals = tensor1.shape[0]
    correlation_coefficients = np.zeros(num_signals)

    for i in range(num_signals):
        mask = (tensor1[i] > 0) & (tensor1[i] < 1)

        correlation_coefficients[i] = np.corrcoef(tensor1[i][mask], tensor2[i][mask])[0, 1]

    correlation_coefficients = torch.tensor(correlation_coefficients)
    return correlation_coefficients


def setProp(specbrand, clay, silt, SOM, nitrogen, CEC, OC, tP, pH_w, Fe, dens, ec=-1, caco3=-1, tc=-1):
    prop = ['Samples : Soil']
    if not specbrand == None:
        prop = prop + ['Spectrometer : ' + specbrand]
    if clay >= 0:
        prop = prop + ['Clay content : ' + str(clay) + ' %']
    if silt >= 0:
        prop = prop + ['Silt content : ' + str(silt) + ' %']
    if silt >= 0 and clay >= 0:
        prop = prop + ['Sand content : ' + str(100 - clay - silt) + ' %']
    if SOM >= 0:
        prop = prop + ['Soil organic matter : ' + str(SOM) + ' g/kg']
    if nitrogen >= 0:
        prop = prop + ['Total nitrogen content : ' + str(nitrogen) + ' g/kg']
    if CEC >= 0:
        prop = prop + ['Cation exchange capacity : ' + str(CEC) + ' cmol(+)/kg']
    if OC >= 0:
        prop = prop + ['Organic carbon content : ' + str(OC) + ' g/kg']
    if tP >= 0:
        prop = prop + ['Total phosphorus content : ' + str(tP) + ' mg/kg']
    if pH_w >= 0:
        prop = prop + ['pH measured from water solution : ' + str(pH_w)]
    if Fe >= 0:
        prop = prop + ['Iron content : ' + str(Fe) + ' mg/kg']
    if dens >= 0:
        prop = prop + ['Soil bulk density : ' + str(dens) + ' g/cm3']
    if ec >= 0:
        prop = prop + ['Electrical conductivity : ' + str(ec) + ' mS/m']
    if caco3 >= 0:
        prop = prop + ['CaCO3 content : ' + str(caco3) + ' g/kg']
    if tc >= 0:
        prop = prop + ['Total carbon content : ' + str(tc) + ' %']

    return prop


def saveSpectra2heliosxml(spectra_data, filename):
    """
    Save multiple spectra to an XML file.

    :param spectra_data: list of tuples, where each tuple contains:
        - wavelengths: 1D array-like of wavelengths
        - spectrum: 1D array-like of spectrum data
        - label: str, the label to use in the XML file
    :param filename: str, the name of the file to save
    """
    # Begin constructing XML-like string
    xml_content = '<helios>\n\n\t<!-- -->\n'

    # Iterate over all spectra data
    for wavelengths, spectrum, label in spectra_data:
        xml_content += f'\t<globaldata_vec2 label="{label}">\n'

        # Adding wavelength and spectrum data
        for w, s in zip(wavelengths, spectrum):
            xml_content += f"\t\t{w} {s:.6f}\n"

        xml_content += "\t</globaldata_vec2>\n\n"

    # Closing tag
    xml_content += '</helios>'

    # Save string to file
    with open(filename, "w") as file:
        file.write(xml_content)