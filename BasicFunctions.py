# Function check if a string is a float
import torch
import numpy as np
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


def words2indices(propdata,word2idx, maxsentences=42,maxwords=16,pad_index = 509):
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
    mean_squared_error = np.mean(squared_errors)

    # Take the square root of the mean squared error to get the RMSE
    rmse_value = np.sqrt(mean_squared_error)

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

    return correlation_coefficients