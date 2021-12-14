# =====Tianyi Zhang======
# read eicu data files, and save the medical records and labels as numpy array
#
# input: drug name:...
#        admission diagnosis:...
# predict: disease(apacheadmissiondx)[CVA, CHF, Sepsis, Asthma, pulmonary, Pneumonia, Cardiomyopathy]
#        unitdischargeoffset, unittype, unitstaytype, unitdischargelocation, unitdischargestatus

import os
import numpy as np
import pandas as pd
import re, io
import torch
from torch.nn.utils.rnn import pad_sequence
root = '../../datasets/eicu'

drugspath = os.path.join(root, 'admissionDrug.csv')
patientspath = os.path.join(root, 'patient.csv')
diagnosispath = os.path.join(root, 'admissionDx.csv')

drugsdata = pd.read_csv(drugspath, usecols=['patientunitstayid', 'drugname'])
patientsdata = pd.read_csv(patientspath, usecols=['patientunitstayid', 'gender', 'age', 'hospitaldischargeoffset', 'unitdischargeoffset', 'hospitaldischargestatus', 
                                                'hospitaldischargelocation', 'unittype', 'unitstaytype', 'unitdischargelocation', 'unitdischargestatus', 'apacheadmissiondx'])
diagnosisdata = pd.read_csv(diagnosispath, usecols=['patientunitstayid', 'admitdxpath'])
                                                
print(drugsdata.head(5))
print(patientsdata.head(5))
print(diagnosisdata.head(5))
print(len(drugsdata.iloc[:, 0]))

# null
null_columns=drugsdata.columns[drugsdata.isnull().any()]
print(drugsdata[null_columns].isnull().sum())
null_columns=patientsdata.columns[patientsdata.isnull().any()]
print(patientsdata[null_columns].isnull().sum())
null_columns=diagnosisdata.columns[diagnosisdata.isnull().any()]
print(diagnosisdata[null_columns].isnull().sum())

patientsdata.fillna(0, inplace=True)
allDrugs = list(set(drugsdata.iloc[:, 1]))
allpatientIDs = list(set(drugsdata.iloc[:, 0]))
print('num of drugs', len(allDrugs))
print('num of patients', len(allpatientIDs))
# testids = list(set(diagnosisdata.iloc[:, 0]))
# print('num of patients', len(testids))
# print(allpatientIDs[10000])
# print(allpatientIDs.index(1731426))

### data
# integrate all drug names
drugs = {}
drug = "Drug name: "
for i in range(len(drugsdata)):
    id = drugsdata.iloc[i, 0]
    # print(drug)
    if i < len(drugsdata) -1:
        if drugsdata.iloc[i+1, 0] == id:
            drug = drug + drugsdata.iloc[i, 1].split('       ')[0] + " | "
            # print(drug)
        else:
            drug = drug + drugsdata.iloc[i, 1].split('       ')[0]
            drugs[id] = drug
            drug = "Drug name: "
    else:
        if drugsdata.iloc[i-1, 0] == id:
            drug = drug + drugsdata.iloc[i, 1].split('       ')[0] + " | "
            drugs[id] = drug
            drug = "Drug name: "
        else:
            drug = "Drug name: "
            drug = drug + drugsdata.iloc[i, 1].split('       ')[0]
            drugs[id] = drug
            drug = "Drug name: "
    # print(id)
    # print(drug)
    # print(drugs)
print("drugs names finished!")

# integrate drug names and diagnosis
infos = {}
info = ""
for i in range(len(diagnosisdata)):
    id = diagnosisdata.iloc[i, 0]
    if id in allpatientIDs:
        info = drugs[id]
        # print(info)
        info = info + '\n' + diagnosisdata.iloc[i, 1]
        # print(info)
        infos[id] = info

def preprocess(s):

    s = re.sub(r"([?!.,¿])", r" \1", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    
    s = s.strip()
    return s

def tokenizer(data):
    lib = []
    tensor = []
    text2index = {}
    index2text = {}
    vocab_sz = 0

    for i in range(len(data)):
        text = data[i].split()
        for j in range(len(text)):
            if text[j] not in lib:
                lib.append(text[j])

    text2index = {m:n for n, m in enumerate(lib)}
    index2text = {n:m for n, m in enumerate(lib)}
    vocab_sz = len(lib)
    for i in range(len(data)):
        text = data[i].split()
        # pylint: disable=no-member
        id = torch.zeros(len(text), dtype = torch.int)
        for j in range(len(text)):
            id[j] = text2index[text[j]]
        tensor.append(id)
    tensor = pad_sequence(tensor, batch_first = True, padding_value = 0)
    tensor = np.asarray(tensor)
    return tensor, text2index, index2text, vocab_sz

data = []
for i in range(len(infos)):
    lines = list(infos.values())[i].strip().split('\n')
    data.append(preprocess(' '.join(lines)))

tensor, text2index, index2text, vocab_sz = tokenizer(data)

### disease label
allD = ['CVA', 'CHF', 'Sepsis', 'Asthma', 'pulmonary', 'Pneumonia', 'Cardiomyopathy']
dlabels = []
pdisease = patientsdata.iloc[:, 3]
print(pdisease[0])
ids = list(infos)
pids = list(patientsdata.iloc[:, 0])
for i in range(len(infos)):
    dlabel = []
    id = ids[i]
    if id in pids:
        index = pids.index(id)
        diseases = pdisease[index].split(', ')
        for j in range(len(allD)):
            if allD[j] in diseases:
                dlabel.append(1)
            else:
                dlabel.append(0)
        # print(id)
        # print(dlabel)
        dlabels.append(dlabel)

print("save as file")
outputpath = os.path.join(root, '/data/diagdata.npz')
np.savez(outputpath, data = tensor,                        
        t2i = text2index, i2t = index2text, v_sz = vocab_sz)
# outputpath = os.path.join(root, 'label2.npz')
# np.savez(outputpath, label)
outputpath = os.path.join(root, '/data/disease.npz')
np.savez(outputpath, dlabels)