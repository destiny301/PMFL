import os
import numpy as np
import pandas as pd

root = '../../Dataset/eicu'

drugspath = os.path.join(root, 'admissionDrug.csv')
patientspath = os.path.join(root, 'patient.csv')

drugsdata = pd.read_csv(drugspath, usecols=['patientunitstayid', 'drugname'])
patientsdata = pd.read_csv(patientspath, usecols=['patientunitstayid', 'gender', 'age', 'hospitaldischargeoffset', 'unitdischargeoffset', 'hospitaldischargestatus', 
                                                'hospitaldischargelocation', 'unittype', 'unitstaytype'])
print(drugsdata.head(5))
print(patientsdata.head(5))
print(len(drugsdata.iloc[:, 0]))

# null
null_columns=drugsdata.columns[drugsdata.isnull().any()]
print(drugsdata[null_columns].isnull().sum())
null_columns=patientsdata.columns[patientsdata.isnull().any()]
print(patientsdata[null_columns].isnull().sum())

patientsdata.fillna(0, inplace=True)
allDrugs = list(set(drugsdata.iloc[:, 1]))
allpatientIDs = list(set(drugsdata.iloc[:, 0]))
print('num of drugs', len(allDrugs))
print('num of patients', len(allpatientIDs))
print(allpatientIDs[10000])
print(allpatientIDs.index(1731426))

allage = list(set(patientsdata.iloc[:, 2])) # 3(NULL = 0)
print(len(allage))
print(allage)
allgender = list(set(patientsdata.iloc[:, 1])) # 3(NULL = 0)
print(len(allgender))
print(allgender)
alllocations = list(set(patientsdata.iloc[:, 4])) # 9(NULL = 0)
print(len(alllocations))
print(alllocations)
allstatus = list(set(patientsdata.iloc[:, 5])) # 3 (NULL = 0)
print(len(allstatus))
print(allstatus)
alltypes = list(set(patientsdata.iloc[:, 6])) # 8
print(len(alltypes))
print(alltypes)
allstaytypes = list(set(patientsdata.iloc[:, 7])) # 4
print(len(allstaytypes))
print(allstaytypes)

doffset = patientsdata.iloc[:, 3] # <=3000, <=10000, 
uoffset = patientsdata.iloc[:, 8] # <=0, <=5000, 
print(np.mean(doffset), np.median(doffset))
print(np.mean(uoffset), np.median(uoffset))
print(np.min(doffset), np.max(doffset))
print(np.min(uoffset), np.max(uoffset))

print("generate array!")
# generate data and label array
data = np.zeros((len(allpatientIDs), len(allDrugs)+2))
print(np.shape(data))

label = np.zeros((len(allpatientIDs), 6))
print(np.shape(label))

print(len(drugsdata))
print(len(patientsdata))
for i in range(len(drugsdata)):
    pid = drugsdata.iloc[i, 0]
    pdrug = drugsdata.iloc[i, 1]
    indexofID = allpatientIDs.index(pid)
    indexofDrug = allDrugs.index(pdrug)
    data[indexofID][indexofDrug] = 1

for i in range(len(patientsdata)):
    pid = patientsdata.iloc[i, 0]
    if pid in allpatientIDs:
        indexofID = allpatientIDs.index(pid)
        if patientsdata.iloc[i, 1] == 'Female':
            data[indexofID][-2] = 1
        elif patientsdata.iloc[i, 1] =='Male':
            data[indexofID][-2] = 2
        data[indexofID][-1] = allage.index(patientsdata.iloc[i, 2])

        if patientsdata.iloc[i, 3] <= 4000:
            label[indexofID][0] = 0
        elif patientsdata.iloc[i, 3] <=10000:
            label[indexofID][0] = 1
        else:
            label[indexofID][0] = 2
        
        if patientsdata.iloc[i, 8] <= 2000:
            label[indexofID][1] = 0
        elif patientsdata.iloc[i, 8] <=5000:
            label[indexofID][1] = 1
        else:
            label[indexofID][1] = 2
        label[indexofID][2] = allstatus.index(patientsdata.iloc[i, 5])
        label[indexofID][3] = alllocations.index(patientsdata.iloc[i, 4])
        label[indexofID][4] = alltypes.index(patientsdata.iloc[i, 6])
        label[indexofID][5] = allstaytypes.index(patientsdata.iloc[i, 7])

print("save as file")
outputpath = os.path.join(root, 'data.npz')
np.savez(outputpath, data)
outputpath = os.path.join(root, 'label.npz')
np.savez(outputpath, label)