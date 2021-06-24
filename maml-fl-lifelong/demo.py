import os
import numpy as np
import pandas as pd
# from torch._C import int64

root = '../../Dataset/eicu'
data = np.load(os.path.join(root, 'data.npz'), allow_pickle=True)['arr_0'].astype(np.int64)
label = np.load(os.path.join(root, 'label.npz'), allow_pickle=True)['arr_0'].astype(np.int64)
print(data.shape, label.shape)

print(data[0], label[0])