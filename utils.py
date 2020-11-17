import pickle
from tqdm import tqdm
import numpy as np

# Data save
def save_list(data, path):
    with open(path, "wb") as fp:   #Pickling
        pickle.dump(data, fp)

# Data load   
def load_list(path):
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return b

# Data load
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# For the DCol Library --> input data format is arff file.
def to_arff(dataset_name : str, filename : str, data, label):
    data = data.reshape(data.shape[0], -1)
    num_data, num_feature = data.shape[0], data.shape[1]
    data = data.tolist()
    uniq_label = np.unique(label)
    
    print('Dataset size :', num_data)
    
    f = open(filename, 'w')
    
    relation = '@relation %s\n\n' % dataset_name
    f.write(relation)
    
    for pixel in range(num_feature):
        attribute = '@attribute pixel%d numeric\n' % (pixel + 1)
        f.write(attribute)
        
    f.write('@attribute label {%s}\n\n' % np.array2string(uniq_label, separator=',')[1:-1])
    f.write('@data\n')
    
    for idx, val in tqdm(enumerate(data)):
        val = str(val).replace(' ', '')[1:-1]
        val += ',%d\n' % label[idx]
        f.write(val)

    f.close()