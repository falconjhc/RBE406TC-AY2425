import os
from tqdm import tqdm
import yaml






dataPathRoot= '/data0/haochuan/'
styleDir = 'CASIA_Dataset/HandWritingData_240Binarized/CASIA-HWDB2.1/'
train_list= '/data-shared/server09/data1/haochuan/Codes/WNet-20240614-NewAttepmt01/FileList/HandWritingData/Char_0_3754_Writer_1001_1300_Cursive.txt'


## Writing
trainGT = '../YamlLists/HW300/TrainGroundTruth-Cursive.yaml'


def readtxt(path):
    res = []
    f = open(path,'r')
    lines = f.readlines()
    for ll in lines:
        res.append(ll.replace('\n',''))
    f.close()
    return res

if __name__ == '__main__':
    dataset = readtxt(train_list)
    iteration_dir = {x.split("@")[3]:[] for x in dataset}
    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        iteration_dir[name] = [os.path.join(os.path.join(dataPathRoot, styleDir), splits[-1]), content, style]     
    print("Save trainset")

    with open(trainGT, 'w') as file:
        yaml.dump(iteration_dir, file, allow_unicode=True)



    # 
