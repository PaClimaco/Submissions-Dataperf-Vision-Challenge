# Author: Paolo Climaco
# Author email: climaco@ins.uni-bonn.de

import numpy as np
from operator import itemgetter
import torch
from dgl.geometry import farthest_point_sampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from torch.utils.data import Dataset, DataLoader

#------------------------------------------
# PATHS
#-----------------------------------------

embeddings_path = 'dataperf-vision-selection/data/embeddings/train_emb_256_dataperf.parquet'
examples_path = 'dataperf-vision-selection/data/examples/'
output_path = 'output_repo/' #Change
machine_human_labels_path =  os.path.join(output_path ,"oidv6-train-annotations-human-imagelabels.csv") #DO NOT CHANGE


#------------------------------------------
# FUNCTIONS
#-----------------------------------------
def download_csv_from_url(url, local_filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded and saved as: {local_filename}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")

def csv_to_json(csv, json_file_path):
    data = []

    data = csv
    dictio ={}
    for i , c in zip(data['ImageID'], data['Confidence']):
        dictio[i]= int(c)

    # Write JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json.dumps(dictio, indent=4))

def read_parquet(path: str):
        return pd.read_parquet(path, engine="pyarrow", use_threads=True)



def get_labelled_samples( task = None, mhl = None, embeddings = None, examples_path = None):

    #load examples
    examples_df =pd.read_csv(os.path.join(examples_path, f'alpha_example_set_{task}.csv'))
    # get label name examples
    LabelName = examples_df['LabelName'][0]

    mydf = mhl[mhl['ImageID'].isin(embeddings['ImageID'])]
    labelled_samples_df = mydf[mydf['LabelName']== LabelName]
    labelled_samples_pd = pd.DataFrame(embeddings[embeddings['ImageID'].isin(labelled_samples_df['ImageID'])])
    labelled_samples_pd.loc[:,'Confidence'] = np.asarray([labelled_samples_df[labelled_samples_df['ImageID'] == image]['Confidence'].item() for image in labelled_samples_pd['ImageID']])
    labelled_samples_vect = np.stack(labelled_samples_pd['embedding'].to_numpy())
    #return only those with confidence 1
    return labelled_samples_vect[labelled_samples_pd['Confidence']==1], labelled_samples_pd[labelled_samples_pd['Confidence']==1]

#------------------------------------------
# DOWNLOAD EMBEDDINGS
#-----------------------------------------

print("Loading embeddings")
embeddings = read_parquet(embeddings_path)
embeddings_vect = np.stack(embeddings["embedding"].to_numpy())
print("Embeddings loaded")



#------------------------------------------
# LOAD MACHINE-HUMAN LABELS
#-----------------------------------------

try:
   mhl = pd.read_csv(machine_human_labels_path)
except FileNotFoundError:
    print('loading human-verified labels...')
    url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv"
    download_csv_from_url(url, machine_human_labels_path)
    mhl = pd.read_csv(machine_human_labels_path)


confidence1_Cupcake_vect, confidence1_Cupcake_pd = get_labelled_samples( 'Cupcake', mhl, embeddings, examples_path)
confidence1_Sushi_vect, confidence1_Sushi_pd = get_labelled_samples( 'Sushi', mhl, embeddings, examples_path)
confidence1_Hawk_vect,  confidence1_Hawk_pd = get_labelled_samples( 'Hawk', mhl, embeddings, examples_path)


#------------------------------------------
# FARTHEST POINT SAMPLING: it takes around 17 minutes but it is a one time effort. SAVE AND LOAD
#-----------------------------------------

try:
    idx_fps= np.load(os.path.join(output_path ,'FPS.npy'))

except FileNotFoundError:
    print('FPS running...it takes a few minutes but it is a one-time effort')
    t_emb = torch.from_numpy(embeddings_vect).unsqueeze_(0)
    idx_fps= farthest_point_sampler(t_emb, 1000,start_idx = 381478 ).numpy()[0].tolist()
    np.save(os.path.join(output_path ,'FPS.npy'), idx_fps)
    idx_fps = np.asarray(idx_fps)
#------------------------------------------
# TRAIN FNN
#------------------------------------------

#fix seed for reproducibility
torch.manual_seed(23)
#associate labels with classes
y_cup =  np.zeros(len(confidence1_Cupcake_vect)) #cupcake associated with label 0
y_sus = np.ones(len(confidence1_Sushi_vect)) #sushi associated with label 1
y_hawk =  np.ones(len(confidence1_Hawk_vect))*2 #hawk associated with label 2

#set training set
x_train = np.vstack( (confidence1_Cupcake_vect[:100], confidence1_Sushi_vect, confidence1_Hawk_vect))
y_train = np.hstack( (y_cup[:100], y_sus[:], y_hawk[:]))
x_train =  x_train.astype(np.float32)
y_train = y_train.astype(np.int64)

# Define the dimensions of the input data and the number of classes
input_dim = 256
num_classes = 3

# Define a custom dataset for the training data
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.softmax(self.fc4(x), dim=1)
        return x

# Create an instance of the model
model = Net()

# Load the model

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a custom dataset and dataloader for the training data
train_dataset = CustomDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 150
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



#------------------------------------------
# LABELLING SET SELECTED WITH FPS
#------------------------------------------
# set set to be labelled 
x_test = embeddings_vect[idx_fps].astype(np.float32)


# Evaluate the model on the test data
with torch.no_grad():
    outputs = model(torch.from_numpy(x_test))
    _, predicted = torch.max(outputs.data, 1)

predicted = predicted.numpy()



#------------------------------------------
# CREATE csv and .json FILES
#------------------------------------------
submission = pd.DataFrame(embeddings.iloc[idx_fps]["ImageID"])
 
for task in ['Cupcake', 'Hawk', 'Sushi']:
        confidence = np.zeros(len( submission), dtype= int)
        if task == 'Cupcake':
            confidence[predicted== 0] = 1
            submission['Confidence'] =confidence.tolist()
        elif task == 'Sushi':
            confidence[predicted== 1] = 1
            submission['Confidence'] =confidence.tolist()  
        elif task == 'Hawk':
            confidence[predicted== 2] = 1
            submission['Confidence'] =confidence.tolist()    
        submission.to_csv(os.path.join(output_path,f'{task}_fps_nn.csv'), index=False)
        csv_to_json(submission, os.path.join(output_path, f'{task}_fps_nn.json') )