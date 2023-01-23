import torch
import numpy as np
from torchvision.transforms import ToTensor, Lambda





# Load and evalulate MMS model
# Model class must be defined somewhere
#model = torch.load("./model/st_gcn.kinetics-6fa43f73.pth")
#model.eval()

# Replace last layer with own classifier

# Create dataset

# Create dateset loader

# Create optimiser

# optimizer = optim.ADAM(model.parameters(), lr = 0.001, momentum =0.9)

# Split data into 

# Finetuning
# Freeze all the parameters in the network
#for param in model.parameters():
#    param.requires_grad = False
# replace last layer (this example is resnet)
#model.fc = nn.Linear(512, 10)
# Optimize only the classifier
#optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


    

class ZebData(torch.utils.data.Dataset):
    
    """Class for test data - because data isnt large just load data at init """
    def __init__(self, data_file =None, label_file= None, transform=None,
                 target_transform = None, ideal_sample_no =None, augment = False, shift = False):
        
        if data_file is not None:
            self.data = np.load(data_file)
        
            # catch incorrectly loaded shape
            if self.data.shape[0] ==1 :
                self.data = self.data.reshape(*self.data.shape[1:])

            # catch incorrectly loaded shape
            self.labels = np.load(label_file).astype("int64")
            if self.labels.shape[0] == 1 :
                self.labels = self.labels.reshape(*self.labels.shape[1:])

            # drop junk 0 cluster
            #self.data = self.data[self.labels>0]
            #self.labels = self.labels[self.labels>0]
            
            mapping = {k:v for v, k in enumerate(np.unique(self.labels))}
            for k, v in mapping.items():
                self.labels[self.labels==k] = v 
            
            

            if augment:
                
                self.dynamic_augmentation()

            if shift:
                #move into positive space

                self.data = self.data + np.array([50, 50])
                
        elif data_file is None:
            # data can be added manually later
            self.data = None
            self.labels = None
        
        self.ideal_sample_no = ideal_sample_no
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        behaviour = self.data[idx] 
        label = self.labels[idx]
        
        if self.transform is not None:
            
            if self.transform == "align":
                
                behaviour = self.align(behaviour)
                behaviour = torch.from_numpy(behaviour).to(torch.float32)
            
        
        if self.transform is None:
            behaviour = torch.from_numpy(behaviour).to(torch.float32)
            
        #if self.target_transform is not None:
            #label = self.target_transform(label)
        label = torch.tensor(label).long()
        return behaviour, label
    
    def align(self, bhv_rs):
        # assumes bout already centered - add check if nose node is <
        #first node coords is vector from center 
        nose_node_vector = bhv_rs[0:2, 0, 0]
        nose_node_mag = np.linalg.norm(nose_node_vector)
        center_vector = np.array([0, 1]).reshape(-1, 1) # changed from nose_node_mag
        center_vector_mag = np.linalg.norm(center_vector)
        #cosine rule to find angle between nose node vector and center vector
        #theta  = (np.arccos(np.dot(nose_node_vector.flatten(), center_vector.flatten())/(nose_node_mag*center_vector_mag)))
        #if nose_node_vector[0] < 0:
            #theta = -theta
        #if nose_node_vector[0] < 0:
            #theta = np.arctan2(-nose_node_vector[0], nose_node_vector[1])[0]
            
        #else:
        theta = np.arctan2(nose_node_vector[0], nose_node_vector[1])[0] # using arctan2 this way gives angle from (0, 1)

        #counterclockwise rotation
        if theta < 0 :
            theta = theta + (2 * np.pi) # important to make sure counter rotates properly when given signed angle
            
            
        c, s = np.cos(theta), np.sin(theta)
        # rotation matrix to use to transform coordinate space
        
        
        
        R = np.array(((c, -s), (s, c)))
        #transform with rotation matrix
        bhv_rs[0:2, :] = np.dot(R,  bhv_rs[0:2, :].reshape((2, -1))).reshape(bhv_rs[0:2, :].shape)
        return bhv_rs
    
    def center_all(self, bout, center_node):
        center_node_xy = bout[0:2, :, center_node]

        center_node_xy = center_node_xy.reshape(center_node_xy.shape[0], 
                                                        center_node_xy.shape[1], 
                                                        -1, 
                                                        center_node_xy.shape[2])# reshaping to match shape of main array for subtraction



        centered_bout = bout.copy()        
        centered_bout[0:2] = centered_bout[0:2] - center_node_xy        
        return centered_bout
    
    def center_first(self, bout, center_node):
        center_node_xy = bout[0:2, 0, center_node]
        center_node_xy = center_node_xy.reshape(2, -1, 1)

        center_node_xy = center_node_xy.reshape(center_node_xy.shape[0], 
                                                        center_node_xy.shape[1], 
                                                        -1, 
                                                        center_node_xy.shape[2])# reshaping to match shape of main array for subtraction



        centered_bout = bout.copy()        
        centered_bout[0:2] = centered_bout[0:2] - center_node_xy
        return centered_bout
    
    def pad(self, bout, new_T):

    
        pose = bout
        padding = new_T - pose.shape[1] # difference between standard T =50 and the length of actual sequence
        ratio = padding/pose.shape[1]
        if ratio > 1:
            bhv_pad = np.concatenate((pose, pose), axis = 1)


            for r in range(int(ratio)-1):
                bhv_pad = np.concatenate((bhv_pad, pose), axis = 1)

            diff = new_T - bhv_pad.shape[1]
            bhv_pad = np.concatenate((bhv_pad, pose[:, :diff]), axis = 1)

        elif (ratio<1) & (ratio>0):
            diff = new_T - pose.shape[1]
            bhv_pad = np.concatenate((pose, pose[:, :diff]), axis = 1)
        elif ratio < 0:
            bhv_pad = pose[:, :new_T]

        return bhv_pad
    
    
    # find angle from [0, 1]
    def angle_from_norm(self, coord):
        theta = np.degrees(np.arctan2(coord[0], coord[1])) # arctan must be y then x
        return theta
    
   

    def get_heading_change(self, bout):
        nose = bout[:2, :, 0]
        last_nose = nose[:, -1]
        angle = self.angle_from_norm(last_nose)
        return angle
    
    def dynamic_augmentation(self):
        
        drop_labels = []
        
        augmented_data = []
        augmented_labels = []
        bhv_idx = [] 
        for label in np.unique(self.labels):

            if label > 0:

                filt = self.labels == label
                label_count = self.labels[filt].shape[0]
                label_subset = self.data[filt]

                augmented = np.zeros((self.ideal_sample_no +label_count, *label_subset.shape[1:]))

                if label_count <100: # this drops rare behaviours
                    drop_labels.append(v)

                elif label_count > self.ideal_sample_no:

                    #augmented = label_subset[:ideal_sample_no]
                    print("sample greater than ideal sample no")
                    print(augmented.shape)
                    break

                else:

                    ratio = self.ideal_sample_no / label_subset.shape[0]

                    augmentation_types = 4
                    remainder = int(ratio % augmentation_types)
                    numAug = int(ratio / augmentation_types)

                    # loop through behaviours in subset
                    for b in range(label_subset.shape[0]):
                            bhv = label_subset[b].copy()
                            rotated = self.rotate_transform(bhv, numAug + remainder)
                            jittered = self.jitter_transform(bhv, numAug)
                            scaled = self.scale_transform(bhv, numAug)
                            sheared = self.shear_transform(bhv, numAug)

                            #concatenate 4 augmentations and original
                            augmented = np.concatenate([bhv.reshape(-1, *bhv.shape), rotated, jittered, scaled, sheared])
                            augmented_data.append(augmented)
                            augmented_labels.append(np.array([label]*augmented.shape[0]).flatten())
                            bhv_idx.append(np.array([b]*augmented.shape[0]).flatten())

        self.augmented_data = np.array(augmented_data)
        #self.data = self.augmented_data.reshape((-1, *augmented_data[0].shape[1:]))
        self.data = np.concatenate(self.augmented_data)
        self.augmented_labels = np.array(augmented_labels)
        self.labels = np.concatenate(self.augmented_labels)
        self.bhv_idx = np.concatenate(np.array(bhv_idx))
        
        
    def rotate_transform(self, behaviour, numAngles):
        """ Rotates poses returning a set number of rotated poses.

         # N, C, T, V, M"""

        rotated = np.zeros((numAngles, *behaviour.shape))

        for angle_no in range(numAngles):

                    # random angle between -50 and + 50
                    angle  = (np.random.random(1) * 60) - 30
                    angle = np.radians(angle[0])

                    # rotation matrix to use to transform coordinate space
                    c, s = np.cos(angle), np.sin(angle)
                    R = np.array([[c, s], [-s, c]]) #clockwise


                    # rotate all time points in behaviour by multiplying rotation matrix with behaviour X, Y
                    transformed = np.dot(R,  behaviour[:2].reshape(2, -1)).reshape(behaviour[0:2, :].shape)
                    rotated[angle_no] = behaviour.copy()
                    rotated[angle_no, :2] = transformed

        return rotated


    def jitter_transform(self, behaviour, numJitter):
        """ Adds noise to poses returning a set number of rotated poses.

         # N, C, T, V, M"""

        jittered = np.zeros((numJitter, *behaviour.shape))

        for jitter_no in range(numJitter):


                # random jitter between -5 and +5 pixels
                jitter = (np.random.random(behaviour[:2].shape)*4) - 2

                jittered[jitter_no] = behaviour.copy()
                jittered[jitter_no, :2] = behaviour[:2] + jitter


        return jittered


    def scale_transform(self, behaviour, numScales):
        """ Randomly scales poses"""

        scaled = np.zeros((numScales, *behaviour.shape))

        for scale_no in range(numScales):

            # create random scales between 0 and 3
            scale = (np.random.random((1)) * 3)

            scaled[scale_no] = behaviour.copy()
            scaled[scale_no, :2] = behaviour[:2] * scale

        return scaled


    def shear_transform(self, behaviour, numShears):
        sheared = np.zeros((numShears, *behaviour.shape))

        for shear_no in range(numShears):

            # create random scales between -1.5 and 1.5
            shear_x = (np.random.random((1)) * 2)-1
            shear_y = np.random.random((1)) * 1

            shear_matrix = np.array([[1,  shear_x[0]],
                                     [shear_y[0], 1]])


            transformed = np.dot(shear_matrix,  behaviour[:2].reshape(2, -1)).reshape(behaviour[0:2, :].shape)
            sheared[shear_no] = behaviour.copy()
            sheared[shear_no, :2] = transformed


        return sheared

class HyperParams(object):
    
    def __init__(self, batch_size, lr, dropout):
        self.batch_size = batch_size
        self.learning_rate = lr
        self.dropout = dropout        