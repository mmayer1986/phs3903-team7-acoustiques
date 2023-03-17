# # ---------- PARTT 0 ----------------- packages to install
# pip install h5py matplotlib --no-index
# pip install torchvision torch tensorboard --no-index
#Restart kernel after installation if using jupyter


# ------------------- PARAMETERS ---------------------
batch = 1
shuff = False
learning_rate = 1e-5
epochs = 100
lam1 = 0
sample_rate = 20.8e6
pixel_dilation = 1


# ----------PART 1 ------------------    Boot up important packages
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter





# -------------PART 2 --------------- Load dictionary into a dataset of tensors

#We need to include the actual images we want to be created. First let's make a function.
#This function converts a signal_number into a 125x125 nd array
#These two functions are lower in OG code

#def ImageArray 1px version
# def ImageArray(signal_number, dataset):
#     length = len(dataset)
#     zero_mat = np.zeros([length,1]) #create a vector array of zeros
#     zero_mat[signal_number, 0] = 1  # places the signal at max value
#     a = zero_mat.reshape(int(length**0.5), int(length**0.5))   # reshape to proper size
#     a2 = a[::-1]            # this reverses order, python places the value opposite to matlab
#     #aten = torch.from_numpy(a2.copy()).float()
#     #aten.view(1,125,125)
#     #print(aten.size())
#     return a2

def ImageArray(signal_number, dataset_of_tensors, n=2):
    length = len(dataset_of_tensors)
    zero_mat = np.zeros([length,1]) #create a vector of zeros
    zero_mat[signal_number, 0] = 1  # places the signal at max value
    a = zero_mat.reshape(int(length**0.5), int(length**0.5))   # reshape to proper size
    a2 = a[::-1]                        # this reverses order, python places the value opposite to matlab
    px = np.where(a2==1)
    
    
    a2[px[0][0]:px[0][0]+n, px[1][0]:px[1][0]+n] = 1 # 21 x 21
    a2[px[0][0]:px[0][0]+n, px[1][0]-n:px[1][0]] = 1
    a2[px[0][0]-n:px[0][0], px[1][0]-n:px[1][0]] = 1
    a2[px[0][0]-n:px[0][0], px[1][0]:px[1][0]+n] = 1
    
    return a2





#Let's start by loading the necessary files

#The next code automates the filename location
#Be sure the file schmerow.py file is in the working directory (it can be any file in that directory)


directory = os.path.dirname(os.path.abspath("schmerow.py"))


#Include the dictionary path here
filename = directory + '/DataAbstractLight.h5'

#Load it with the h5py package (don't forget to close the file to save memory!)
f = h5py.File(filename, 'r')
#f is now the whole .h5 file that is loaded. To see what is in the .h5, you can list the keys

list(f.keys()) ## choose appropriate group
FullDict = f.get('DictQuartz')


#Notice the Dictionary has two mor groups, pull out the RF data, specifically he center angle, which is 3 in pythonbut 4 in matlab!
FullDict
list(FullDict.keys())

#Make the RF variable into the necessary torch tensor
RF = FullDict.get('DataRF')
RF = RF[:,3,:] ## this makes a 2D matrix, the middle row are the 7 angles

#Let's normalize RF
for i in range(len(RF)):
    RF[i] = RF[i]/np.linalg.norm(RF[i])

#RF = np.transpose(RF) ## transpose if needed
#For this mod list, I will do all three dimensions in one fell swoop: the time, the label and the images
RF2 = []
image_list2 = []
NumList = []


##Select the image that is repeated
#ImageNum = [1060, 2310, 4560, 4820, 6010, 6360,  7530, 9850, 10060, 12200 ]
ImageNum = [6360, 6360, 6360, 6360, 6360, 6360, 6360, 6360, 6360, 6360] #made to have a single without changing too much code


i = 0
while i < len(RF):
    #print("i= ", i)
    j=0
    while j < 10:
        RF2.append(RF[ImageNum[j]])
        image_list2.append(ImageArray(ImageNum[j], RF, pixel_dilation))
        NumList.append(ImageNum[j])
        i += 1
        j += 1
        if i == len(RF):
            break


print("done!")


    
image_list2 = np.array(image_list2)
image_list2 = torch.from_numpy(image_list2.copy()).float()    

NumList = np.asarray(NumList)
NumList = NumList.astype(int) #convert to int
NumList = torch.tensor(NumList) #convert to tensor

        
        
RF2 = np.asarray(RF2)
    
RF = torch.from_numpy(RF).float()
RF2 = torch.from_numpy(RF2).float()

#To include the labels with this new tensor we need to create a dataset! 
#Let's first create the label tensor I will make a tensor of integers to indicate location

float_label_array = np.linspace(0, len(RF) - 1, len(RF)) #makes array of floats
int_label_array = float_label_array.astype(int) #convert to int
torch_labels = torch.tensor(int_label_array) #convert to tensor




image_list = []
for i in range(len(RF)):
    image_list.append(ImageArray(i, RF, pixel_dilation))
image_list = np.array(image_list)
image_list = torch.from_numpy(image_list.copy()).float()


#Use subset for now, choose square number
# sbst = 30**2
# RF = RF[0:sbst]
# torch_labels = torch_labels[0:sbst]
# image_list = image_list[0:sbst]

#Now we create the training dataset, put labels in second row, features in first
training_data = TensorDataset(RF,torch_labels, image_list)


#Let's create a new training_data set with the same image everywhere




#RF2 = np.asarray(RF2)
#print(type(RF2))
#RF2 = torch.from_numpy(RF2).float()
    
training_data = TensorDataset(RF2,NumList, image_list2)    

    
#test_data = ... to be completed

#close the file
f.close()

#clear unnecessary info
del RF
del torch_labels
del image_list
del float_label_array 
del int_label_array 


del RF2
del NumList
del image_list2

#------------------------ PART 3 ---------------------------------------------------
# Code to display an image of the pixel location in dictionary. Tweaked to match mat lab
# At the moment, all center pixels are 5x5 instead of 1 x 1 to help with learning

# Code to display an image of the pixel location in dictionary. Tweaked to match mat lab
# Works okay but needs a bit of fixing on top edge and corners


# def DisplayImage(signal_number, dataset_of_tensors):
#     length = len(dataset_of_tensors)
#     zero_mat = np.zeros([length,1]) #create a vector of zeros
#     zero_mat[signal_number, 0] = 1  # places the signal at max value
#     a = zero_mat.reshape(int(length**0.5), int(length**0.5))   # reshape to proper size
#     a2 = a[::-1]                        # this reverses order, python places the value opposite to matlab
#     plt.imshow(a2, cmap = "gray")
#     return a2


#21x21 for center pixel,
def DisplayImage(signal_number, dataset_of_tensors, n = 2):
    length = len(dataset_of_tensors)
    zero_mat = np.zeros([length,1]) #create a vector of zeros
    zero_mat[signal_number, 0] = 1  # places the signal at max value
    a = zero_mat.reshape(int(length**0.5), int(length**0.5))   # reshape to proper size
    a2 = a[::-1]                        # this reverses order, python places the value opposite to matlab
    px = np.where(a2==1)
    
    a2[px[0][0]:px[0][0]+n, px[1][0]:px[1][0]+n] = 1 # 21 x 21
    a2[px[0][0]:px[0][0]+n, px[1][0]-n:px[1][0]] = 1
    a2[px[0][0]-n:px[0][0], px[1][0]-n:px[1][0]] = 1
    a2[px[0][0]-n:px[0][0], px[1][0]:px[1][0]+n] = 1

    plt.imshow(a2, cmap = "gray")
    return a2



# Code to display the time signatures. 
def TimeSig(signal_number, samplingrate, dataset_of_tensors):
    x = np.linspace(0, len(dataset_of_tensors[0][0])-1, len(dataset_of_tensors[0][0])).tolist()
    for i in range(len(x)):
        x[i] = x[i]*(len(dataset_of_tensors[0][0])/samplingrate)

    y = dataset_of_tensors[signal_number][0].tolist()

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel("Temps (microsecondes)")
    plt.show()



#The following code allows you to view a random image
rand = torch.randint(len(training_data), size=(1,)).item() #get random int

img, label, img2 = training_data[rand]  #get img and label from the dataset

print("Displaying time signature number "+str(label.tolist()))


smplrt = sample_rate

plt.title('Signal number '+str(label.tolist()))
DisplayImage(label.tolist(), training_data, pixel_dilation)
TimeSig(label.tolist(), smplrt, training_data)



## Data must be prepped for use with data_loader
batch = batch

train_dataloader = DataLoader(training_data, batch_size = batch, 
    shuffle = shuff)

#No test data yet
#test_dataloader = DataLoader(test_data, batch_size = batch, shuffle = True)

## Display a single image and label of the data_loader. 
# Since shuffle = True, they should change
train_features, train_labels, train_images = next(iter(train_dataloader))
train_features = train_features.view(batch,1, len(train_features[0]))
print(f"Feature batch shape: {train_features.size()}")
print("Displaying time signature number of first item ", train_labels[0].tolist())
plt.title('Signal number '+str(train_labels[0].tolist()))
DisplayImage(train_labels[0].tolist(), training_data, pixel_dilation)
TimeSig(train_labels[0].tolist(), smplrt,  training_data)




#------------------------ PART 4 -----------------------------
## Data must be prepped for use with data_loader
batch = batch

train_dataloader = DataLoader(training_data, batch_size = batch, 
    shuffle = shuff)

#No test data yet
#test_dataloader = DataLoader(test_data, batch_size = batch, shuffle = True)

## Display a single image and label of the data_loader. 
# Since shuffle = True, they should change
train_features, train_labels, train_images = next(iter(train_dataloader))
train_features = train_features.view(batch,1, len(train_features[0]))
print(f"Feature batch shape: {train_features.size()}")
print("Displaying time signature number of first item ", train_labels[0].tolist())
plt.title('Signal number '+str(train_labels[0].tolist()))
DisplayImage(train_labels[0].tolist(), training_data, pixel_dilation)
TimeSig(train_labels[0].tolist(), smplrt,  training_data)




##WARNING, this box initializes model

# ------------ Part 5 ----------------- Prep CPU/GPU, create models
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## Create your network; this network is good for an x = 3072, 1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.n_chans = 64
        self.n_chans_2d = 1  #set to 1 if no 2D conv
        
        self.reflective = nn.ReflectionPad1d(1013) 
        #Note: Padding depends on number of dilated layers. Reduction of each layer is length - 2d where d = 2**dilation. For 9 layers, that is 2046 and 2046/2 = 1023
        
        self.cnn1 = nn.Sequential(
        nn.Conv1d(1, self.n_chans , kernel_size = 3, dilation = 2**0, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
    
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**1, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**2, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**3, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**4, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**5, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**6, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
            
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**7, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
            
        nn.Conv1d(self.n_chans, self.n_chans , kernel_size = 3, dilation = 2**8, padding = 1),
        nn.BatchNorm1d(self.n_chans),
        nn.ReLU(inplace = False),
        #nn.MaxPool1d(2, stride = 2),
        
        nn.Conv1d(self.n_chans, int(self.n_chans) , kernel_size = 3, dilation = 2**9, padding = 1),
        nn.BatchNorm1d(self.n_chans), #Removed the int(self.n_chas/2) and replaced with a MaxPool in first layer
        nn.ReLU(inplace = False),
        nn.MaxPool1d(2, stride = 2),
        
#         nn.Conv1d(1, self.n_chans , kernel_size = 3, dilation = 2**10, padding = 1),
#         nn.BatchNorm1d(self.n_chans),
#         nn.ReLU(inplace = False),
#         #nn.MaxPool1d(2, stride = 2),
        
#         nn.Conv1d(1, self.n_chans , kernel_size = 3, dilation = 2**11, padding = 1),
#         nn.BatchNorm1d(self.n_chans),
#         nn.ReLU(inplace = False),
#         #nn.MaxPool1d(2, stride = 2),
        
        )

        self.linearlayers = nn.Sequential(
        #24576 if you use 1d Conv, 3072 if you skip 1D conv
        nn.Linear(int(3072/2)*self.n_chans,  15625), #3072/2 because of final max pool
        nn.ReLU(inplace = False)    
        #nn.Linear(120, 84),
        #nn.Linear(1024, 10),
        )

       
        self.cnn2 = nn.Sequential(
        nn.Conv2d(self.n_chans_2d, self.n_chans_2d , 3, padding = 1),
        nn.BatchNorm2d(self.n_chans_2d),
        nn.PReLU(),
        #nn.MaxPool2d(2, 2),
        nn.Conv2d(self.n_chans_2d, self.n_chans_2d , 5, padding = 2),
        nn.BatchNorm2d(self.n_chans_2d),
        nn.PReLU(),            
        #nn.MaxPool2d(2, 2),
        nn.Conv2d(self.n_chans_2d, 1 , 7, padding = 3),
        #nn.ReLU(inplace = True),
        #nn.MaxPool2d(2, 2),
        )
        

    def forward(self, x):
        x = self.reflective(x)
        x = self.cnn1(x)
        #print("1: ", x.size()) ## these prints are here to check the size
        x = x.view(x.size(0),-1)
        #print("2: ", x.size())
        x = self.linearlayers(x)
        #print("3: ", x.size())
        x = x.view(x.size(0), self.n_chans_2d, 125, 125)
        #print("4: ", x.size())
        # x = self.cnn2(x)
        #print("5: ", x.size())
        return x #torch.sigmoid(x) 
    
model = CNN().to(device)
print(model)

# --- Load Model


#The process for loading a model includes re-creating the 
# model structure and loading the state dictionary into it.

model = CNN()
model.load_state_dict(torch.load("ERUI_10BatchShuff_5px_2022_08_08.pth"))
model.cuda() #if using a gpu

##------------------------PART 6--------------------------

#hyperparameters

learning_rate = learning_rate
batch_size = batch
epochs = epochs
lam1 = lam1 #Hyperparameter associated with L1 


# Initialize loss function (L1 will be added directly in function)
loss_fn = nn.MSELoss()


# Select function as optimization algorithm (like stochastic gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# This function returns a tensor where each row is the sum of the last two dimensions (i.e.: it is made to retain batch length but sum the matrices in for each entry)
# def L1(tensor_of_images):
#     sum1 = torch.sum(tensor_of_images, dim = 1, keepdim=False)
#     sum2 = torch.sum(sum1, dim = 1, keepdim=False)
#     return sum2

#Everything is implemented in functions that loop the optomization
#def train_loop(dataloader, model, loss_fn1, loss_fn2, optimizer):
def train_loop(dataloader, model, loss_fn, optimizer, lam):
    size = len(dataloader.dataset)
    mat_size = int(size**0.5)
    for batch, (X, y, z) in enumerate(dataloader):
        
        
        if device == "cuda":
            X, y, z = X.cuda(), y.cuda(), z.cuda() # add this line
        #Compute prediction and loss
        X = X.view(len(X),1 ,len(X[0])) #Note, len(X) gives length of batch size and len(X[a]) where 0<a<batch_size gives the length of time signal
        z = z.view(len(X), 1, mat_size, mat_size)
        # X = X.view(len(X),1,len(X[1])) #Note, len(X) gives length of batch size and len(X[a]) where 0<a<batch_size gives the length of time signal
        # z = z.view(len(X), 1, mat_size, mat_size)
        pred = model(X)
        #print(torch.amax(pred))
        loss = loss_fn(pred, z) # This needs to be fixed so that L1 is on whole image

        #Here we add the L1 loss of whole image
        loss = loss + lam*torch.sum(pred)


        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
        #Accuracy
         
        
        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>7f}]/{size:>5d}")
            print("Max Value: ", torch.amax(pred.detach()))
            print("Biases: ", torch.unique(list(model.linearlayers.children())[0].bias.grad))
            
    return loss.item()

#Fix test loop when time comes
# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#         test_loss /= num_batches
#         correct /= size
#         print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#Let's create the datasaver used to save the information collected at each epoch, these can be loaded to continue from where we left off

def save_checkpoint(model, optimizer, save_path, EPOCH, LOSS):
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, save_path)



#and to load:
def load_checkpoint(model, optimizer, load_path, loss):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss


#------------- Part 7 ----------------------------------------

#Loops for the epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------")
    #The following line does the looping per epoch and bounces final loss value to "loss" variable
    loss = train_loop(train_dataloader, model, loss_fn, optimizer, lam1)
    #test_loop(test_dataloader, model, loss_fn)
     
    print()
    #Each epoch is saved here
    save_checkpoint(model, optimizer,"ERUI_3px.pt", t+1 , loss)
    print(f"successfully saved Epoch {t+1}")
    print()

    #The loss value of epochs are also saved as runs for tensorboard
    writer = SummaryWriter()
    writer.add_scalar('Loss', loss, t+1)

writer.close()
print("Done!")

#---------------------------Part 8 ------------------------ Save and recall
#A common way to save a model is to serialize the internal 
# state dictionary (containing the model parameters).

torch.save(model.state_dict(), "ERUI_3px_1img_100eps_Aug_10_2022.pth")
print("Saved PyTorch Model State to model.pth")
print()

#The process for loading a model includes re-creating the 
# model structure and loading the state dictionary into it.

# model = CNN()
# model.load_state_dict(torch.load("MNIST_CNN.pth"))

#This model can now be used to make predictions.




#---------------------------Part 9 ------------------------ #Test model

# Code to view         
# train_features, train_labels, train_images = next(iter(train_dataloader))
# train_features = train_features.view(batch,1, len(train_features[0]))
# train_features = train_features.cuda()
# predicted = model(train_features)
# first = predicted[0,0,:,:]
# first = first.cpu().detach().numpy()
# plt.imshow(first,cmap='gray')

# plt.figure()
# control = train_images[0,:,:]
# plt.imshow(control,cmap='gray')