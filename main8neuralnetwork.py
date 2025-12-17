# 10/30/2025

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

from main6 import getData

# This class makes your lists look like the MNIST dataset object
class RavdessDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_index=1):
        """
        target_index: Which label to train on?
        0 = Gender
        1 = Emotion (Default)
        2 = Intensity
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_index = target_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Get the image
        img_array = self.images[idx]
        # Convert numpy array to PIL Image so Torchvision transforms work
        img = Image.fromarray(img_array)
        
        # 2. Apply transforms (Resize, ToTensor, etc)
        if self.transform:
            img = self.transform(img)
            
        # 3. Get the specific label we want (e.g., just Emotion)
        # The label is a vector [Gender, Emotion, Intensity]
        # We pick just one so the training loop doesn't get confused
        label = self.labels[idx][self.target_index]
        
        # Return exactly what MNIST returns: (Image Tensor, Single Integer Label)
        return img, int(label)

def get_formatted_train_test_data(target_feature=0):
    # 1. Get raw data from your conversion script
    # Ensure getData() is imported or available in this scope!
    print("Importing raw data...")
    raw_names, raw_data = getData() 
    
    # 2. Parse Labels (Gender, Emotion, Intensity)
    parsed_labels = []
    for name in raw_names:
        parts = name.split("-")
        try:
            # -1 to convert 1-8 to 0-7 (Zero Indexing)
            emotion = int(parts[2]) - 1 
            intensity = int(parts[3]) - 1
            # Even=Female(1), Odd=Male(0)
            gender = 1 if (int(parts[6]) % 2 == 0) else 0
            parsed_labels.append([gender, emotion, intensity])
        except:
            print(f"Skipping malformed file: {name}")

    # 3. Create the Transforms
    # CRITICAL: We resize to 64x64. 
    # MNIST is 28x28, but your spectrograms might be rectangular. 
    # Squishing them to squares (64x64) is standard for simple CNNs.
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # 4. Create ONE big dataset
    full_dataset = RavdessDataset(
        images=raw_data, 
        labels=parsed_labels, 
        transform=train_transform,
        target_index=target_feature
    )

    # 5. Split into Train (for training/CV) and Test (for final eval)
    # MNIST usually has 60k Train, 10k Test (~15% test)
    total_count = len(full_dataset)
    test_count = int(total_count * 0.15) # 15% for final testing
    train_count = total_count - test_count

    print(f"Splitting data: {train_count} Training items, {test_count} Test items")

    train_data, test_data = random_split(
        full_dataset, 
        [train_count, test_count],
        generator=torch.Generator().manual_seed(42)
    )

    return train_data, test_data

import torch

# set up device
print(f"CUDA available? {torch.cuda.is_available()}")
print(f"CPU available? {torch.cpu.is_available()}")

deviceName=""
# set to false if no gpu, set to true if there is gpu
if(torch.cuda.is_available()):
    deviceName="cuda"
elif(torch.cpu.is_available()):
    deviceName="cpu"
else:
    print("No device available!!!")
    quit()
print(f"Using {deviceName}")


import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms


# download MNIST data for training and use
import torch
from torchvision import datasets, transforms

# pyplot
import matplotlib.pyplot as plt

# 1. Define a transform to convert the images to Tensors
transform = transforms.ToTensor()

# 2. Download and load the training data
#    download=True will download it to the 'data' folder if it's not already there.
train_data, test_data = get_formatted_train_test_data(target_feature=0)
#train_data = datasets.MNIST(
#    root="data",         # Where to store the data
#    train=True,          # Get the training set
#    download=True,       # Download it if not present
#    transform=transform  # Apply the transformation
#)


#print(f"type(train_data): {type(train_data)}")
## get 20% (or whatever CV_data_proportion is) out of train_data and put it in a CV_data variable
#number_of_items_to_remove_from_training_data=int(len(train_data)//(1/CV_data_proportion))
#print(f"number_of_items_to_remove_from_training_data: {number_of_items_to_remove_from_training_data}")

from torch.utils.data import random_split

# 1. Calculate the lengths
total_length = len(train_data) # Use len(), not .size
CV_data_proportion=0.2 # 20% for CV
cv_length = int(total_length * CV_data_proportion) # 20% for CV
train_length = total_length - cv_length # The rest for training

# 2. Use random_split to create two new dataset objects
# generator=torch.Generator().manual_seed(42) ensures you get the same split every time you run it
train_subset, cv_subset = random_split(
    train_data, 
    [train_length, cv_length], 
    generator=torch.Generator().manual_seed(42)
)

print(f"Training set size: {len(train_subset)}")
print(f"CV set size: {len(cv_subset)}")

# 3. Now pass THESE into your DataLoaders
#train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
#cv_loader = DataLoader(cv_subset, batch_size=64, shuffle=False)


# 3. Download and load the test data
#test_data = datasets.MNIST(
#    root="data",
#    train=False,         # Get the test set
#    download=True,
#    transform=transform
#)

print(f"Training data length: {len(train_data)}")
print(f"Test data length: {len(test_data)}")

# You can now use this with a DataLoader
from torch.utils.data import DataLoader

# batch size=1 is stochastic gradient descent, too small and unpredictable
# batch size=60000 is batch gradient descent, too large for gpu memory, and can get stick in smaller local minima easier
# batch size=32, 64, 128, 256 is mini-batch gradient descent, much better
train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
cv_loader = DataLoader(cv_subset, batch_size=256, shuffle=False)
#train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Get one batch of images and labels
images, labels = next(iter(train_loader))

print(f"Shape of one batch of images: {images.shape}") # [batch_size, color_channels, height, width]
print(f"Shape of one batch of labels: {labels.shape}")



## define neural network
#class HiMom(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.flatten = nn.Flatten()
#        self.linear_relu_stack = nn.Sequential(
#            nn.Linear(28*28,512),
#            nn.ReLU(),
#            nn.Linear(512,512),
#            nn.ReLU(),
#            nn.Linear(512,10),
#
#        )
#
#    def forward(self, x):
#        x=self.flatten(x)
#        logits=self.linear_relu_stack(x)
#        return logits

# convolutional neural network
class HiMom(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers "see" 2D shapes (lines, curves) instead of just pixels
        self.features = nn.Sequential(
            # Layer 1: Sees edges
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), # makes it work better
            nn.ReLU(),
            nn.MaxPool2d(2), # Shrinks image from 28x28 -> 14x14

            # Layer 2: Sees shapes (loops, corners)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # makes it work better
            nn.ReLU(),
            nn.MaxPool2d(2)  # Shrinks image from 14x14 -> 7x7
        )
        
        # Linear layers decide what the shapes mean (e.g., "Loop + Line = 9")
        self.classifier = nn.Sequential(
            #nn.Flatten(),
            
            # Dropout: Randomly zeroes out neurons during training. 
            # This forces the model to learn robust features, not just memorize pixels.
            # Good to prevent overfitting.
            nn.Dropout(p=0.5),

            # not 64 * 7 * 7
            nn.Linear(16384, 128), # 64 features * 7 * 7 pixels
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output 2 classes (Male/Female)
        )

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16384,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU(),
            nn.Linear(10,2),

        )

    def forward(self, x):
        x = self.features(x) # comment out to turn into linear only and it will work just fine
        x=self.flatten(x)
        #print(f"shape of x: {x.shape}")
        logits = self.classifier(x)
        #logits=self.linear_relu_stack(x)
        return logits

#some_data=[[2,3,4],[3,4,5]]

#model = HiMom().to("cuda")
#model = HiMom().to("cpu")
model = HiMom().to(deviceName)

#skiptraining=True
skiptrainingifpossible=False
#if skiptraining:
#    model.load_state_dict(torch.load("model_weights.pth"))
#    model.eval
#else:
if os.path.exists("model_weights.pth") and skiptrainingifpossible==True:
    print("Loading saved model weights...")
    model.load_state_dict(torch.load("model_weights.pth"))
    print("Loaded saved model weights")
else:
    # Run your entire training loop here
    print("No saved model found, training...")
    # train the model
    # CrossEntropyLoss is the standard "grader" for classification like this
    loss_fn = nn.CrossEntropyLoss()

    # Adam is a popular "teacher" that updates the model's weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)








    # model test before training
    #X = torch.tensor(some_data,dtype=torch.float32).to(deviceName)
    #X = torch.randn(2, 28, 28).to(deviceName)
    X = images.to(deviceName)
    logits=model(X)
    pred_probab=nn.Softmax(dim=1)(logits)
    y_pred=pred_probab.argmax(1)
    #print(f"untrained guesses: {y_pred}")
    labels_on_device=labels.to(deviceName)
    print("")
    print("Untrained model:")
    print(f"num correct?: {sum(y_pred==labels_on_device)}/{len(labels_on_device)}")
    print(f"proportion correct: {sum(y_pred==labels_on_device)/len(labels_on_device)*100}%")



    # train it!
    # --- START of Training Loop ---
    print("\n--- Starting Training ---")
    # typically 5
    num_epochs = 50 # How many times to go over the entire training dataset

    # Set the model to "training mode"
    model.train() 

    # Capture training CV losses for data checking and visualization
    train_losses=[]
    cv_losses=[]

    # Save best CV loss for comparison to save models with lowest CV loss
    best_cv_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Loop over the training data in batches
        train_loss_sum=0
        for i, (images_batch, labels_batch) in enumerate(train_loader):
            # Move the data to the device
            images_batch = images_batch.to(deviceName)
            labels_batch = labels_batch.to(deviceName)

            # 1. Forward pass: Get model's predictions (logits)
            logits = model(images_batch)

            # 2. Calculate the loss (how wrong was it?)
            loss = loss_fn(logits, labels_batch)

            # 3. Backward pass & Optimize
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Calculate new gradients based on the loss
            optimizer.step()      # Update the model's weights

            train_loss_sum+=loss.item()
            # Print a progress update every 200 batches
            if (i + 1) % 200 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        
        
        print(f"Running cross validation check for epoch {epoch+1}")
        ## CV loss list
        #cv_losses_iteration=[]
        # CV loss sum
        cv_loss_sum=0
        # Set the model to evaluation mode
        model.eval()
        # Cross validation loss check loop to prevent overfitting
        for i, (original_cv_images_batch, original_cv_labels_batch) in enumerate(cv_loader):
            # Move the data to the device
            cv_images_batch=original_cv_images_batch.to(deviceName)
            cv_labels_batch=original_cv_labels_batch.to(deviceName)

            # 1. Forward pass: Get model's predictions (logits)
            cv_logits=model(cv_images_batch)

            # 2. Calculate the loss (how wrong was it?)
            cv_loss=loss_fn(cv_logits,cv_labels_batch)

            # 3. Add CV loss to list for means
            cv_loss_sum+=cv_loss.item()
            #cv_losses_iteration.append(cv_loss.item())
        
        
        # Add mean CV loss to CV loss list
        #cv_losses.append(mean(cv_losses_iteration))
        #cv_losses.append((cv_loss/len(cv_loader)).cpu().detach())
        train_losses.append((train_loss_sum/len(train_loader)))
        cv_losses.append((cv_loss_sum/len(cv_loader)))

        print(f"CV loss for epoch {epoch+1}: {cv_losses[epoch]}")
        print(f"train loss for epoch {epoch+1}: {train_losses[epoch]}")

        # Saving logic to save best model with lowest CV loss
        current_cv_loss = cv_losses[epoch]
        
        # Only save if this is the best score we've ever seen
        if current_cv_loss < best_cv_loss:
            best_cv_loss = current_cv_loss
            torch.save(model.state_dict(), "best_gender_model.pth")
            print(f"   > New best model saved! (Loss: {best_cv_loss:.4f})")

        # increments the learning rate scheduler
        scheduler.step()
        


    print("--- Training Finished ---")
    # --- END of Training Loop ---


    # save it
    # --- AFTER your training loop ---
    model_filename = "model_weights.pth"
    full_model_path = os.path.abspath(model_filename)
    
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to: {full_model_path}")



    #print(f"pred_probab: {pred_probab}")
    #print(f"And my prediction is... {y_pred}")

    #print("Test data:")
    #print(test_data)

    # Print graph of CV loss over time
    # Setup the plot
    plt.figure(figsize=(12, 7))
    # Plots the Cross-Validation Loss for this specific size
    #plt.plot(cv_loss, label=f'{n_hidden} Neurons (Acc: {acc:.3f})')
    

    #import numpy as np
    #from sklearn.metrics import balanced_accuracy_score
    #
    #def get_acc(net, X, y):
    #    """
    #    Helper function to calculate balanced accuracy.
    #    Feeds data through the network, thresholds at 0.5, and compares to true labels.
    #    """
    #    # Feedforward all samples
    #    preds_soft = np.apply_along_axis(net.feedforward, 1, X)
    #    # Convert probabilities to binary predictions (0 or 1)
    #    preds_hard = np.where(preds_soft >= 0.5, 1, 0)
    #    return balanced_accuracy_score(y, preds_hard)
    #net=
    #acc = get_acc(net, X_test, y_test)
    #plt.plot(cv_loss, label=f'{n_hidden} Neurons (Acc: {acc:.3f})')
    #plt.plot((cv_loss.cpu()).detach(), label=f'CV loss')
    plt.plot(cv_losses, label=f'CV loss')
    plt.plot(train_losses, label=f'train loss')
    print("printed data:")
    print(cv_losses)
    # CV Loss Plot formatting and plotting
    plt.title('CV Loss Convergence by Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Cross-Validation) (log scale)')
    plt.yscale('log')
    plt.legend(title="Hidden Layer Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

correct=0
total=len(test_data)
#for test_case in test_data:
    #pred_probab=nn.Softmax(dim=1)(logits)
    #y_pred=pred_probab.argmax(1)
    # if the label in the test data 
    # matches the predicted label then
    # add one to correct
# Set the model to evaluation mode (e.g., turns off dropout)
model.eval()

# same misclassified images
misclassified_images=[]

# evaluate the model
print("Testing model...")
# Tell PyTorch we don't need to calculate gradients, which saves memory and speeds up
with torch.no_grad():
    for test_case in test_data:
        # test_case is a tuple (image_tensor, label)
        #image = test_case[0].to(deviceName) # Get the image and send to device
        image = test_case[0].unsqueeze(0).to(deviceName) # Get the image and send to device
        label = test_case[1]               # Get the correct label (just a number)
        
        # 1. Get the model's output (logits) for the single image
        # The image shape is [1, 28, 28], flatten makes it [1, 784]
        logits = model(image) 
        
        # 2. Get the prediction
        # We don't need softmax, just the argmax (the index of the highest logit)
        # torch.max returns (values, indices)
        _, predicted_index = torch.max(logits.data, 1)

        prediction=predicted_index.item()
        
        # 3. Check if the label matches the predicted label
        # .item() gets the number (e.g., 7) out of the tensor (e.g., tensor([7]))
        if predicted_index.item() == label:
            correct += 1
        else:
            misclassified_images.append((image.cpu(),prediction,label))
print(f"Number correct: {correct}/{total}")
accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f} %")


print("\n--- Saving all misclassified images to a new folder with labels ---")

if len(misclassified_images) > 0:
    error_dir = "misclassified_errors_with_labels" # Using a new folder name
    os.makedirs(error_dir, exist_ok=True)

    for i, (image, pred, actual) in enumerate(misclassified_images):
        # Create a unique filename for each image
        filename = f"error_{i:03d}_pred_{pred}_actual_{actual}.png"
        filepath = os.path.join(error_dir, filename)
        
        # Squeeze image from [1, 28, 28] to [28, 28] for plotting
        #img_data = image.squeeze()
        # 1. Remove the batch dimension (squeeze)
        # 2. Swap dimensions: (C, H, W) -> (H, W, C) using permute
        img_data = image.squeeze().permute(1, 2, 0)

        # Create a figure and axis for this single image
        fig, ax = plt.subplots(figsize=(1.5, 1.5)) # Small figure size for single image
        
        ax.imshow(img_data)
        ax.set_title(f"Pred: {pred}\nActual: {actual}", fontsize=10) # Add title with labels
        ax.axis('off') # Hide axes

        # Save the figure to the file
        plt.tight_layout() # Adjust layout to prevent title overlap
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1) # Saves with tight bounding box
        plt.close(fig) # Close the figure to free up memory

    full_dir_path = os.path.abspath(error_dir)
    print(f"Saved {len(misclassified_images)} wrong images in: {full_dir_path}")
else:
    print("No errors to save.")






#calculon huggingface
#classify 2-5 classes
#use accuracy or confusion matrix to test it
# make sure the pipeline is working
# do classification

