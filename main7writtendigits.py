# 10/30/2025

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

# 1. Define a transform to convert the images to Tensors
transform = transforms.ToTensor()

# 2. Download and load the training data
#    download=True will download it to the 'data' folder if it's not already there.
train_data = datasets.MNIST(
    root="data",         # Where to store the data
    train=True,          # Get the training set
    download=True,       # Download it if not present
    transform=transform  # Apply the transformation
)

# 3. Download and load the test data
test_data = datasets.MNIST(
    root="data",
    train=False,         # Get the test set
    download=True,
    transform=transform
)

print(f"Training data length: {len(train_data)}")
print(f"Test data length: {len(test_data)}")

# You can now use this with a DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Get one batch of images and labels
images, labels = next(iter(train_loader))

print(f"Shape of one batch of images: {images.shape}") # [batch_size, color_channels, height, width]
print(f"Shape of one batch of labels: {labels.shape}")



# define neural network
class HiMom(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),

        )

    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
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









    #X = torch.tensor(some_data,dtype=torch.float32).to(deviceName)
    #X = torch.randn(2, 28, 28).to(deviceName)
    X = images.to(deviceName)
    logits=model(X)
    pred_probab=nn.Softmax(dim=1)(logits)
    y_pred=pred_probab.argmax(1)



    # train it!
    # --- START of Training Loop ---
    print("\n--- Starting Training ---")
    # typically 5
    num_epochs = 5 # How many times to go over the entire training dataset

    # Set the model to "training mode"
    model.train() 

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Loop over the training data in batches
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

            # Print a progress update every 200 batches
            if (i + 1) % 200 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    # --- END of Training Loop ---


    # save it
    # --- AFTER your training loop ---
    model_filename = "model_weights.pth"
    full_model_path = os.path.abspath(model_filename)
    
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to: {full_model_path}")



    print(f"pred_probab: {pred_probab}")
    print(f"And my prediction is... {y_pred}")

    print(test_data)

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

import matplotlib.pyplot as plt
# same misclassified images
misclassified_images=[]

# evaluate the model
print("Testing model...")
# Tell PyTorch we don't need to calculate gradients, which saves memory and speeds up
with torch.no_grad():
    for test_case in test_data:
        # test_case is a tuple (image_tensor, label)
        image = test_case[0].to(deviceName) # Get the image and send to device
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
        img_data = image.squeeze()

        # Create a figure and axis for this single image
        fig, ax = plt.subplots(figsize=(1.5, 1.5)) # Small figure size for single image
        
        ax.imshow(img_data, cmap='gray')
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

