# Plant-disease-classification
For this project we set out to tackle plant disease classification problem using three models (ResNet18, MobileNetV3, InceptionV3). we train this 3 models from scratch on 3 seperate datsets which are split into train,validation and test. we evaluate the performace of these models on the datasets and then perform transfer learning on 2 of these trained models. Finally using grid search, we perform hyperparemeter tuning one one of the model in an attempt to improve model's accuracy.

# Requirements
This entire project is done with Pytorch, below are the packages required to run this project

Packages  
------------- | 
Pytorch  |
MatplotLib  | 
Numpy  | 
Seaborn  | 
TorchVision  | 

# Training models
in order to train the models, first ensure you have all the packages installed, ensure you have already downloaded the datasets. replace the path dataset with the path where you have the dataset. then simply run the cells. it is recommended you used GPU for quicker processing time

# Using already provided pretrained models
Each model has it's own folder, in order to use the pre-trained model on the provided datasets, simply load the model `model = models.resnet18(weights=None)`
`model.load_state_dict(torch.load('/content/drive/My Drive/resnet18Dataset1_model.pth'))` assuming you want to test ResNet18 trained on dataset1.
replace the path with the actual path where the pre-trained model is
Next fomr the sample dataset simply load the data for dataset1 and evalute with the model.
`true_labels_test = []
predicted_labels_test = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels_test.extend(labels.tolist())
        predicted_labels_test.extend(predicted.tolist())
accuracy = accuracy_score(true_labels_test, predicted_labels_test)
print(f"Testing Accuracy: {accuracy}")`

# Downloading the Dataset
for the datsets, simply visit the following links and click download
* [Dataset 1]([https://pages.github.com/](https://www.kaggle.com/datasets/goelvanshaj/plant-disease-classification-dataset)).
* [Dataset 2]([https://pages.github.com/]([https://www.kaggle.com/datasets/goelvanshaj/plant-disease-classification-dataset](https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset))).
* [Dataset 3]([https://pages.github.com/]([https://www.kaggle.com/datasets/goelvanshaj/plant-disease-classification-dataset](https://www.kaggle.com/datasets/amandam1/healthy-vs-diseased-leaf-image-dataset))).
