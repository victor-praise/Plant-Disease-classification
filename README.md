# Plant Disease Classification Using CNN

For this project, we tackled the plant disease classification problem using three deep learning models: ResNet18, MobileNetV3, and InceptionV3. We trained these models from scratch on three separate datasets covering various plant species including apple, cherry, chili, grape, potato, and tomato, which were split into training (70%), validation (10%), and test (20%) sets. After evaluating the performance of the models on the datasets, we performed transfer learning on the ResNet18 and MobileNetV3 models. Finally, we conducted hyperparameter tuning on the MobileNetV3 model using grid search in an attempt to improve its accuracy.

## Requirements

The following packages are required to run the code:

Packages  
------------- |
[PyTorch](https://pytorch.org/)
[NumPy](https://numpy.org/)
[Matplotlib](https://matplotlib.org/)
[Seaborn](https://seaborn.pydata.org/)
[Scikit-learn (sklearn)](https://scikit-learn.org/)

## Training and Evaluating The Models

To train and evaluate the models, you can use Google Colab or Anaconda with Jupyter Notebooks. Here's a step-by-step guide:

1. Upload the Notebook

Each of the folders named after a CNN architecture contains a subfolder called `Notebooks`. Simply upload your desired combinations of dataset + architecture onto Google Colab or Jupyter Notebook.

2. Update the Directories

Each `.ipynb` file has a variable called `dataset_dir`. Update this variable to the path of the correct dataset on your local machine or Google Colab.

```sh
dataset_dir = DATASET_PATH
```

Additionally, update the path in the following line to change the location where the trained model should be saved:

```sh
torch.save(model.state_dict(), DESIRED_PATH_TO_SAVE_MODEL)
```

3. Run the Cells

After updating the directories, simply run each of the cells to train and evaluate the models. It is recommended to use a GPU for quicker processing times.

4. Monitor the Training and Evaluation

During the training and evaluation process, you can monitor the following:

- Training loss and validation loss
- Training accuracy and validation accuracy
- Visualization of the training and validation curves
- Evaluation metrics (e.g., precision, recall, F1-score, confusion matrix)

5. Save the Trained Model

Once you're satisfied with the model's performance, save the trained model weights for future use or deployment.

## Running Pre-Trained Models

To run the pretrained models on the provided sample test dataset, follow these steps:

1. Set up the Environment

Follow steps 1 and 2 covered in the previous section.
Upload the Notebook and update the directories to the location of the sample dataset.

2. Load the Pretrained Model

In the line where the model is being initialized, add the following code below it. This is an example for MobileNetV3:

```sh
model = models.mobilenet_v3_large(weights=None)
model.load_state_dict(torch.load('PATH_TO_SAVED_MODEL'))
model = model.to(device)
```

Replace `PATH_TO_SAVED_MODEL` with the actual path to the saved model file.

3. Run the Cells

You should now be able to run the pretrained models on the sample test dataset.

## Link the Datasets
* [Dataset 1](https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset) - 6 Classes
* [Dataset 2](https://www.kaggle.com/datasets/amandam1/healthy-vs-diseased-leaf-image-dataset) - 16 Classes
* [Dataset 3](https://www.kaggle.com/datasets/goelvanshaj/plant-disease-classification-dataset) - 28 Classes
