# WheatPrediction
## Abstract
Wheat is a key ingredient of many of the food items we consume on daily basis. It is a highly essential crop. For any crop, having data on the production of the crop at certain areas during certain times will help the farmer make better decision next time. So plant scientists use images of wheat crops to gather important data. The data includes important factors like size and density of the wheat heads. The first step in order to do this is to identify the wheat heads from images of wheat crops. This project is about using object detection to find the wheat heads in an image of wheat crop. I have used the Faster RCNN ResNet50 model pretrained on COCO data, changed it's head and then trained it on the train dataset of 3373 images of wheat crops. I ran the model on the 10 images as part of the test dataset, printed images with predicted boxes on them. The images are given in the end of this writeup and we can see that the model has produced very good predictions of the wheat heads.

## Problem Statement
The main goal of this project is to predict bounding boxes around the wheat heads in images of wheat crops. We have a train dataset of 3373 images to train the model and a test dataset of 10 images to test the model.

## Related Work
I have always shown passion towards building tools to improve agriculture and have read about many startups working towards creating and analyzing data on agricultural crops. There are many key things like Species Recognition, Yield Prediction, and Disease Detection through which Deep Learning can benefit the field of agriculture. This project is a part of the Yield detection. It predicts the wheat yield. The following are the articles that inspired me to take up this topic: <br />
https://medium.com/sciforce/machine-learning-in-agriculture-applications-and-techniques-6ab501f4d1b5 <br />
https://objectcomputing.com/expertise/machine-learning/machine-learning-in-agriculture <br />
https://blog.dominodatalab.com/bringing-ml-to-agriculture/ <br />
https://iopscience.iop.org/article/10.1088/1748-9326/aae159

## Methodology
### Data:
I downladed the data from a Kaggle competition that was held last year on predicitng bounding boxes around wheat heads. The link to the project is https://www.kaggle.com/c/global-wheat-detection/overview. I uploaded the dataset to my Google Drive and then imported it into Google Colab.

### Understanding the dataset
The training data is in a file called "train.csv". Here is a preview of the first few lines of the data: <br />
![alt text](https://github.com/ruthviksai/WheatDetection/blob/main/train_data.png?raw=true)
The image_id represents the unique ID associated with an image of wheat crop. Width and height represent the dimensions of the image. All images are 1024x1024. bbox represents the dimensions of a bounding box in the image associated with image_id. Each image has multiple bounding boxes and that's the reason there are multiple rows associated with a single image_id. Each row contains only one bounding box. The format of the bounding box is [xmin, ymin, width, height] It has the coordinates of left most point, width and height of the image. Since the images are gathered from various regions across the world, the variable source contains information about where the image is taken from. There are 3373 unique images in the train dataset. I have divided the train dataset into training and validating data as follows: 80% training data and 20% validating data. This resulted in 2698 training images and 675 validating images. Here are 4 exmaple training images with bounding boxes: <br />
![alt text](https://github.com/ruthviksai/WheatDetection/blob/main/train_images.png?raw=true)

### Data transformation
I have used Albumentations python library to transform the training and validation data. Albumentations is a Python library for image augmentation. It is used to increase the quality of the trained models.

### Model
I have used the Faster RCNN ResNet50 model pretrained on the COCO dataset. Then I changed the head of the model to an appropriate predictor. You can import the model and change the head using the following command:
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```
Then I have trained the model for 8 epochs. The training loss has gone down from 0.96 in epoch 1 to 0.76 in epoch 8. The validation loss has gone down from 1.16 in epoch 1 to 1.004 in epoch 8. The losses have come down to an optimal level and thus 8 epochs worked out to produce good results. The plots for both the losses is: <br />
![alt text](https://github.com/ruthviksai/WheatDetection/blob/main/losses.png?raw=true)

### Evaluation/Results
After training the model, I have evaluated the model on the test dataset by running it on test images and predicting bounding boxes for the images. The testing data does not have data regarding the bounding boxes for test images and thus the model predicted bounding boxes have nothing to be compared against. So I couldn't create a numerical estimate of the results. But I have produced test images with predicted bounding boxes and it can be explicitly seen that the model has done a very good job in predicting the bounding boxes: <br />
![alt text](https://github.com/ruthviksai/WheatDetection/blob/main/test_images.png?raw=true)

### Code/Video
The code is present in the file "WheatDetection.ipynb". The video recording of the presentation of project is present in the file "presentation.mp4".

