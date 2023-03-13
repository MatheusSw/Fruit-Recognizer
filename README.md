# Fruit Recognizer

## Goal
Recognize fruits! Because, why not?

## Dataset
It's currently using the [Fruits-360](https://www.kaggle.com/moltean/fruits) dataset

## Challenges
The models I've been trying are doing great validation and training accuracy wise, however, when it comes to extracing features it's getting a little confused, as it'll constantly mistake avocado for Cucumber with 90%+ confidence, my thought is that it's extracting the "texture" as a feature, but not taking into account overral shape and color.

## Directory structure

- `/models` - All trained models are saved here with some information about them on the title.
- `main.py` - Holds the current code for training the model
- `predicter.py` - Run this to predict results on a given image or generator
- `Figures_{}.png` - Val/Train Loss and Accuracy.

## History

Results from running a model, this is the average accuracy and loss.
![](history-figures/Figure_3.png)