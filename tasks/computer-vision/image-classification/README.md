# APTOS 2019 Blindness Detection

1.download data-set
1.1register account in kaggle
1.2go to the page
https://www.kaggle.com/c/aptos2019-blindness-detection/data
1.3join the competition
1.4then you can download data from it.

2.put the data in the directories as the code image_classification.py
train_csv  =pd.read_csv('../input/aptos2019-blindness-detection/train.csv') 
train      = '../input/aptos2019-blindness-detection/train_images/'
test       = '../input/aptos2019-blindness-detection/test_images/'

3.run the code
python image_classification.py 256
