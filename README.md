# spider-seg-e19005
Πτυχιακή Εργασία Κωνσταντίνος Ανθούλης e19005 2023 2024 <br>
Undergrad Thesis Konstantinos Anthoulis e19005 2023 2024 <br> 
Datatset & Challenge: https://spider.grand-challenge.org/ <br>

## Dependencies
Πριν από την εκτέλεση εντολής, αφού έχετε ανοίξει terminal στο virtual environment της επιλογής σας <br>
Before running the command below, after running terminal on your virtual environment of choice <br>
```
pip install -r requirements.txt
```
πρώτα μπείτε στο αρχείο και βγάλετε το σχόλιο από την εντολή για ανάλογο install GPU/CPU <br>
go in the file and remove the comment on the install you would like to perform depending on GPU/CPU setup <br>

## Script Order 
(final readme will be more polished here) <br>
- dataset_split: split 3D images 80/20 or ratio of choice <br>
splitting the 3D images and not the 2D images ensures series from the same patient are present in either train or test <Br>
- extract_slices: take 3D images apply resampling and get 2D images to train model on <br>
- crop_slices: crop images to ROI as feature extraction/dimensionality reduction (optional if your machine has enough memory)<br>
- tensor_dims: comb through the dataset for tensor data (normalisation one-hot encoding etc) <br>
++ add start_training and continue_training scripts <br>



