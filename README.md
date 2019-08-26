# Smart Wall

## Usage
* Put all.csv file into data/directory.
* Run `$ python data_preprocessing.py` to get X.npy and y.npy under ./data/npy/directory.
* Run `$ python train.py | tee ./log.txt` to start training. Tee will redirect training info to both screen and ./log.txt file.

## Images
- Figures for training visualization is under ./imgs/ directory.

## Tools
There are some useful scripts in tools directory.
* `concat_csv.py` : concat csv files into a single file. 
* `db2csv.sh` : convert .db files to .csv files. Thanks to [convert-db-to-csv](https://github.com/darrentu/convert-db-to-csv).

## Bug Tracking
- Why is the test set accuracy up to 100% after only 1 epoch training? After several days of mental torture, I finally found the answer. The original labels are -1 and 1, and the result of this keras API: keras.utils.to_categorical(y_train, num_classes=2) is wrong. This API can only convert labels starting from zero to one-hot encoding. I mistakenly thought that it can automatically convert -1 and 1 to 0 and 1 corresponding one-hot encoding, consequently causing all labels to be 1. This bug is too difficult to check. So we need to first convert -1 and 1 to 0 and 1, and then convert 0 and 1 to the corresponding one-hot encoding as the final labels. Everything is completely back to normal now!