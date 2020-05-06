# German Traffic Sign Recognition using Pytorch

## Requirement

### Environment

- Python 3.x
- Pytorch
- numpy
- matplotlib
- Pillow
- csv

### Data

German Traffic Sign Recognition Benchmark, which can be downloaded [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

## File Included

- *dataset.py* include a class named DataSet.
- *model.py* include a class named Net.
- *train_eval* do the actual training and evaluation.
- *model_64batch.pth* the trained model file.

## How to run

### Training
1. Put files into the same directory with GTSRB.
2. Modify the *train_eval.py*. Replace ```EPOCHS``` with how many epochs you want at the very beginning. Replace path of GTSRB in ```main``` function.
3. Run the *train_eval.py* on GPU.

### Testing
1. Load ```.pth``` file and test.