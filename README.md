# Efficient-Data-Quality-Measures-for-High-Dimensional-Classification-Data
Implementation of ["Data Quality Measures and Efficient Evaluation Algorithms for Large-Scale High-Dimensional Data"](https://www.mdpi.com/2076-3417/11/2/472/html).

## Directory tree
  - dataset : Dataset folder

```
Project
  |--- dataset
  |    |--- mnist
  |    |--- cifar10
  |    |--- STL10
  |    |--- ...
  |
  |--- measure.py
  |--- ...
  
```

## Requirements
- numpy
- matplotlib
- scikit-learn
- torch
- torchvision
- tqdm
- scikit-image
- argparse

### Install requirements
```python
pip3 install -r requirements.txt
```

## Run code
### Example
```python
python3 run.py --dataset_name=cifar10 --root=dataset --ratio=0.25 --sampling_count=100 --vec=10
```
