# RL2020 - final project
## The dataset can be downloaded from `https://www.dropbox.com/s/2i43mosbokbo0nr/dataset.zip?dl=0` that includes 2 folders (after unzipping)
- `dataset` folder: keeps the training datasets
- `dataset_testing` folder: keeps the testing datasets

## Run training code by (you may need to create a folder named `output` to keep the checkpoints and other log files)
```
python train.py --dataset_path=<path to dataset folder> --batch_size=<2> => train with 2-layer FC policy network
```
```
python train_rnn.py --dataset_path=<path to dataset folder> --batch_size=<2> => train with seq2seq RNN policy network
```

## Run testing code by:
```
python train.py --dataset_path=<path to dataset test folder> --testing=True --checkpoint_path=<path to checkpoint> => test with 2-layer FC policy network
```
```
python train_rnn.py --dataset_path=<path to dataset test folder> --testing=True --checkpoint_path=<path to checkpoint> => test with seq2seq RNN policy network
```

## You can contact me at the email: levanduc@snu.ac.kr
