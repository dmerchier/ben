# Training a Neural Network to Play

This is a tutorial describing how to train a neural network to play. 

What you will need:
- make sure that you have successfully [installed](https://github.com/lorserker/ben/blob/main/README.md#installation) the bridge engine
- the data is contained in the file `play_data.zip`
- scripts to transform the data into the binary format expected by the neural network are `*_binary.py`
- scripts which trains neural networks are `*_nn.py`
- scripts which continue to train an existing neural network are `*_nn_continue.py`

### Instructions

You need to be located in the `scripts/training/player_models` directory when you execute the following steps.

First, activate the environment

```
conda activate ben
```

Unzip the `play_data.zip`. You should have `play_data_train.txt` and `play_data_val.txt` files now which contains information of games like this:
```text
W:KQJT8.KT7653.A.8 7653.A.4.KJ96432 A42.Q.QT9762.AQT 9.J9842.KJ853.75
E ALL 4S.=.W
1D PP 1H 3C PP PP 4S PP PP PP
HAHQH2H6D4D7D5DAH3S5SAH4S2S9SQS3H7S6CTH8S7S4D3SKHKC2D6H9HTC3D2HJDKSTC4D9C8C6CQC7DQD8H5CKCAC5SJCJS8C9DTDJ
W:A93.QJ87.543.T94 6.96532.K876.J85 K52.AK.AQJ92.632 QJT874.T4.T.AKQ7
S - 2S.=.S
1S PP PP 2D 2S PP PP PP
D3D8DJDTHKH4H8H5HAHTH7H9C3C7C4C8S6S2SJSAHQH6C6S4SQS3D6SKDAS8D4D7STS9H2S5CAC9C5C2CKCTCJD2S7D5H3DQCQHJDKD9
...
```

Run the script to transform the data into binary format. (the first argument is the dataset for training, the second argument is the one for validating and the third is the script output.)

```
mkdir -p binary/declarer models/declarer

python declarer_binary.py play_data_train.txt play_data_val.txt binary/declarer/
```

The above command will create four new files into the `binary/bidding` folder: `X_train.npy`, `Y_train.npy`, `X_val.npy` and `Y_val.npy`. `X_*.npy` contains the inputs to the neural network and `Y_*.npy` contains the expected outputs. All are stored in numpy array format.
Then, run the training script. This will take several hours to complete, but it will save snapshots of the model as it progresses. If you have a GPU, the training will run faster, but not much faster, because GPUs are not so well suited for the type of NN used.

```
python declarer_nn.py binary/declarer/ models/declarer/
```

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, edit the [code here](https://github.com/lorserker/ben/blob/main/src/nn/models.py#L21) inserting the path to the network which you just trained.

You can do the same thing for `dummy`, `lefty` and `righty`.

#### How to continue training an already trained model

This part describes how you can load an already trained model and continue training it (without the training starting from scratch)

Let's say your already trained model is stored in the `models/declarer` folder and you want to continue training it and then store the results to the `models/declarer-bis` folder. You can do this by running the [declarer_nn_continue.py](declarer_nn_continue.py) script.

```
mkdir -p models/declarer-bis

python declarer_nn_continue.py binary/declarer/ models/declarer/decl-1000000 models/declarer-bis
```

As before, you can do the same thing for `dummy`, `lefty` and `righty`.
