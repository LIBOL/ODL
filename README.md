# Online Deep Learning: Learning Deep Neural Networks on the Fly
An implementation of the Hedge Backpropagation(HBP) proposed in Online Deep Learning: Learning Deep Neural Networks on the Fly 

IJCAI-ECAI-18

Preprint: [https://arxiv.org/abs/1711.03705](https://arxiv.org/abs/1711.03705).

# Requirements and Installation
- Theano 0.8.2
- Keras 1.2.1

To install HBP, you need to replace the Keras's ```keras/engine/training.py``` file with our modified ```training.py```. this doesn't affect normal projects that don't use HBP.
Note that as the current HBP implementation only supports Keras 1.

# Experiments
- To run HBP on the sample Higgs dataset, first download the data:
```sh
wget -O data/higgs.mat https://www.dropbox.com/s/fvqnhe34cf0mlz9/higgs_100k.mat
```
- To train HBP, run:
```sh
cd src/hbp
python main.py -c hb19.yaml
```

- To train other baseline models, run:
```sh
cd src/baselines
./run.sh
```

# Train HBP on your own data
We provide a sample script in ```src/train.py``` to train HBP on a new dataset. Feel free to modify the code to suit your experiments.
