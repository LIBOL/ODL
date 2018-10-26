# Online Deep Learning: Learning Deep Neural Networks on the Fly
An implementation of the Hedge Backpropagation(HBP) proposed in Online Deep Learning: Learning Deep Neural Networks on the Fly 
```
@inproceedings{sahoo2018online,
  title     = {Online Deep Learning: Learning Deep Neural Networks on the Fly},
  author    = {Doyen Sahoo and Quang Pham and Jing Lu and Steven C. H. Hoi},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {2660--2666},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/369},
  url       = {https://doi.org/10.24963/ijcai.2018/369},
}
```

[Link](https://www.ijcai.org/proceedings/2018/369) to publication

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

# Data sets
The data used in our experiments are available at https://drive.google.com/drive/folders/1fNZHK2NYbgfz27PPdSSA6lZTkoFakH28?usp=sharing
# Train HBP on your own data
We provide a sample script in ```src/train.py``` to train HBP on a new dataset. Feel free to modify the code to suit your experiments.
