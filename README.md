# KafkaOpenNMT
A brief adaptation of Kafka to openNMT translation model

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

NOTE: If you have MemoryError in the install try to use: 

```bash
pip install -r requirements.txt --no-cache-dir
```
Note that we currently only support PyTorch 1.1 (should work with 1.0)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder.
If you want to train on GPU, you need to set, as an example:
CUDA_VISIBLE_DEVICES=1,3
`-world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only.
To know more about distributed training on single or multi nodes, read the FAQ section.

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

### ATTENTION TO MODELS

## Train using GPU

!python train.py -data data/demo -save_model demo-model -world_size 1 -gpu_ranks 0

## Train and save checkpoints model

!python train.py -data data/demo -save_model saves/demo_model -world_size 1 -gpu_ranks 0 -rnn_size 100 -word_vec_size 50 -layers 1 -train_steps 100000 -optim adam -learning_rate 0.001 -save_checkpoint_steps 50

## Preprocess data 

!python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo

## Translate 

!python translate.py -model saves/demo_model_step_45100.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose 
