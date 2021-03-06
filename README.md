# Improved Weighted Reverse Dictionary

Kevin, Himanshu, Bonnie

## Requirements

- Python 3.x
- Pytorch 1.x
- Others: numpy, tqdm, nltk, gensim, thulac
- torch.cuda.is_available()

## Data

- Data for paper section 3.1 and 3.2: download the English Reverse Dictionary data from the MultiRD paper from [Google Drive](https://drive.google.com/drive/folders/1jeyPE8iGdGUSVJe_6Smr_NzoWfR52f4g?usp=sharing)
- Data for paper section 3.3: For data augmentation process for our data, see ./DataAugScripts/ and ./SynonymScripts/. We are hosting the data at
  [Google Drive](https://drive.google.com/file/d/1-irISSNJ8MdgOy2H3m2UNdtgnNUXEZYt/view?usp=sharing)
- Data for training the Morfessor model used in data augmentation is from [morpho.aalto.fi](http://morpho.aalto.fi/events/morphochallenge2010/datasets.shtml). The trained model is also in the codebase, stored in the file `SynonymScripts/morphemes/model.bin`

put unzipped `data/` in this directory ./

## Run

For section 3.1 and results in table 1:

```
cd code
# run BiLSTM baseline
python main.py -e 15 -v -dr ../runs/mc/

# run Multi Channel baseline
python main.py -e 15 -v -m rsl -dr ../runs/mc/

# run Learning Channels
python main.py -e 15 -v -lc -dr ../runs/v1/

cd ..
python evaluate_result.py -m runs/mc/b
python evaluate_result.py -m runs/mc/rsl
python evaluate_result.py -m runs/v1/b

```

For section 3.2 and results in table 2 and 3:

```
cd codec
# contrastive joint
python main.py -e 15 -v -dr ../runs/cb/ -m b -ls ce

# contrastive one end
python main.py -e 15 -v -dr ../runs/cf/ -m f -ls ce

# contrastive momentum
python main.py -e 15 -v -dr ../runs/cm/ -m m -ls ce

cd ..
python evaluate_result.py -m runs/cb/b
python evaluate_result.py -m runs/cf/f
python evaluate_result.py -m runs/cm/m

```

To run freezing/no freezing, modify the start_step parameter in codec/model (default 20000)

For section 3.3, run the BiLSTM baseline on both data from MultiRD and our data.

For data augmentation process for our data, see ./DataAugScripts/ and ./SynonymScripts/

## Reference

Zhang et al, Multi-channel Reverse Dictionary Model 2019.
[Paper](https://arxiv.org/abs/1912.08441)
[Repo](https://github.com/thunlp/MultiRD)

### Thank you and happy holidays ????
