# ImagiNarrate: Building a Narrative with Images and Generated Captions

## Abstract

In this paper, we introduce a new natural language processing (NLP) approach to solve the
problem of visual storytelling that utilizes image features to generate captions and subsequently
develop a coherent story line for the images. By incorporating image features in
the caption generation process, our proposed approach aims to provide a more relevant and
informative description of the images that can be used to build a cohesive and engaging narrative.
We evaluate our model in comparison to the AREL model (Wang et al., 2018) used
for story generation on the basis of traditional metrics like Meteor and Bleu as well as human
evaluation of the generated stories.

This repo is the implementation of our NLP Project "ImagiNarrate: Building a Narrative with Images and Generated Captions".
In the project, we introduce a new natural language processing (NLP) approach to solve the problem of visual storytelling that utilizes image features to generate captions and subsequently develop a coherent story line for the images.

<p align="demo">
<img src="demo1.png">
</p>

## Prerequisites 
- Python 2.7
- PyTorch 0.3

## Usage
### 1. Setup
Clone this github repository recursively: 

```
git clone --recursive https://github.com/Asmita-Chotani/NLP_Paper_Implementation.git ./
```

Download the preprocessed ResNet-152 features [here](https://vist-arel.s3.amazonaws.com/resnet_features.zip) and unzip it into `DATADIR/resnet_features`, where DATADIR is the VIST folder.
Check the file `opt.py` for more options, where you can play with some other settings.

### 2. Story Generation Learning
To train an AREL model, run

```
python train_AREL.py --id AREL --start_from_model PRETRAINED_MODEL
```

Note that `PRETRAINED_MODEL` some saved models. You can use our best model which can be found [here](https://drive.google.com/drive/folders/1HvQ3YBnELZcsvbAI1jCBVM77ewHeRERW?usp=sharing) by saving the model in 'data/save/AREL'.
Check `opt.py` for more information.

### 3. Testing
To test the model's performance, run
```
python train_AREL.py --option test --beam_size 3 --start_from_model BEST_MODEL
```
You can load our best model found [here](https://drive.google.com/drive/folders/1HvQ3YBnELZcsvbAI1jCBVM77ewHeRERW?usp=sharing) by saving the model in 'data/save/AREL' and set that as the BEST_MODEL mentioned in the command.

## Acknowledgements & References
* [VIST evaluation code](https://github.com/lichengunc/vist_eval)
```
@InProceedings{xwang-2018-AREL,
  author = 	"Wang, Xin and Chen, Wenhu and Wang, Yuan-Fang and Wang, William Yang",
  title = 	"No Metrics Are Perfect: Adversarial Reward Learning for Visual Storytelling",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"899--909",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1083"
  git = "https://github.com/eric-xw/AREL.git"
}
```
