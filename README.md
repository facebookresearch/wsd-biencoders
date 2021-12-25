# Gloss Informed Bi-encoders for WSD 

This is the codebase for the paper [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://blvns.github.io/papers/acl2020.pdf). 

![Architecture of the gloss informed bi-encoder model for WSD](https://github.com/facebookresearch/wsd-biencoders/blob/main/docs/wsd_biencoder_architecture.jpg)
Our bi-encoder model consists of two independent, transformer encoders: (1) a context encoder, which represents the target word (and its surrounding context) and (2) a gloss encoder, that embeds the definition text for each word sense. Each encoder is initalized with a pertrained model and optimized independently.

## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Pytorch Transformers 1.1.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.

## How to Run 
To train a biencoder model, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python biencoder.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. The model weights and predictions for the biencoder reported in the paper can be found [here](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj).

Similar commands can be used to run the frozen probe for WSD (`frozen_pretrained_encoder.py`) and the finetuning a pretrained, single encoder classifier for WSD (`finetune_pretrained_encoder.py`).

## Citation
If you use this work, please cite the corresponding [paper](https://blvns.github.io/papers/acl2020.pdf):
```
@inproceedings{
  blevins2020wsd,
  title={Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders},
  author={Terra Blevins and Luke Zettlemoyer},
  booktitle={Proceedings of the 58th Association for Computational Linguistics},
  year={2020},
  url={https://blvns.github.io/papers/acl2020.pdf}
}
```

## Contact
Please address any questions or comments about this codebase to blvns@cs.washington.edu. If you want to suggest changes or improvements, please check out the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This codebase is Attribution-NonCommercial 4.0 International licensed, as found in the [LICENSE](https://github.com/facebookresearch/wsd-biencoders/blob/master/LICENSE) file.
