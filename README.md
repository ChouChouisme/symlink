# symlink
Code for "JBNU-CCLab at SemEval-2022 Task 12: Machine Reading Comprehension and Span Pair Classification for Linking Mathematical Symbols to Their Descriptions", SemEval@NAACL2022 (1st at the all subtasks)

[Paper - (https://aclanthology.org/2022.semeval-1.231/)](https://aclanthology.org/2022.semeval-1.231/)

## Requirements
* [PyTorch](http://pytorch.org/) >= 1.7.1
* pytorch-lightning==0.9.0
* tokenizers==0.9.3
* pandas==1.3.3
* sklearn
* transformers==4.10.2

## Process

1. Environment Setting
```console
pip install -r ./NER/requirements.txt
pip install -r ./RE/requirements.txt
```

2. Infer
```console
bash infer.sh
```

## Model Checkpoint
https://drive.google.com/file/d/1SFm791Z7KW0GAZ45PyIak5xsL_LdVFOb/view?usp=sharing

## References
* [mrc-for-flat-nested-ner](https://github.com/ShannonAI/mrc-for-flat-nested-ner)

## Q&A
If you encounter any problem, leave an issue in the github repo.


## Citation
```bibtex

@inproceedings{lee-na-2022-jbnu,
    title = "{JBNU}-{CCL}ab at {S}em{E}val-2022 Task 12: Machine Reading Comprehension and Span Pair Classification for Linking Mathematical Symbols to Their Descriptions",
    author = "Lee, Sung-Min  and
      Na, Seung-Hoon",
    editor = "Emerson, Guy  and
      Schluter, Natalie  and
      Stanovsky, Gabriel  and
      Kumar, Ritesh  and
      Palmer, Alexis  and
      Schneider, Nathan  and
      Singh, Siddharth  and
      Ratan, Shyam",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.231",
    doi = "10.18653/v1/2022.semeval-1.231",
    pages = "1679--1686",
    abstract = "This paper describes our system in the SemEval-2022 Task 12: {`}linking mathematical symbols to their descriptions{'}, achieving first on the leaderboard for all the subtasks comprising named entity extraction (NER) and relation extraction (RE). Our system is a two-stage pipeline model based on SciBERT that detects symbols, descriptions, and their relationships in scientific documents. The system consists of 1) machine reading comprehension(MRC)-based NER model, where each entity type is represented as a question and its entity mention span is extracted as an answer using an MRC model, and 2) span pair classification for RE, where two entity mentions and their type markers are encoded into span representations that are then fed to a Softmax classifier. In addition, we deploy a rule-based symbol tokenizer to improve the detection of the exact boundary of symbol entities. Regularization and ensemble methods are further explored to improve the RE model.",
}
```

