# A Unified MRC Framework for Named Entity Recognition 
The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**A Unified MRC Framework for Named Entity Recognition** <br>
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1910.11476)<br>
If you find this repo helpful, please cite the following:
```latex
@article{li2019unified,
  title={A Unified MRC Framework for Named Entity Recognition},
  author={Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1910.11476},
  year={2019}
}
```


## Overview 

The task of NER is normally divided into **nested** NER and **flat** NER depending on whether named entities are nested or not. Instead of treating the task of NER as a sequence labeling problem, we propose to formulate it as a SQuAD-style machine reading comprehension (MRC) task. <br>

For example, the task of assigning the [PER] label to *"[Washington] was born into slavery on the farm of James Burroughs"* is formalized as answering the question *"Which person is mentioned in the text?"*. <br>

By unifying flat and nested NER under an MRC framework, we're able to gain a huge improvement on both flat and nested NER datasets, which achives SOTA results.

We use `MRC-NER` to denote the proposed framework. <br>
Here are some of the **highlights**:

1. *MRC-NER* works better than BERT-Tagger with less training data. 
2. *MRC-NER* is capable of handling both flat and nested NER tasks under a unified framework.  
3. *MRC-NER* has a better zero-shot learning ability which can predicts labels unseen from the training set.  
4. The query encodes prior information about the entity category to extract and has the potential to disambiguate similar classes. 

## Experimental Results on Flat/Nested NER Datasets
Experiments are conducted on both *Flat* and *Nested* NER datasets. The proposed method achieves vast amount of performance boost over current SOTA models. <br>
We only list comparisons between our proposed method and previous SOTA in terms of span-level micro-averaged F1-score here. 
For more comparisons and span-level micro Precision/Recall scores, please check out our [paper](https://arxiv.org/abs/1910.11476.pdf). <br> 
### Flat NER Datasets
Evaluations are conducted on the widely-used bechmarks: `CoNLL2003`, `OntoNotes 5.0` for English; `MSRA`, `OntoNotes 4.0` for Chinese. We achieve new SOTA results on `OntoNotes 5.0`, `MSRA` and  `OntoNotes 4.0`, and comparable results on `CoNLL2003`.

| Dataset |  Eng-OntoNotes5.0 | Zh-MSRA | Zh-OntoNotes4.0 | 
|---|---|---|---|
| Previous SOTA | 89.16 | 95.54  | 81.63 | 
| Our method |  **91.11** | **95.75** | **82.11** | 
|  |  **(+1.95)** | **(+0.21)** | **(+0.48)** | 


### Nested NER Datasets
Evaluations are conducted on the widely-used `ACE 2004`, `ACE 2005`, `GENIA`, `KBP-2017` English datasets.

| Dataset | ACE 2004 | ACE 2005 | GENIA | KBP-2017 | 
|---|---|---|---|---|
| Previous SOTA | 84.7 | 84.33 | 78.31  | 74.60 | 
| Our method | **85.98** | **86.88** | **83.75** | **80.97** | 
|  | **(+1.28)** | **(+2.55)** | **(+5.44)** | **(+6.37)** | 

Previous SOTA:
 
* [DYGIE](https://www.aclweb.org/anthology/N19-1308/) for ACE 2004.
* [Seq2Seq-BERT](https://www.aclweb.org/anthology/P19-1527/) for ACE 2005 and GENIA.
* [ARN](https://www.aclweb.org/anthology/P19-1511/) for KBP2017.  