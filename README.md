# NLP_Code_Index


## Deep Learning

- ### [meProp (**m**inimal **e**ffort back **prop**agation method for deep learning)](https://github.com/lancopku/meProp)

  Code for “[meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf)”. This work only computes a small subset of the full gradient to update the model parameters in back propagation, leading to a linear reduction in the computational cost. This does not result in a larger number of training iterations. More interestingly, the accuracy of the resulting models is actually improved.  


- ### [meSimp (**m**inimal **e**ffort **simp**lification method for deep learning)](https://github.com/lancopku/meSimp)

  Codes for “[Training Simplification and Model Simplification for Deep Learning: A Minimal Effort Back Propagation Method](https://arxiv.org/pdf/1711.06528.pdf)”. This work only computes a small subset of the full gradient to update the model parameters in back propagation and further simplifies the model by eliminating the rows or columns that are seldom updated. Experiments show that the model could often be reduced by around 9x, without any loss on accuracy or even with improved accuracy.   


- ### [Label Embedding Network](https://github.com/lancopku/label-embedding-network)

  Code for “[Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks](https://arxiv.org/pdf/1710.10393.pdf)”. This work learns label representations and makes the originally unrelated labels have continuous interactions with each other during the training process. The trained model can achieve substantially higher accuracy and with faster convergence speed. Meanwhile, the learned label embedding is reasonable and interpretable. 

## Neural Machine Translation

- ### [Deconv Dec (Deconvolution-Based Global Decoding)](https://github.com/lancopku/DeconvDec)

  Code for “[Deconvolution-Based Global Decoding for Neural Machine Translation](https://arxiv.org/pdf/1806.03692.pdf)”. This work proposes a new NMT model that decodes the
sequence with the guidance of its structural prediction of the context of the target sequence. The model gets very competitive results. It is robust to translate sentences of different lengths and it also
reduces repetition phenomenon. 

- ### [bag-of-words as target for NMT](https://github.com/lancopku/bag-of-words)

  Code for “[Bag-of-Words as Target for Neural Machine Translation](https://arxiv.org/pdf/1805.04871.pdf)”. This work uses both the sentences and the bag-of-words as targets in the training stage, which encourages
the model to generate the potentially correct sentences that are not appeared in the training set. Experiments show the model outperforms the strong baselines by a large margin.

- ### [ACA4NMT (Adaptive Control of Attention for NMT)](https://github.com/lancopku/ACA4NMT)

  Code for “[Decoding History Based Adaptive Control of Attention for Neural Machine Translation](https://arxiv.org/pdf/1802.01812.pdf)”. This model learns to control the attention by
keeping track of the decoding history. The model is capable of generating translation with less repetition
and higher accuracy.     



## Neural Abstractive Summarization 

- ### [LancoSum](https://github.com/lancopku/LancoSum) (toolkit)
  This repository provides a toolkit for abstractive summarization, which can assist researchers to implement the common baseline, the attention-based sequence-to-sequence model, as well as three high quality models proposed by our group LancoPKU recently. By modifying the configuration file or the command options, one can easily apply the models to his own work. 

- ### [Global-Encoding](https://github.com/lancopku/Global-Encoding)

  Code for “[Global Encoding for Abstractive Summarization](https://arxiv.org/pdf/1805.03989.pdf)”. This work proposes a framework which controls the information flow from the encoder to the decoder based on the global information of the source context. The model outperforms the baseline models and is capable of reducing repetition.  

- ### [HSSC (Hierarchical Summarization and Sentiment Classification model)](https://github.com/lancopku/HSSC)

  Code for “[A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification](https://arxiv.org/pdf/1805.01089.pdf)”. This work proposes a model for joint learning of text summarization and sentiment classification. Experimental results show that the proposed model achieves better performance than the strong baseline systems on both abstractive summarization and sentiment classification. 


- ### [WEAN (Word Embedding Attention Network)](https://github.com/lancopku/WEAN)

  Code for “[Query and Output: Generating Words by Querying Distributed Word Representations for Paraphrase Generation](https://arxiv.org/pdf/1803.01465.pdf)”. This model generates the words by querying
distributed word representations (i.e. neural word embeddings) in summarization. The model outperforms the baseline models by a large margin and achieves state-of-the-art performances
on three benchmark datasets. 

- ### [SRB (Semantic Relevance Based neural model)](https://github.com/lancopku/SRB)

  Code for “[Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization](https://arxiv.org/pdf/1706.02459.pdf)”.  This work improves the semantic
relevance between source texts and summaries in Chinese social media summarization by encouraging
high similarity between the representations of texts and summaries. 



- ### [superAE (supervision with autoencoder model)](https://github.com/lancopku/superAE)

  Code for “[Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization](https://arxiv.org/pdf/1805.04869.pdf)”. This work regards a summary autoencoder as an assistant supervisor of Seq2Seq to get more informative representation of source content. Experimental results show that the model achieves the state-of-the-art performances on the benchmark dataset.  

## Text Generation

- ### [Unpaired-Sentiment-Translation](https://github.com/lancopku/Unpaired-Sentiment-Translation)

  Code for “[Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach](https://arxiv.org/pdf/1805.05181.pdf)". This work proposes a cycled reinforcement learning method to realize sentiment-to-sentiment translation. The proposed method does not rely on parallel data and significantly outperforms the state-of-the-art systems in terms of the content preservation.  

- ### [DPGAN (Diversity-Promoting GAN)](https://github.com/lancopku/DPGAN)

  Code for “[DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text](https://arxiv.org/pdf/1802.01345.pdf)”. This work novelly introduces a language-model based discriminator in Generative Adversarial Network. The proposed model can generate substantially more diverse and informative text than existing baseline methods.  


## Dependency Parsing

- ### [Hybrid Oracle for Dependency Parsing](https://github.com/lancopku/nndep)

  Code for “[Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing](https://arxiv.org/pdf/1711.10163.pdf)”. This work uses all the correct transitions for a parsing
state to provide better supervisory signal in loss function. The new parsers outperform the parsers using the traditional oracle in Chinese dependency parsing and can be used to generate different transition sequences for a sentence.   

## Chinese Segmentation

- ### [PKUSeg](https://github.com/lancopku/PKUSeg) (toolkit)

  This repository provides a toolkit for Chinese segmentation. PKUSeg is easy to use and supports word segmentation for different fields. It has greatly improved the accuracy of word segmentation in different fields.


## Chinese Named Entity Recognition
- ### [ChineseNER](https://github.com/lancopku/ChineseNER)

  Code for “[Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8411523)”. This work combines out-of-domain corpora and in-domain unannotated text to improve NER performance in social media. The unified model yields an obvious improvement over strong baselines.
  
## Sequence Labeling

- ### [Multi-Order-LSTM](https://github.com/lancopku/Multi-Order-LSTM)

  Code for “[Does Higher Order LSTM Have Better Accuracy for Segmenting and Labeling Sequence Data?](https://arxiv.org/pdf/1711.08231.pdf)”. This work combines low order and high order LSTMs together and considers longer distance dependencies of tags into consideration. The model is scalable to higher order models and especially performs well in recognizing long entities.   


- ### [Decode-CRF (Decode-based CRF)](https://github.com/lancopku/Decode-CRF)

  Code for “[Conditional Random Fields with Decode-based Learning: Simpler and Faster](https://arxiv.org/pdf/1503.08381.pdf)”. This work proposes a decode-based probabilistic online learning method, This method is with fast training, very simple to implement, with top accuracy, and with theoretical guarantees of convergence.  


## Multi-Label Learning

- ###  [SGM (Sequence Generation Model)](https://github.com/lancopku/SGM)

  Code for “[SGM: Sequence Generation Model for Multi-label Classification](https://arxiv.org/pdf/1806.04822.pdf)”. This work views the multi-label classification task as a sequence generation
problem. The proposed methods not only capture the correlations between labels, but also select the most informative
words automatically when predicting different labels.  



## NLP Applications

- ### [AAPR (Automatic Academic Paper Rating)](https://github.com/lancopku/AAPR)

  Code for “[Automatic Academic Paper Rating Based on Modularized Hierarchical Convolutional Neural Network](https://arxiv.org/pdf/1805.03977.pdf)”. This work builds a
new dataset for automatically evaluating academic papers and proposes a novel modularized hierarchical convolutional neural network for this task.  


- ### [tcm_prescription_generation (Traditional Chinese Medicine prescription_generation)](https://github.com/lancopku/tcm_prescription_generation)

  Code for “[Exploration on Generating Traditional ChineseMedicine Prescriptions from Symptoms with an End-to-End Approach](https://arxiv.org/pdf/1801.09030.pdf)”. This work explores the Traditional Chinese Medicine prescription generation task using seq2seq models.   


## NLP Datasets

- ### [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)

  Data for “[A Discourse-Level Named Entity Recognition and Relation Extraction Dataset for Chinese Literature Text](https://arxiv.org/pdf/1711.07010.pdf)”. This work builds a discourse-level dataset from hundreds of Chinese literature articles for improving Named Entity Recognition and Relation Extraction for Chinese literature text.    

- ### [Chinese-Dependency-Treebank-with-Ellipsis](https://github.com/lancopku/Chinese-Dependency-Treebank-with-Ellipsis)

  Data for “[Building an Ellipsis-aware Chinese Dependency Treebank for Web Text](https://arxiv.org/pdf/1801.06613.pdf)”. This work builds a Chinese weibo dependency treebank which contains 572
sentences with omissions restored and contexts reserved, aimed at improving dependency parsing for texts with ellipsis.    

- ### [Chinese-Abbreviation-Dataset](https://github.com/lancopku/Chinese-abbreviation-dataset)

  Data for “[A Chinese Dataset with Negative Full Forms for General Abbreviation Prediction](https://arxiv.org/pdf/1712.06289.pdf)”. This work builds a dataset for general Chinese abbreviation prediction. The dataset incorporates negative full forms to promote the research in this area.  




