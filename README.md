#RNN Language Model Variations#

This is an attempt of implementing different types of RNN language models. Instead of using sequence segmentation, I use sentence as the context unit.
That is to say, the context is only limited to previous words in the same sentence and the memory will not be carried over to the next sentence.
Sentences with different lengths are masked to compute the exact perplexity. However, from my test, using sentence context does not affect the result much compared to sequence segmentation.

Currently I have implemented
#### Standard LSTM
Perplexity 115

#### Gated Feedback LSTM
A variation of the model described in *Gated Feedback Recurrent Neural Networks (Chung et al., 2015)*. Use 2 or more layers for it.  
Perplexity 107

#### 1D-Grid LSTM
1D version of Grid LSTM with cells on depth dimension as well. The original paper is *Grid LSTM (Kalchbrenner et al., 2015)*, also described in *Depth-Gated LSTM (Yao et al., 2015)*.
Perplexity 112

