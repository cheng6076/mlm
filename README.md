#RNN Language Model with Sentence Boundary#

This is an attempt of implementing a RNN language model with sentence boundary, which means the context is only limited to previous words in the same sentence. The memory will not be carried over to the next sentence.
I mask sentences with different lengths to compute the exact perplexity. 

Without any hyper-parameter tuning, the model gives perplexity around 112 on test set. 

