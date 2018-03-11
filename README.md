# matlab-neural-network

This is a homework for the **Machine Learning and Applications** 2017 class at the **[University of Naples](http://www.unina.it)**.

This exampe uses the **[MNIST](http://yann.lecun.com/exdb/mnist/)** archive but any kind of labeled data-set should do the trick (with some minor modifications). It tries different sizes with either a batch or online approach and goes on with training as long as the validation error is less than the training error.
It comes packed with functions for
+ `classic back-propagation`
+ `gradient-descent`
+ `rprop`

supported error functions are:
+ `sum-of-squared`
+ `cross-entropy`

..but the code is flexible enough to use whatever functions you want to, as long as it respects the signature (and of course, are valid weights update methods or valid error functions).

There's also a pdf file (Italian only!) that goes deeper into specifics of NN and implementation.

# how to use
Download the files, uncompress the 'immagini.mat' 7zip archive and just run the `main` function.
