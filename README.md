# Multiclass-Classifier

This AI Project is a multiclass classifier that distinguishes between handwritten digits `1, 2, 3, and 4`. 

I created a multi-class classification extracting the label (the number) and the associated feature vector, which represented the pixels of the image of the handwritten digits with part of the dataset. 
Then, I used *batch learning* for running the *softmax regression optimizer*. 
After training my model, I tested it on test data. 

I computed the *confusion matrix*, which showed that most predicted labels reflected the actual label. 
However, my results were far from perfect and could be improved through more training data, smaller testing batches, and more training epochs in order to yield more accurate results. 

Through this project, I learned how to use `tensorflow` and the *stochastic gradient descent optimizer* as well as how to manipulate the learning rate and momentum to achieve better results.
