# BS-Robot: Building a Bill Simmons Language Model Using Transfer Learning

I am a longtime fan of the sportswriter/podcaster Bill Simmons, a former ESPN and Grantland writer, and founder of the website The Ringer.  As CEO of The Ringer, Simmons certainly has his hands full, and his previously prolific literary output has slowed to a drip.  Since Simmons has ceased writing almost completely, I decided to build a language model to (hopefully) mimic his writing style.  

I'm also a fan of Jeremy Howard and his fast.ai library, and decided to implement techniques I learned in his fast.ai Deep Learning For Coders courses.  I heavily borrow from his language model lessons to implement my Bill Simmons language model.  I tried training a model from scratch, which is one technique Mr. Howard teaches, and then work on improving the model by using transfer learning and fine-tuning a language model originally trained on a very-large, unrelated corpus.

## Data Acquisition

Bill Simmons has written for several different websites over the years, and fortunately, some of his fans on Reddit have compiled links to his articles [here](https://www.reddit.com/r/billsimmons/comments/81fupe/the_mostly_complete_bill_simmons_archives/).  Using the Scrapy framework in Python, I scraped all of his ESPN 2001-2007 columns to give myself data to build the language model. The final dataset includes dozens of articles and over one million words.  

## A Comparison of Methods-Training From Scratch vs Transfer Learning and Fine Tuning

Language models are widely used, and have applications in speech recognition and sentiment analysis.  While language models have existed for awhile, deep learning has more recently become the defacto way to build language models, with great success.  One way to build a language model is to train a recurrent neural network (usually of the long-short term memory variety) from scratch on a corpus of text.  The text is tokenized, broken up into sequences, and fed into the NN, which learns the probability distribution of these sequences.  

In the last year, [work](https://arxiv.org/abs/1801.06146) has been published by multiple groups, including Jeremy Howard, that describes how transfer learning can be incorporated into Natural Language Process to build a better language model.  While  pre-trained embedding matrices have been used for a few years, pre-training an entire language model and fine tuning on a new data set is not something that has been extensively studied.  

My goal was to test out both techniques, and see how they compared.  

### Metric comparison

The below plots displays the loss for both models. 

(Insert Plot)













(Describe what happens)



### Output comparison

As a comparison, I fed the same two seed sentences into both models, and had it predict the next 100 words, after training the models.  I've added capitalization to start the sentences, but otherwise am displaying the first predictions made by the model at each training level (no cherry picking the good stuff here).

Note that in this comparison, the trained-from-scratch model has been trained for more epochs than the model using transfer learning.  I did this to give the from-scratch model extra time to "catch up", since the transfer learning model had been pretrained. 

For seed sentences, I chose one sentence completely unrelated to sports, from A Tale of Two Cities by Charles Dickens, and one from Bill Simmon's most recent written work, his 2019 Trade Value Column. 

Seed sentence 1: we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way






Seed sentence 2: Sure, not everything has changed. We still laugh when Knicks fans convince themselves that some A-list free agent is definitely coming. We still joke that


## Fun With Bill-Generating Additional Sentences




## Conclusion


TO DO STILL:

Rewrite code so can be a python script or function.  Make function input be how many epochs it runs for and prints results.


