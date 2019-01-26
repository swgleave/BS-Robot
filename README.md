# BS-Robot: Building a Bill Simmons Language Model Using Transfer Learning

I am a longtime fan of the sportswriter/podcaster Bill Simmons, a former ESPN and Grantland writer, and founder of the website The Ringer.  As CEO of The Ringer, Simmons certainly has his hands full, and his previously prolific literary output has slowed to a drip.  Since Simmons has ceased writing almost completely, I decided to build a language model to (hopefully) mimic his writing style.  

I'm also a fan of Jeremy Howard and his fast.ai library, and decided to implement techniques I learned in his fast.ai Deep Learning For Coders courses.  I heavily borrow from his language model lessons to implement my Bill Simmons language model.  I tried training a model from scratch, which is one technique Mr. Howard teaches, and then work on improving the model by using transfer learning and fine-tuning a language model originally trained on a very-large, unrelated corpus.

## Data Acquisition

Bill Simmons has written for several different websites over the years, and fortunately, some of his fans on Reddit have compiled links to his articles [here](https://www.reddit.com/r/billsimmons/comments/81fupe/the_mostly_complete_bill_simmons_archives/).  Using the Scrapy framework in Python, I scraped all of his ESPN 2001-2007 columns to give myself data to build the language model. The final dataset includes dozens of articles and over one million words.  

## A Comparison of Methods-Training From Scratch vs Transfer Learning and Fine Tuning

Language models are widely used, and have applications in speech recognition and sentiment analysis.  While language models have existed for awhile, deep learning has more recently become the defacto way to build language models, with great success.  One way to build a language model is to train a recurrent neural network (usually of the long-short term memory variety) from scratch on a corpus of text.  The text is tokenized, broken up into sequences, and fed into the NN, which learns the probability distribution of these sequences.  

In the last year, [work](https://arxiv.org/abs/1801.06146) has been published by multiple groups, including Jeremy Howard, that describes how transfer learning can be incorporated into Natural Language Process to build a better language model.  While  pre-trained embedding matrices have been used for a few years, pre-training an entire language model and fine tuning on a new data set is not something that has been extensively studied.  Jeremy Howard has pretrained a language model on a very large Wikipedia corpus, and shared this model in his fast.ai course.  

I built two language models, one trained from scratch with the Bill Simmons corpus, and another fine-tuned on the Bill Simmons corpus after being pre-trained by Stephen Merity of Salesforce on the Wikipedia corpus. Below is a comparison of the models.

### Metric comparison

The below plot displays the loss for both models. 

<img src="https://github.com/swgleave/BS-Robot/blob/master/images/Plot%20Loss.png" height="375" width="500">

For the pretrained model, the validation loss quickly reaches its minimum of 3.908 at 5 epochs, and steadily increases, while training loss steadily decreases.  As the majority of the model has be pretrained, it takes very few training epochs for validation loss to reach its minimum.

For the model trained from scratch, both validation loss and training loss are on a downward trajectory throughout the first 50 trainig epochs.  I continued to train this model past the 50 epoch mark, and validation loss hits its minimum value of 4.14 after 58 training epochs and slowly increases afterward. Training loss continues to decrease through the first 200 epochs, which is as far as I trained the model.

Intuitively, it makes sense that the pretrained model would quickly reach a validation minimum, as the weights only need to be tweaked, and not learned from scratch.  The pretrained model ultimately has a better validation loss, with a minimum value of 3.908, compared to the best validation loss of 4.14 from the model trained from scratch.  It appears that the pretraining that was done on a much larger corpus has allowed the model to develop a better understanding of language, driving an improved validation loss.

### Output comparison

As a comparison, I fed the same two seed sentences into both models, and had it sample the next 100 tokens, after training the models.  I've added capitalization to start the sentences, but otherwise am displaying the first predictions made by the model for the seed phrases, near where validation loss is minimized(no cherry picking the good stuff here).  

For seed sentences, I chose one sentence completely unrelated to sports, from A Tale of Two Cities by Charles Dickens, and one from Bill Simmon's most recent written work, his 2019 Trade Value Column. 

Seed sentence 1: "*we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way...*"

Model from scratch:
"*..., they 're only starting to talk about it for three straight weeks . So it 's definitely weird when they won't make a first - fire game . Just that it's lost . It's also the bad branch thing on the network , amazing , as well as another team ready to call him " big play . " Why can't we be complaining about him it? You can't find the Cavs swinging two days in 2006. Maybe Shaq or v - seemed should have killed him because nobody did money . ...*"

Pretrained model:
"*...to watch any hometown coach normally busting his chops and taking 
 five steps from someone else 's table during every game . I don't see how he is on his team anymore . Our first three reasonable yes . 
 ( YES ! YEAH ! ! ! ) 
 BREAK THE ESPY for " best press conference of the night " 
 after jon kitna 's first laugh at the telecast , I called me and explained in the theater , " I enjoyed the actress with her look , I really could...*"


Seed sentence 2: "*Sure, not everything has changed. We still laugh when Knicks fans convince themselves that some A-list free agent is definitely coming. We still joke that...*"

Model from scratch:
"*game 6 of the Warriors - Warriors game in ' 97 ' Ladainian Tomlinson , when Nick Horry is bounced into game 6 of the Bulls - Spurs series in Indiana . I put two highlights on this . Our Celtics Celtics grab high schooler Kareem ( an frustrated if Manny slipped ) , Spree sees the 2006 Sacramento masters Sox graphic , as well as the 2002 eastern finals for Madden ( with the Lakers ) on involved . I'm just waiting . I'm not trying to change the comic written you about this . Shaq should ...*"

Pretrained model:
"*...was hoping for LJ to start screaming at various stiffs . But that's just me . In fact , when Barkley came up red in ' 93 , our generation ended up being slightly more likely to crack our top 50 . Almost like , if you wish people knew it was coming , then they should have ended up telling him you wouldn't want to see that , that was ate up my back in Patriots week 15 , when the Pats won the super bowl in the NFC silence ( we wouldn't get a... *"

It appears that in both models, the model does exhibit a general understanding of the language.  In the outputs seeded with the sentence from the Bill Simmons model, the models see that the Knicks from the NBA are mentioned and continue to mention basketball terms, before changing subjects.  I was especially impressed that the pretrained model mentions LJ, who was a player on the Knicks!  Overall, the language generated by the pretrained model feels slightly more fluid than the trained from scratch model.  The sentences seem generally to have better structure and seem more realistic (although nobody is mistaking this as from a human author).  The pretrained model seemed to generally produce better samples throughout my experimentation.

## Fun With Bill-Generating Additional Sentences

Here are some of my favorite paragraphs the language model generated:

### Topic = Celtics

*celtics ...
...team , which won the title and lasted only six years . It 's also strange how many people have played this game since I was still watching the Lakers . I don't think Peja , Artest , Stephon marbury , Kidd , Clemens or Allen Iverson and Stephon Marbury , where he handles a pipe dream , so everyone knows how to win a game forgetting the difference . You wonder how the Pistons dribble , Vin Baker and Ray Allen , as well as lots of young P.J. Brown 's partner and last year 's $ 50 million expiring deal [ with Indiana ] . at least that ends with Greg Oden helping to equal some Dan Dickau luxury box offers . except one thing 's going to stomach me : the trade Troy Williams , who 's wasted...*



### Topic = Movies

*movies ...
.... here are the relevant points of the movie : 
 I don't know exactly what to say . first of all , it 's not my favorite movie preview , and I don't get it . Honestly , I still moved into the theater 7 - 7 times a few times . Bring back the sports gal , spend a ton of time punching 
 guys , settle down in a pseudo - awful sports movie , lots of lively cameos , cracking big screen actors , always take a cute chick , do some...*

*movies ...
...already scheduled to feature " hot dog , " " Walking Tall , " " 20 minutes , " " Beverly Hills Cop 2 " and " Silent Rage " both of them . Maybe that 's the most confusing moment of my lifetime , I 'm greeted by silence , shocking family members , the earned - saw fights , tons of Knicks fans turning into Fredo--Matt-esque actors , and even the large amount of money that keeps raising my wife 's interest . And yet , it 's like a reality show on national tv...*


### Topic = Red Sox

*Red Sox ...
...players were hung out . So what happened to the players is that the ones who broke the bunch ever completely kissed each other in the post-game festivities ? Why aren't they allowed to transfer to hundreds of organized leagues? Do forget these facts and don't like sportswriters that I owe you a lot ? If the NFL is our own sport , the NBA is fan -tastic . I 'm overmatched . I hope this is it . Seriously . 

 -- Grant Hill 
 
 ohiosg : excellent call*

### Topic = Vegas

*Vegas ... 
...offer unfathomable rehab problems that can only be described as " type of slump . " None of them can . I can't remember them . Dad , I knew he was a free agent . Although this was a sad day for me , disappointed that there wasn't any interest in the slam dunk contest if my father or I had been sitting in his head for the past seven hours . We accepted five people , plus we were expecting something like nine people watching a horse crash . I know if they ever tattoos with one another , then they hang out for three hours , then they 're up by six , if they go away or keep up until they lose the big one .*


## Conclusion

After reviewing example output and plots of the validation loss, it appears that fine tuning a language model on top of pretrained language model produces superior results, when compared with a language model that is trained from scratch (at least for a corpus of this size).  While the model does produce interesting output, in the future I would like to make modifications to better structure the output from the model.  Ideas I have include having the model learn to generate titles for articles, using tags to learn and generate different sections of the article (beginning, middle, or end), or having the model learn to generate specific types of articles (examples include gambling picks or mailbag columns).    



