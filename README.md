# BS-Robot: Building a Bill Simmons Language Model Using Transfer Learning

I am a longtime fan of the sportswriter/podcaster Bill Simmons, a former ESPN and Grantland writer, and founder of the website The Ringer.  As CEO of The Ringer, Simmons certainly has his hands full, and his previously prolific literary output has slowed to a drip.  Since Simmons has ceased writing almost completely, I decided to build a language model to (hopefully) mimic his writing style.  

I'm also a fan of Jeremy Howard and his fast.ai library, and decided to implement techniques I learned in his fast.ai Deep Learning For Coders courses.  I heavily borrow from his language model lessons to implement my Bill Simmons language model.  

## Data Acquisition

Bill Simmons has written for several different websites over the years, and fortunately, some of his fans on Reddit have compiled links to his articles [here](https://www.reddit.com/r/billsimmons/comments/81fupe/the_mostly_complete_bill_simmons_archives/).  Using the Scrapy framework in Python, I scraped all of his ESPN 2001-2007 columns to give myself data to build the language model. The final dataset included dozens of articles and over one million words.

## A Comparison of Methods-Training From Scratch vs Transfer Learning and Fine Tuning



Using transfer learning with fine tuning to build a language model.

Comparison vs training a model from scratch.
