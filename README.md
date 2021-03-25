# PommeNews - ArticleClassifier

### What's this project about?

Long time ago, I had the idea to code a simple app regrouping news from various french tech websites, with a stronger focus on news about Apple. The app is partially implemented and its development stopped when came the idea to classify articles by theme with Machine Learning.


### iOS Application - PommeNews

The app itself is rather basic. Nothing much to say about it. On another repos.


### Article Classifier

Several applications exists for the classification:

- A tagger used to manually classify articles, a very tedious work. This application is also used to verify and accept predictions for unclassified articles.
- An article processor, written in Swift, called from Python and whose main task is to lemmatize texts. Why in Swift? I couldn't find a lemmatizer doing a good enough job in French, which Apple provide a pretty decent one in the new Natural Language framework.
- The machine learning part, written in Python and using Keras from Tensorflow 2.

#### Classifier Training

##### Manual classification

It is not so easy to classify manually articles. 

For instance, among the articles extra below, which are of theme 'iPhone'?

1. "A portable USB-C charger for smartphone for only 49€." 
- "A portable USB-C charger made for iPhone for 49€"
- "iPhone developers wants Apple to reduce the AppStore fee"
- "The new iPhone 12 will be release next week"
- "iOS 12 will bring a total redesign of the user interface"
- "The game XXX is now available on the AppStore for iOS and iPadOS."

What about smartphone? iPad? Tablet? Not so obvious. One might say only the article 4 is about the iPhone, but another will argue and say that the news about iOS 12 concerns iPhones too. A third person can very well say that all of those articles is of theme 'iPhone'. 

That's the issue I encountered while classifying article manually. Depending on the day, I might classify one article as iPhone while another I would not.

Initially, I chose to classify anything that might be related to iPhone as being of theme 'iPhone'. I started to classify that way. The idea behind was that I can then classify articles according to those other themes: "iOS", "apps", "games", "accessories" and can then easily build a "super-theme" build out from those broad themes. For example, the super-theme "iPhone hardware" could be classified as "iPhone" but not "accessory" and not "iOS".

On the road I gradually changed my mind, thinking that first, to classify all those theme accurately, I would need much more data. Combining themes to build super-themes, also amplifies misclassifications... I therefore though to classify more precisely, but it is still not so easy. It would lead to even more imbalanced data.
In total, I classified about 3000 to 4000 thousands articles

##### Manually classification - ML assisted

Manual classification is hard and takes enormous amount of time. To ease the process I trained the classifier with limited tagged data, I then used the classifier on unseed and yet tagged articles and finally approved or disapproved the predicted themes. For the approval, I wrote an utility showing articles with a specific unverified predicted themes. This allowed to speed-up the manual classification.

That utility is only showing articles for which a specific theme is predicted, while others are not. As long as the accuracy is above 50%, this means the number of articles of that specific theme being verified outnumbered those which are not of that theme. This process was slightly **re-balancing** the data.

##### Training


- I went with one theme at a time, since I was having issues to run with the multi-label classification and it was also easier to access it quality. 
- I tried to simplify the text by replacing some expression by simplified more general ones, such as `iOS 10, iOS 11.3, etc` by `iOS`, `iphone, iphone 5s, iphone Xr` by `iPhone`, etc. I couldn't notice any improvement
- The cross-validation was not implemented at all, I didn't reach result good enough to get there.
- I classifier about 3000 articles by hands (out of 15000 I collected), which . The dataset is rather unbalanced. (For each theme, less than 15-20% of the article quality to it). 2% of 3000 is 600. I split the data 60/15/25 (train/validation/test). For the validation there is about 65 articles of a theme and 360 for the training. 
- Accuracy was about 80-85% while the recall for about 0.6-0.7. So not great.



#### What did I learn?

- A good introduction to Tensorflow.
- Get more familiar with Python.
- Learn how to apply machine learning on text.
- Overview of the Machine Learning and Natural Language Apple frameworks.


#### Why did I stop?

- Manual classification is slow, boring and it not teaching me anything.
- I cannot get enough labelled articles to get good enough data. Not without spending a few weeks of classifying myself thousands of articles.
- Wanted to try Reinforcement learning.