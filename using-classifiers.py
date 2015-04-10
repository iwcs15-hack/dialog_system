
# coding: utf-8

# Today's session shows how you can build classifiers to respond to the gist or purpose of an utterance.  Our techniques illustrate what's called [supervised machine learning][1]: finding generalizations from labeled examples that we can use to respond corretly to new, unlabeled examples.  
# 
# This session draws heavily on the [NLTK toolkit][2].  It has a bunch of convenient functions for doing supervised machine learning, which is helpful.  But there are lots of other good Python utilities for doing machine learning, especially [scikit-learn][3], which gives access to a really wide variety of methods.  NLTK's learning is more of a greatest hits compilation by comparison, but we're only going to use the simplest possible learning method: [naive Bayes classifiers][4]. What NLTK also has is a bunch of bundled training data: collections of language that have been marked up by hand to indicate the answers to important questions.  In order to do machine learning, we need that kind of training data.
# 
# We're going to focus on two problems that are particularly relevant for a chatbot.  
# - The first is [sentiment analysis][5], which is understanding whether a comment conveys positive or negative information.  The idea is that we'll be able to classify the user's input, and be able to respond appropriately with markers of feedback like _Cool!_ or _Too bad..._  Obviously you don't want to use these the wrong way!
# - The second is *dialogue act tagging*, a problem first explored in [this journal paper][6], which is understand the kind of contribution that somebody is making to a conversation with an utterance.  It's not always easy to tell whether a statement is supposed to give new information or comment on something that's been said before, or whether a question is asking for clarification or proposing a new topic for conversation.  However, there are some cues and patterns in text that we can learn from data, so that our chatbot can respond differently to these different moves.
# 
# We start with the usual invocations of NLTK.
# 
# [1]:http://en.wikipedia.org/wiki/Supervised_learning
# [2]:http://www.nltk.org/
# [3]:http://scikit-learn.org/stable/
# [4]:http://en.wikipedia.org/wiki/Naive_Bayes_classifier
# [5]:http://en.wikipedia.org/wiki/Sentiment_analysis
# [6]:http://www.aclweb.org/anthology/J/J00/J00-3003.pdf
# 

# In[1]:

import nltk, re
from nltk.corpus import movie_reviews


# We'll start with sentiment analysis.  This code has a few sources.  The first is [Chapter 6 of the NLTK Book][1].  This gives a good overview of supervised learning and the interfaces that NLTK offers for dealing with the data.  The second is [a series of blog posts by Jacob Perkins][2].  (I've given you the link to the first post in the series, but they're all linked together.)  Perkins is a much better source about how to represent documents, select features and evaluate the results.  I've also consulted [Chris Potts's tutorial about sentiment analysis][3] which is a nice mix of academic and practical.
# 
# [1]:http://www.nltk.org/book/ch06.html
# [2]:http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
# [3]:http://sentiment.christopherpotts.net/index.html
# 
# My goal here, by the way, is to show you some techniques that are known to have good results and illustrate the process of using machine learning to make decisions in a program.  This is not the whole picture, and there's a lot more to say about how you go about using machine learning carefully and creatively for a new problem.  Some of the shortcuts that I'm using here (like training on all the available data, without saving anything for development and testing) could lead to very bad reslults if we had to tinker to put the program together rather than just reusing techniques that other people have already worked out.
# 
# For this classification problem, we represent a text snippet as a collection of features.  The features in a document list the _informative_ words that _occur_ in the document.  There are a couple choices here that are not obvious but are important.
# - We use _informative_ words so that the learning algorithm and the final classifier do not get confused by irrelevant information.  Noisy features can make it harder for the learner to lock on to what really matters for a decision; it requires more search and more data to be able to find the real correlations in the presence of many unreliable features, so we'll just get rid of the stuff that's not likely to be useful.  We'll measure informativeness statistically, using a $\chi^2$ statistic derived from the prevalence of feature counts among positive or negative examples.  The $\chi^2$ statistic says how unlikely your observed feature counts are to have been derived by chance.  The more unlikely they are, the higher the $\chi^2$ statistic, and the better evidence the feature gives you about the true category of the data.
# - We only have features for the words that _occur_ in the document.  The NLTK book suggests that you have features for the words that occur in the document _and_ for the words that do not occur in the document, but this does not work very well in our setting.  Normally in supervised classification problem, the features are the most meaningful measurements that you can get automatically from your data to make a decision.  If you're diagnosing a disease, for example, you might decide to do a particular blood test and then record whether the outcome of the test is positive or negative.  Observing a word in a review isn't really like this.  You'd like to know whether the reviewer thought the movie was, say, _interesting_ (which is probably a good thing) but that's not the same thing as whether the reviewer actually used the word _interesting_ in the review.  The gap is particularly important in short utterances like you get in a chatbot - normally most words aren't going to be used because the utterances are short, so you don't really get any evidence that the user didn't think something just because they don't use that word.  Naive Bayes models can handle this easily because the probability models apply when you observe any subset of features -- features that you haven't observed can just be ignored in your learning and decision making.
# 
# I have packaged up the reasoning in a function called `compute_best_features`, which we'll use both to build our sentiment analyzer and our dialogue act tagger.

# In[2]:

def compute_best_features(labels, feature_generator, n) :
    feature_fd = nltk.FreqDist()
    label_feature_fd = nltk.ConditionalFreqDist()
    
    for label in labels:
        for feature in feature_generator(label) :
            feature_fd[feature] += 1
            label_feature_fd[label][feature] += 1
 
    counts = dict()
    for label in labels:
        counts[label] = label_feature_fd[label].N()
    total_count = sum(counts[label] for label in labels)
    
    feature_scores = {}
 
    for feature, freq in feature_fd.iteritems():
        feature_scores[feature] = 0.
        for label in labels :
            feature_scores[feature] +=             nltk.BigramAssocMeasures.chi_sq(label_feature_fd[label][feature],
                                            (freq, counts[label]), 
                                            total_count)
 
    best = sorted(feature_scores.iteritems(), key=lambda (f,s): s, reverse=True)[:n]
    return set([f for f, s in best])


# This block of code computes the features and defines a function to extract the features corresponding to a list of words.  You won't want to execute part of this block - it's a coherent unit of code - so it's commented inline.  It takes a little while to run because it's going through the whole corpus, but it's not so slow for right now that it's worth pickling the best_word_list and loading it in later.

# In[3]:

stop_word_file = "stop-word-list.txt"
with open(stop_word_file) as f :
    stop_words = set(line.strip() for line in f)

def candidate_feature_word(w) :
    return w not in stop_words and re.match(r"^[a-z](?:'?[a-z])*$", w) != None

def movie_review_feature_generator(category) :
    return (word
            for word in movie_reviews.words(categories=[category])
            if candidate_feature_word(word))

best_sentiment_words = compute_best_features(['pos', 'neg'], movie_review_feature_generator, 2000)
 
def best_sentiment_word_feats(words):
    return dict([(word, True) for word in words if word in best_sentiment_words])


# We're going to explore a few ways of doing the classification, so we'll put some infrastructure in place.  First, we load in all the data as a `training_corpus` of `(word_list, category)` pairs.  Then, we create a dummy Python class called `Experiment` that will let us package together comparable values made using different instantiations of the features and learning algorithms and play with the results.

# In[4]:

training_corpus = [(list(movie_reviews.words(fileid)), category)
                   for category in movie_reviews.categories()
                   for fileid in movie_reviews.fileids(category)]

class Experiment(object) :
    pass


# Our first experiment uses the `best_word_feats` that we've just computed - it understands the sentiment in the text based on the most informative words that occur.
# 
# Here's the basic strategy for building and using the classifier:
# - Create a list of training pairs for the learner of the form `(feature dictionary, category label)`
# - Train a naive Bayes classifier on the training data
# - Write a feature extractor that will take raw text into a feature dictionary
# - Write a classification function that will predict the sentiment of raw text
# 
# This also takes a moment to run as it scans through the corpus, makes the features, aggregates them into counts, and uses the counts to build a statistical model.  Again, if it bugs you, you could pickle the classifier.

# In[5]:

expt1 = Experiment()
expt1.feature_data = [(best_sentiment_word_feats(d), c) for (d,c) in training_corpus]
expt1.opinion_classifier = nltk.NaiveBayesClassifier.train(expt1.feature_data)
expt1.preprocess = lambda text : best_sentiment_word_feats([w.lower() for w in re.findall(r"\w(?:'?\w)*", text)])
expt1.classify = lambda text : expt1.opinion_classifier.classify(expt1.preprocess(text))    


# NLTK's `show_most_informative_features` method allows you to see what the classifier has learned.  You can see that a big effect comes from adjectives and a few verbs that do express really strong opinions one way or the other.

# In[6]:

expt1.opinion_classifier.show_most_informative_features(20)


# Here's an example of sentiment detection in action.

# In[7]:

expt1.classify("The dinner was outstanding.")


# [Jacob Perkins recommends][1] including particularly important bigrams in the feature representation of each document.  _Bigram_ is a fancy word for two words that occur successively in a document.  NLTK's collocation finder selects bigrams that occur much more frequently than you would expect by chance - this is an indication that the two words together make an idiomatic expression for conveying a single concept that is important to the document.
# 
# Here we repeat the usual pipleline to include 200 useful bigram features on each document.
# 
# [1]:http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/
# 

# In[8]:

def best_bigram_word_feats(words, score_fn=nltk.BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = nltk.BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_sentiment_word_feats(words))
    return d

expt2 = Experiment()
expt2.feature_data = [(best_bigram_word_feats(d), c) for (d,c) in training_corpus]
expt2.opinion_classifier = nltk.NaiveBayesClassifier.train(expt2.feature_data)
expt2.preprocess = lambda text : best_bigram_word_feats([w.lower() for w in re.findall(r"\w(?:'?\w)*", text)])
expt2.classify = lambda text : expt2.opinion_classifier.classify(expt2.preprocess(text))    


# [Perkins reports][1] that the bigram features do lead to a measurable performance improvement.  In particular, adding bigrams improves the recall of negative classification, which means that the classifier is much better at reporting negative reviews that are truly negative when it is able to include some of these complex expressions.  (Conversely, the classifier also improves the precision with which it recognizes positive reviews, which means that the things that it classifies as positive are more likely to actually be positive.)  Probably this is due to the fact that the classifier can now recognize that _not good_ expresses a negative opinion... 
# 
# [1]:http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
# 
# There are a bunch of interesting bigrams as informative features in the new classifier.

# In[9]:

expt2.opinion_classifier.show_most_informative_features(50)


# Now we turn to dialogue act tagging.  NLTK comes with [a collection of text chat utterances that were collected by Craig Martell and colleagues at the Naval Postgraduate School.][1]  These items have been hand annotated with a number of categories indicating the different roles the utterances play in a conversation.  The list of tags appears here.  The best way to understand what the tags mean is to see an example utterance from each class, so running this code also prints out some examples.  The examples also show what the corpus is like -- including the way user names have been anonymized...
# 
# [1]:http://faculty.nps.edu/cmartell/NPSChat.htm
# 

# In[10]:

chat_utterances = nltk.corpus.nps_chat.xml_posts()

dialogue_acts = ['Accept', 
                 'Bye', 
                 'Clarify', 
                 'Continuer', 
                 'Emotion', 
                 'Emphasis', 
                 'Greet', 
                 'nAnswer', 
                 'Other', 
                 'Reject', 
                 'Statement', 
                 'System', 
                 'whQuestion', 
                 'yAnswer', 
                 'ynQuestion']

for a in dialogue_acts :
    for u in chat_utterances :
        if u.get('class') == a:
            print "Example of {}: {}".format(a, u.text)
            break


# This kind of language is pretty different from the edited writing that many NLP tools assume.  Obviously, for machine learning, it hardly matters what the input to the classifier is.  But it does pay to be smarter about dividing the text up into its tokens (the words or other meaningful elements).  So we'll load in [the tokenizer that Chris Potts wrote][1] to analyze twitter feeds.  Some of the things that it does nicely:
# - Handles emoticons, hashtags, twitter user names and other items that mix letters and punctuation
# - Merges dates, URLs, phone numbers and similar items into single tokens
# - Handles ordinary punctuation in an intelligent way as well
# 
# [1]:http://sentiment.christopherpotts.net/tokenizing.html

# In[11]:

from happyfuntokenizing import Tokenizer
chat_tokenize = Tokenizer(preserve_case=False).tokenize


# Now we set up the features for this data set.  The code is closely analogous to what we did with the sentiment classifier earlier.  The big difference is the tokenization and stopword elimination.  Content-free words and weird punctuation bits like `what` and `:)` are going to be very important for understanding what dialogue act somebody is performing so we need to keep those features around!

# In[12]:

def chat_feature_generator(category) :
    return (word
            for post in chat_utterances
            if post.get('class') == category
            for word in chat_tokenize(post.text))

best_act_words = compute_best_features(dialogue_acts, chat_feature_generator, 2000)
 
def best_act_word_feats(words):
    return dict([(word, True) for word in words if word in best_act_words])

def best_act_words_post(post) :
    return best_act_word_feats(chat_tokenize(post.text))


# Here again is the setup to build the classifier and apply it to novel text.  No surprises here.

# In[13]:

expt3 = Experiment()
expt3.feature_data = [(best_act_words_post(p), p.get('class')) for p in chat_utterances]
expt3.act_classifier = nltk.NaiveBayesClassifier.train(expt3.feature_data)
expt3.preprocess = lambda text : best_act_word_feats(chat_tokenize(text))
expt3.classify = lambda text : expt3.act_classifier.classify(expt3.preprocess(text))    


# Here's a little glimpse into what this classifier is paying attention to.

# In[14]:

expt3.act_classifier.show_most_informative_features(20)


# This demonstration wouldn't be complete without an illustration of how to use the classifiers we've created for an actual chatbot.  We've already seen a whole bunch of ways to produce the content of the response -- I won't repeat that here.  What's interesting here is to show how you can use the classification results in coherent ways to shape the course of the conversation.
# 
# The strategy I illustrate here is to have a different response generator for each of the different dialogue act types.  Each response generator gets the input text (that's not used here, but you'd have to use it to make a pattern-matching response or an information-retrieval response like we've seen ealier).  It also gets the recognized sentiment of the input text as an argument, so it can potentially do something different depending on whether the input is recognized as expressing a positive opinion or a negative opinion.
# 
# I store the response generators in a dictionary -- Python doesn't have a `switch` statement like C or Java, but it does have first class functions.  That makes an array of functions the easiest way to choose a range of behavior conditioned on a value from a small set of possibilities (like the set of dialogue acts).  So the basic pattern of a response is to classify the act and sentiment of the input, and then call the response generator for the recognized act with the original text and the recognized sentiment.  
# 
# Obviously, this is just an invitation to take this further....

# In[15]:

def respond_question(text, valence) :
    if valence == 'pos' :
        return "I wish I knew."
    else :
        return "That's a tough question."
    
def respond_other(text, valence) :
    return ":P  Well, what next?"

def respond_statement(text, valence) :
    if valence == 'pos' :
        return "Great!  Tell me more."
    else :
        return "Ugh.  Is anything good happening?"
    
def respond_bye(text, valence) :
    return "I guess it's time for me to go then."

def respond_greet(text, valence) :
    return "Hey there!"

def respond_reject(text, valence) :
    if valence == 'pos' :
        return "Well, if you insist!"
    else :
        return "I still think you should reconsider."
    
def respond_emphasis(text, valence) :
    if valence == 'pos' :
        return '!!!'
    else :
        return ":("
    

responses = {'Accept': respond_other, 
             'Bye': respond_bye, 
             'Clarify': respond_other, 
             'Continuer': respond_other, 
             'Emotion': respond_other, 
             'Emphasis': respond_emphasis, 
             'Greet': respond_greet, 
             'nAnswer': respond_other, 
             'Other': respond_other,  
             'Reject': respond_reject, 
             'Statement': respond_statement, 
             'System': respond_other, 
             'whQuestion': respond_question, 
             'yAnswer': respond_other, 
             'ynQuestion': respond_question}

def respond(text) :
    act = expt3.classify(text)
    valence = expt1.classify(text)
    return responses[act](text, valence)
    


# In[16]:

respond("Everything sucks")


# In[17]:

respond("I've got fantastic news!")


# In[18]:

respond("A hot cup of tea always makes me happy.")


# In[19]:

respond("Did you hear what happened to me?")


# In[20]:

respond("brb")

