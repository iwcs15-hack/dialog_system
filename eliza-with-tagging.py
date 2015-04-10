
# coding: utf-8

# Version of the Eliza Program with a part-of-speech tagging preprocessor to transform input text with more intelligence.

# In[1]:

import random
import re


# This file takes advantage of Python's `pickle` feature, which allows you to dump an object that you've created to a binary file and then load it in later.
# 
# In this case, we're loading a [part of speech tagger][1], which is a program that takes a list of words (and other tokens from the input sentence) and analyzes them to disambiguate the basic grammatical role that they play in the sentence.
# 
# I built this part of speech tagger using [NLTK][2], which is freely available.  NLTK comes with several taggers built in, but none does the kind of disambiguation you'd want for Eliza, which is why I built it myself.
# 
# I trained the tagger on [the Brown Corpus][3], which has been marked up with relatively rich part of speech tags.  You can see a list of those tags [here][4].  Two things are particularly useful about this tag set.  
# - First, it distinguishes between subject pronouns and object pronouns (using the tags `PPSS` for a subject pronoun and `PPO` for an object pronoun).  This means that you can look at how the word _you_ is tagged to decide whether to swap it with _I_ or _me_. 
# - Second, it indicates whether nouns are common (tags including the symbols `NN`) or proper (tags including the symbols `NP`).  That means you can tell whether a word that's capitalized at the beginning of a sentence deserves to stay capitalized when you repeat it in the middle of a sentence.
# 
# Here is the code that I used to build the tagger and save it to a pickle.
# ```python
# import nltk
# from pickle import dump
# unigram_tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())
# brill_tagger_trainer = nltk.tag.brill_trainer.BrillTaggerTrainer(unigram_tagger, 
#                                                                  nltk.tag.brill.brill24())
# tagger = brill_tagger_trainer.train(nltk.corpus.brown.tagged_sents(), max_rules=1000)
# outfile = open("bbt.pkl", "wb")
# dump(tagger, outfile, -1)
# outfile.close()
# ```
# As you can see, all the work to get this going is pretty much done by NLTK.  But you should know a little bit about what's going on under the hood.  We're using an algorithm called [Brill Tagging][5].  The NLTK book also talks about this in [Section 6 of Chapter 5][6].  This algorithm starts from a simple tagger -- here we're using the `UnigramTagger` that just tags each symbol with its most likely reading, ignoring the surrounding context.  Then we use the `BrillTaggerTraininer` to find a collection of rules that correct mistakes in the default tagging.  These rules can use a variety of features including what words are nearby and what parts of speech are nearby.  We use our training data to search through the possible rules and find the ones that work the best at correcting the errors of the unigram tagger.  This took several hours to run (I'm not sure how long because I didn't stick around to wait).  So it was important to store the result as a pickle for whenever we need it later.  This also has the advantage that the tagger we store no longer depends on NLTK so it's easier to use.  
# 
# [1]:http://en.wikipedia.org/wiki/Part-of-speech_tagging
# [2]:http://www.nltk.org/
# [3]:http://en.wikipedia.org/wiki/Brown_Corpus
# [4]:http://www.comp.leeds.ac.uk/ccalas/tagsets/brown.html
# [5]:http://en.wikipedia.org/wiki/Brill_tagger
# [6]:http://www.nltk.org/book/ch05.html
# 

# In[2]:

from pickle import load
infile = open("bbt.pkl", "rb")
tagger = load(infile)
infile.close()


# Once we load in the pickle as `tagger`, we're in business.  We convert the user's utterance into a list of tokens `L` and call `tagger.tag(L)`.  What we get back is a list of pairs, where the first item is the word and the second item is the part of speech that the tagger inferred for the word.  
# 
# Here's quickie code to split a string up into suitable tokens.  It's a little annoying because of the strange ways we write punctuation in English, and because of the funny way that Python's `split` utility winds up returning lots of copies of the empty string when matching complex conditional patterns.

# In[3]:

def tokenize(text) :
    return [tok for tok in re.split(r"""([,.;:?"]?)   # optionally, common punctuation 
                                         (['"]*)      # optionally, closing quotation marks
                                         (?:\s|\A|\Z) # necessarily: space or start or end
                                         (["`']*)     # snarf opening quotes on the next word
                                         """, 
                                    text, 
                                    flags=re.VERBOSE)
            if tok != '']


# Now we restrict our translation just to the unambiguous cases:

# In[4]:

untagged_reflection_of = {
    "am"    : "are",
    "i"     : "you",
    "i'd"   : "you would",
    "i've"  : "you have",
    "i'll"  : "you will",
    "i'm"   : "you are",
    "my"    : "your",
    "me"    : "you",
    "you've": "I have",
    "you'll": "I will",
    "you're": "I am",
    "your"  : "my",
    "yours" : "mine"}


# We handle the ambiguous cases in a couple of different ways.  First, there are some tokens that we can now map conditionally just depending on the way they were tagged.

# In[5]:

tagged_reflection_of = {
    ("you", "PPSS") : "I",
    ("you", "PPO") : "me"
}


# Here's the code to translate individual tokens, handling capitalization using the proper name key (`NP`) that fits proper name tags.

# In[6]:

def translate_token((word, tag)) :
    wl = word.lower()
    if (wl, tag) in tagged_reflection_of :
        return (tagged_reflection_of[wl, tag], tag)
    if wl in untagged_reflection_of :
        return (untagged_reflection_of[wl], tag)
    if tag.find("NP") < 0 :
        return (wl, tag)
    return (word, tag)


# On the other hand, handling verbs like _are_ and _were_ is more complicated, because the tagger does not indicate whether they have a second person subject like _you_ or a third person plural subject like _they_.  However, we're lucky because in English the subject is usually pretty close to the verb and there's no way to modify words like _you_ so that material intervenes.  So we can just look for the noun phrase that's nearest the verb and use that to infer whether the subject of the verb is one of the pronouns we want to target.

# In[7]:

subject_tags = ["PPS",  # he, she, it
                "PPSS", # you, we, they
                "PN",   # everyone, someone
                "NN",   # dog, cat
                "NNS",  # dogs, cats
                "NP",   # Fred, Jane
                "NPS"   # Republicans, Democrats
                ]

def swap_ambiguous_verb(tagged_words, tagged_verb_form, target_subject_pronoun, replacement) :
    for i, (w, t) in enumerate(tagged_words) :
        if (w, t) == tagged_verb_form :
            j = i - 1
            # look earlier for the subject
            while j >= 0 and tagged_words[j][1] not in subject_tags :
                j = j - 1
            # if subject is the target, swap verb forms
            if j >= 0 and tagged_words[j][0].lower() == target_subject_pronoun :
                tagged_words[i] = replacement
            # didn't find a subject before the verb, so probably a question 
            if j < 0 :
                j = i + 1
                while j < len(tagged_words) and tagged_words[j][1] not in subject_tags :
                    j = j + 1
                # if subject is the target, swap verb forms
                if j < len(tagged_words) and tagged_words[j][0].lower() == target_subject_pronoun :
                    tagged_words[i] = replacement


# There are four cases that we need to deal with: fixing "are", "am", "were" and "was" when we've changed their subjects to violate English agreement.  We also have to fix some punctuation.

# In[8]:

def handle_specials(tagged_words) :
    # don't keep punctuation at the end
    while tagged_words[-1][1] == '.' :
        tagged_words.pop()
    # replace verb "be" to agree with swapped subjects
    swap_ambiguous_verb(tagged_words, ("are", "BER"), "i", ("am", "BEM"))
    swap_ambiguous_verb(tagged_words, ("am", "BEM"), "you", ("are", "BER"))
    swap_ambiguous_verb(tagged_words, ("were", "BED"), "i", ("was", "BEDZ"))
    swap_ambiguous_verb(tagged_words, ("was", "BEDZ"), "you", ("were", "BED"))


# Here we put it all together.  We expand the sentence into tokens, tag the tokens, translate using the tags, deal with the verbs, and then put things back together.   
# 
# Fortunately, the tagger can alert us to the presence of punctuation of various types, so we know where the spaces belong in the output!

# In[9]:

close_punc = ['.', ',', "''"]
def translate(this):
    tokens = tokenize(this)
    tagged_tokens = tagger.tag(tokens)
    translation = [translate_token(tt) for tt in tagged_tokens]
    handle_specials(translation)
    if len(translation) > 0 :
        with_spaces = [translation[0][0]]
        for i in range(1, len(translation)) :
            if translation[i-1][1] != '``' and translation[i][1] not in close_punc :
                with_spaces.append(' ')
            with_spaces.append(translation[i][0])           
    return ''.join(with_spaces)


# The regular expressions are exactly the same as before.  So is the code to generate a response!

# In[10]:

rules = [(re.compile(x[0]), x[1]) for x in [
   ['How are you?',
      [ "I'm fine, thank you."]],
    ["I need (.*)",
    [   "Why do you need %1?",
        "Would it really help you to get %1?",
        "Are you sure you need %1?"]],
    ["Why don't you (.*)",
    [   "Do you really think I don't %1?",
        "Perhaps eventually I will %1.",
        "Do you really want me to %1?"]],
    ["Why can't I (.*)",
    [   "Do you think you should be able to %1?",
        "If you could %1, what would you do?",
        "I don't know -- why can't you %1?",
        "Have you really tried?"]],
    ["I can't (.*)",
    [   "How do you know you can't %1?",
        "Perhaps you could %1 if you tried.",
        "What would it take for you to %1?"]],
    ["I am (.*)",
    [   "Did you come to me because you are %1?",
        "How long have you been %1?",
        "How do you feel about being %1?"]],
    ["I'm (.*)",
    [   "How does being %1 make you feel?",
        "Do you enjoy being %1?",
        "Why do you tell me you're %1?",
        "Why do you think you're %1?"]],
    ["Are you (.*)",
    [   "Why does it matter whether I am %1?",
        "Would you prefer it if I were not %1?",
        "Perhaps you believe I am %1.",
        "I may be %1 -- what do you think?"]],
    ["What (.*)",
    [   "Why do you ask?",
        "How would an answer to that help you?",
        "What do you think?"]],
    ["How (.*)",
    [   "How do you suppose?",
        "Perhaps you can answer your own question.",
        "What is it you're really asking?"]],
    ["Because (.*)",
    [   "Is that the real reason?",
        "What other reasons come to mind?",
        "Does that reason apply to anything else?",
        "If %1, what else must be true?"]],
    ["(.*) sorry (.*)",
    [   "There are many times when no apology is needed.",
        "What feelings do you have when you apologize?"]],
    ["Hello(.*)",
    [   "Hello... I'm glad you could drop by today.",
        "Hi there... how are you today?",
        "Hello, how are you feeling today?"]],
    ["I think (.*)",
    [   "Do you doubt %1?",
        "Do you really think so?",
        "But you're not sure %1?"]],
    ["(.*) friend(.*)",
    [   "Tell me more about your friends.",
        "When you think of a friend, what comes to mind?",
        "Why don't you tell me about a childhood friend?"]],
    ["Yes",
    [   "You seem quite sure.",
        "OK, but can you elaborate a bit?"]],
    ["No",
    [ "Why not?"]],
    ["(.*) computer(.*)",
    [   "Are you really talking about me?",
        "Does it seem strange to talk to a computer?",
        "How do computers make you feel?",
        "Do you feel threatened by computers?"]],
    ["Is it (.*)",
    [   "Do you think it is %1?",
        "Perhaps it's %1 -- what do you think?",
        "If it were %1, what would you do?",
        "It could well be that %1."]],
    ["It is (.*)",
    [   "You seem very certain.",
        "If I told you that it probably isn't %1, what would you feel?"]],
    ["Can you (.*)",
    [   "What makes you think I can't %1?",
        "If I could %1, then what?",
        "Why do you ask if I can %1?"]],
    ["Can I (.*)",
    [   "Perhaps you don't want to %1.",
        "Do you want to be able to %1?",
        "If you could %1, would you?"]],
    ["You are (.*)",
    [   "Why do you think I am %1?",
        "Does it please you to think that I'm %1?",
        "Perhaps you would like me to be %1.",
        "Perhaps you're really talking about yourself?"]],
    ["You're (.*)",
    [   "Why do you say I am %1?",
        "Why do you think I am %1?",
        "Are we talking about you, or me?"]],
    ["I don't (.*)",
    [   "Don't you really %1?",
        "Why don't you %1?",
        "Do you want to %1?"]],
    ["I feel (.*)",
    [   "Good, tell me more about these feelings.",
        "Do you often feel %1?",
        "When do you usually feel %1?",
        "When you feel %1, what do you do?"]],
    ["I have (.*)",
    [   "Why do you tell me that you've %1?",
        "Have you really %1?",
        "Now that you have %1, what will you do next?"]],
    ["I would (.*)",
    [   "Could you explain why you would %1?",
        "Why would you %1?",
        "Who else knows that you would %1?"]],
    ["Is there (.*)",
    [   "Do you think there is %1?",
        "It's likely that there is %1.",
        "Would you like there to be %1?"]],
    ["My (.*)",
    [   "I see, your %1.",
        "Why do you say that your %1?",
        "When your %1, how do you feel?"]],
    ["You (.*)",
    [   "We should be discussing you, not me.",
        "Why do you say that about me?",
        "Why do you care whether I %1?"]],
    ["Why (.*)",
    [   "Why don't you tell me the reason why %1?",
        "Why do you think %1?" ]],
    ["I want (.*)",
    [   "What would it mean to you if you got %1?",
        "Why do you want %1?",
        "What would you do if you got %1?",
        "If you got %1, then what would you do?"]],
    ["(.*) mother(.*)",
    [   "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?",
        "How does this relate to your feelings today?",
        "Good family relations are important."]],
    ["(.*) father(.*)",
    [   "Tell me more about your father.",
        "How did your father make you feel?",
        "How do you feel about your father?",
        "Does your relationship with your father relate to your feelings today?",
        "Do you have trouble showing affection with your family?"]],
    ["(.*) child(.*)",
    [   "Did you have close friends as a child?",
        "What is your favorite childhood memory?",
        "Do you remember any dreams or nightmares from childhood?",
        "Did the other children sometimes tease you?",
        "How do you think your childhood experiences relate to your feelings today?"]],
    ["(.*)\?",
    [   "Why do you ask that?",
        "Please consider whether you can answer your own question.",
        "Perhaps the answer lies within yourself?",
        "Why don't you tell me?"]],
    ["quit",
    [   "Thank you for talking with me.",
        "Good-bye.",
        "Thank you, that will be $150.  Have a good day!"]],
  ["(.*)",
  [   "Please tell me more.",
      "Let's change focus a bit... Tell me about your family.",
      "Can you elaborate on that?",
      "Why do you say that %1?",
      "I see.",
      "Very interesting.",
      "So %1.",
      "I see.  And what does that tell you?",
      "How does that make you feel?",
      "How do you feel when you say that?"]]
]]

def respond(sentence):
    # find a match among keys, last one is quaranteed to match.
    for rule, value in rules:
        match = rule.search(sentence)
        if match is not None:
            # found a match ... stuff with corresponding value
            # chosen randomly from among the available options
            resp = random.choice(value)
            # we've got a response... stuff in reflected text where indicated
            while '%' in resp:
                pos = resp.find('%')
                num = int(resp[pos+1:pos+2])
                resp = resp.replace(resp[pos:pos+2], translate(match.group(num)))
            return resp


# For reference, here's the code to add to make the program interactive again.
# ```python
# if __name__ == '__main__':
#     print("""
# Therapist
# ---------
# Talk to the program by typing in plain English, using normal upper-
# and lower-case letters and punctuation.  Enter "quit" when done.'""")
#     print('='*72)
#     print("Hello.  How are you feeling today?")
#     s = ""
#     while s != "quit":
#         s = input(">")
#         while s and s[-1] in "!.":
#             s = s[:-1]
#             
#         print(respond(s))
# ```
# 

# The responses below illustrate some of the improvements you can get in Eliza's responses using the new translation method.

# In[11]:

respond("My mother hates me.")


# In[12]:

respond("I'm ``possibly,'' maybe crazy.")


# In[13]:

respond("My dog was crazy.")


# In[20]:

respond("I was crazy")


# In[23]:

respond("You said Fred was crazy.")


# In[35]:

respond("I asked you.")

