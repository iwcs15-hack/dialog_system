
# coding: utf-8

# A basic Python implementation of Eliza.
# 
# Based on [this source code][1].
# 
# [1]:https://www.daniweb.com/software-development/python/code/380743/eliza-aka-therapist-facelift
# 
# Uses a random number generator and the standard Python regular expression code.

# In[1]:

import random
import re


# This chatbot gets all of its content by mirroring phrases that the user has just contributed.  In order to do this meaningfully, the program has to change the perspective of what it says, swapping `I` and `you` for example.  Chatbot programs often use very simple strategies to do something that is approximately correct here.  We'll see a much better way later, using off-the-shelf natural langauge processing tools.
# 
# Here we have a Python dictionary `reflection_of` mapping some words to replacements.  The `translate` function takes a matched pattern, segments it into words using a regular expression, then substitutes reflections wherever they are found.

# In[1]:

reflection_of = {
    "am"    : "are",
    "was"   : "were",
    "i"     : "you",
    "i'd"   : "you would",
    "i've"  : "you have",
    "i'll"  : "you will",
    "i'm"   : "you are",
    "my"    : "your",
    "are"   : "am",
    "you're": "I am",
    "you've": "I have",
    "you'll": "I will",
    "your"  : "my",
    "yours" : "mine",
    "you"   : "me",
    "me"    : "you" }

def translate(this):
    return ' '.join(reflection_of[word] if word in reflection_of else word
                    for word in re.findall(r"[\w']+",this.lower())) 


# The bulk of Eliza is just a set of rules.  Each rule consists of 
# - A regular expression that should be matched against what the user says.  If there is a match, that means that this rule can apply to generate possible responses for the therapist.
# - A list of possible responses.  The therapist picks one of these responses at random.  The responses can have expressions of the form `%n` inside them.  These will be replaced with the translation of group number `n` from the regular expression match.
# This version of Eliza has 39 different rules.  You can imagine that this table could get pretty big and it would potentially lead to more complex and interesting behavior.

# In[3]:

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
      "%1.",
      "I see.  And what does that tell you?",
      "How does that make you feel?",
      "How do you feel when you say that?"]]
]]


# Python's regular expressions make it ridiculously easy to execute the rules.  This `respond` function does it all.  Given a `sentence` from the user, it looks through all the rules, taking the first one for which a match is found.  Then it picks a response at random, replaces the `%n` terms as needed, and returns the result.

# In[4]:

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


# If you want to make an interactive version of Eliza that runs in a shell from a python script, 
# just append the following code to your file:
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
# In the meantime, here are some illustatrations of the `respond` function so you can get a flavor for what Eliza does.

# In[5]:

respond("My mother hates me.")


# In[6]:

respond("How are you?")


# In[7]:

respond("I need a break.")


# Finally, looking ahead, there are some bugs with the `translate` function.  Sometimes it works, but sometimes not.  The problem is that it has no way to recognize the grammatical role of the words in the user's text, so it can't tell whether `you` is the subject of a sentence or the object, or whether `you` or `they` is the subject of the verb `were`.  These kinds of problems require some real AI to fix, but it's actually pretty easy to do!

# In[8]:

translate("my mother hates me")


# In[9]:

translate("i will do anything you ask.")


# In[10]:

translate("the dogs were crazy")


# In[11]:

translate("the dog was crazy")

