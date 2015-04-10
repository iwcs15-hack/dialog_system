
# coding: utf-8

# This file illustrates a general template for making decisions in NLG:
# - Choose the **head word**, given the target meaning and target syntactic category.  The [head][1] is the key word that determines the syntax and semantics of a phrase.
# - Choose **arguments** to express, assign them to **semantic roles**, and figure out how those semantic roles will be realized through **grammatical relationships**.  An [argument][2] is an element that will be interpreted as completing the meaning of the head; a [semantic role][3] (also called thematic role) indicates a particular underlying relationship between an argument and the head, and [grammatical relationships][4] indicate a particular way to realize that relationship in word order and hierarchical structure.
# - Generate constituents for the arguments **recursively**
# - **Inflect** the head for agreement
# - **Linearize** the resulting structure
# 
# [1]: http://en.wikipedia.org/wiki/Head_(linguistics)
# [2]: http://en.wikipedia.org/wiki/Argument_(linguistics)
# [3]: http://en.wikipedia.org/wiki/Thematic_relation
# [4]: http://en.wikipedia.org/wiki/Grammatical_relation
# 
# It also includes a simple python implementation of **feature structures**
# - A [feature structure][5] is a recursive object representing the structure or meaning of a complex linguistic object
# - The basic items are dictionaries
# - Basically, dictionaries pair **features** with **values**, which are strings used to represent symbolic information
# - Recursively, dictionaries can pair **features** with **other feature structures**
# 
# [5]: http://en.wikipedia.org/wiki/Feature_structure
# 
# The process of NLG records results and decisions in feature structures.  Feature structures are mutable objects that are shared across NLG decision making.  So changes made in one part of the derivation become visible in other parts of NLG.  This information sharing makes processes of agreement, linearization and the like easy.
# 
# This file illustrates everything with a classic "locative alternation", also called the "spray/load" alternation, because of the verbs it occurs with.  [This link][6] gives you some resources to learn more about this variability in English verbs.
# 
# [6]: http://allthingslinguistic.com/post/82327954516/list-of-verbs-grouped-by-their-syntactic-processes
# 
# Here is a simple feature structure that we will start with.  It initializes a feature structure with a semantic description, characterizing a change of location event in which one worker acts as the agent performing the action, four machines are moved, and their destination is the place on two trucks.  This particular representation is due to [Ray Jackendoff][7], but unfortunately there doesn't seem to be a nice informal introduction to it online. 
# 
# [7]: http://en.wikipedia.org/wiki/Ray_Jackendoff
# 
# This demo was inspired by [a blog post by Pablo Duboue][8].
# 
# [8]:http://duboue.net/blog5.html

# In[1]:

global m1 
m1 = {}
def reset():
    global m1
    m1 = {"semantics": {"event": "change-of-location",
                    "agent": {"category": "worker",
                              "number": 1},
                    "moved": {"category": "machine",
                              "number": 4},
                    "place": {"relation": "on",
                              "landmark": {"category": "truck",
                                           "number": 2}
                          }}}


# Here is a function that applies one template for the word `load`.  It checks to make sure that what we have is a loading event, and that we know everything about the event that we need to know to describe this event as `(X) loading Z (with Y)`.  If we have this information, we pick the word load, and assign underlying grammatical relationships: `X` is the underlying subject, `Z` is the underlying object, and `Y` is an underlying  `with` prepositional phrase.

# In[2]:

def apply_load_with(fs) :
    features = fs["semantics"] 
    if "event" not in features:
        return False
    if features["event"] != "change-of-location":
        return False
    if "place" not in features:
        return False
    if "relation" not in features["place"] or "landmark" not in features["place"]:
        return False
    if features["place"]["relation"] != "on":
        return False
    fs["verb-stem"] = "load"
    fs["u-obj"] = features["place"]["landmark"]
    if "agent" in features:
        fs["u-subj"] = features["agent"]
    if "moved" in features:
        fs["u-pp-obj"] = features["moved"]
        fs["u-pp-obj"]["role-marker"] = "with"
    return True


# This function that applies the other template for the word `load`.  It checks to make sure that what we have is a loading event, and that we know everything about the event that we need to know to describe this event as `(X) loading Y (on Z)`.  If we have this information, we pick the word load, and assign underlying grammatical relationships: `X` is the underlying subject, `Y` is the underlying object, and `Z` is an underlying  `on` prepositional phrase.

# In[3]:

def apply_load_on(fs) :
    features = fs["semantics"] 
    if "event" not in features:
        return False
    if features["event"] != "change-of-location":
        return False
    if "moved" not in features:
        return False
    fs["verb-stem"] = "load"
    fs["u-obj"] = features["moved"]
    if "agent" in features:
        fs["u-subj"] = features["agent"]
    if "place" in features and "landmark" in features["place"] and "relation" in features["place"]:
        fs["u-pp-obj"] = features["place"]["landmark"]
        fs["u-pp-obj"]["role-marker"] = features["place"]["relation"]
    return True


# The next functions translate the underlying semantic roles to relationships in surface syntax, and realzie the arguments recursively.  There are two options.  In an active sentence, the underlying subject is the surface subject; the underlying object is the surface object, and the underlying prepositional phrases are realized as modifiers:

# In[4]:

def realize_active(fs) :
    if "u-subj" not in fs:
        return False
    fs["voice"] = "active"
    realize_np(fs["u-subj"])
    fs["subj"] = fs["u-subj"]
    if "u-obj" in fs:
        realize_np(fs["u-obj"])
        fs["dobj"] = fs["u-obj"]
    if "u-pp-obj" in fs:
        realize_np(fs["u-pp-obj"])
        fs["mod"] = [fs["u-pp-obj"]] 
    return True


# In a passive sentence, the underlying subject is the surface object; the underlying subject is realized with a `by` prepositional phrase, and the underlying prepositional phrases are realized as modifiers:

# In[5]:

def realize_passive(fs) :
    if "u-obj" not in fs:
        return False
    fs["voice"] = "passive"
    realize_np(fs["u-obj"])
    fs["subj"] = fs["u-obj"]
    fs["mod"] = []
    if "u-subj" in fs :
        realize_np(fs["u-subj"])
        fs["u-subj"]["role-marker"] = "by"
        fs["mod"].append(fs["u-subj"])
    if "u-pp-obj" in fs:
        realize_np(fs["u-pp-obj"])
        fs["mod"].append(fs["u-pp-obj"])
    return True


# Here's a simple way to realize noun phrases.  You could write this recursively, because in general noun phrases have complicated structures, but this is a start...

# In[6]:

def realize_np(fs):
    if "number" not in fs or fs["number"] == 1:
        fs["g-number"] = "singular"
        fs["string"] = "the " + fs["category"] 
        return True
    fs["g-number"] = "plural"
    fs["string"] = "the " + str(fs["number"]) + " " + fs["category"] + "s"
    return True


# This is another sketch: handling agreement.  What you should really do is distinguish regular verbs and irregular verbs, whose forms are listed.  In the rule for regular verbs, you need to keep track of whether the verb forms the present singular with `-s` or `-es` and whether it forms the past with `-d` or `-ed`.  But making that table is easy -- it's a big list.  The hard part is making sure the information is available to make the right choice.  This function shows how a feature structure puts all the information together to make it work...

# In[7]:

def inflect_verb(fs):
    stem = fs["verb-stem"]
    voice = fs["voice"]
    number = fs["subj"]["g-number"]
    if voice == "active" :
        if number == "singular" :
            fs["verb-form"] = stem + "s"
        else:
            fs["verb-form"] = stem
    else:
        if number == "singular" :
            fs["verb-form"] = "is " + stem + "ed"
        else:
            fs["verb-form"] = "are " + stem + "ed"


# Linearizing is turning a structure into a string.  Here we just join all the constituents in order, separated by spaces.

# In[8]:

def linearize(fs) :
    items = [ fs["subj"]["string"], fs["verb-form"] ]
    if "dobj" in fs :
        items.append(fs["dobj"]["string"])
    if "mod" in fs :
        for x in fs["mod"]:
            items.append(x["role-marker"] + " " + x["string"])
    fs["string"] = ' '.join(items)


# This function puts it all together.  You give it a feature structure, and the option to prefer `with` or `on` to describe movement and `active` or `passive` voice.  The function tries multiple alternatives (since not all the information may necessarily be available) but prefers what you've specified as input.

# In[9]:

def describe_loading(fs, role='with', voice='active') :
    if role=='with':
        pattern1, pattern2 = apply_load_with, apply_load_on
    else:
        pattern1, pattern2 = apply_load_on, apply_load_with
    if not pattern1(fs) and not pattern2(fs) :
        return None
    if voice=='active':
        voice1, voice2 = realize_active, realize_passive
    else:
        voice1, voice2 = realize_passive, realize_active
    if not voice1(fs) and not voice2(fs) :
        return None
    inflect_verb(fs)
    linearize(fs)
    return fs["string"]


# The initial semantics

# In[10]:

reset(); print m1


# Four different grammatical realizations, combining the locative alternation and the active/passive alternation.

# In[11]:

reset(); describe_loading(m1)


# In[12]:

reset(); describe_loading(m1, role="on")


# In[13]:

reset(); describe_loading(m1, voice="passive")


# In[14]:

reset(); describe_loading(m1, role="on", voice="passive")


# Preferences are overridden when we have only partial information about the event.  Perhaps this leads to ambiguity...

# In[15]:

reset(); del m1["semantics"]["agent"]; describe_loading(m1)


# In[16]:

reset(); del m1["semantics"]["agent"]; describe_loading(m1, role="on")


# In[17]:

reset(); del m1["semantics"]["place"]; describe_loading(m1)


# In[18]:

reset(); del m1["semantics"]["moved"]; describe_loading(m1)


# In[19]:

reset(); del m1["semantics"]["moved"]; del m1["semantics"]["agent"]; 
describe_loading(m1)


# In[20]:

reset(); del m1["semantics"]["place"]; del m1["semantics"]["agent"]; 
describe_loading(m1)

