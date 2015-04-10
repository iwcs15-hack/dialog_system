
# coding: utf-8

# A notebook demonstrating some key features of the python programming language.
# 
# Draws particularly closely on ideas from [the Python Tutorial][1] and [the Python Regular Expression Tutorial][2]
# 
# [1]: https://docs.python.org/3/tutorial/index.html
# [2]: https://docs.python.org/2/howto/regex.html
# 
# You might also want to familiarize yourself with IPython notebooks, through sites like [this one][3] or [this one][4].
# 
# [3]: https://blog.safaribooksonline.com/2013/12/12/start-ipython-notebook/
# [4]: http://www.randalolson.com/2012/05/12/a-short-demo-on-how-to-use-ipython-notebook-as-a-research-notebook/
# 
# Don't forget [the argument sketch][5].
# 
# [5]:https://www.youtube.com/watch?v=kQFKtI6gn9Y
# 
# Matthew Stone; Feb 10, 2015; CS 195.

# Numerical computations, specified interactively.

# In[1]:

9 / 2


# Lists: specified using brackets.

# In[2]:

[1, 2, 3]


# Tuples: fixed sequences, specified using parentheses.

# In[3]:

("a", "b")


# Various ways of specifying string literals, using single quotes or double quotes, and using the `r` tag to control how special characters are treated.

# In[4]:

man1 = "I'd like to have an argument please."
man1


# In[5]:

man2 = 'I\'d like to have an argument please.'
man2


# In[6]:

man3 = r"I'd like to have an argument please."
man3


# In[7]:

man4 = r'I\'d like to have an argument please.'
man4


# A variety of operations on strings:
# - `len` (function): Computing the length of a string 
# - `split` (method): Separating a string into tokens based on a simple pattern delimeter
# - indexing and slices using `[:]` notation

# In[8]:

len(man1)


# In[9]:

words = man1.split(' ')
words


# In[10]:

man1[2]


# In[11]:

man1[-2]


# In[12]:

man1[0:15]


# In[13]:

man1[:8]


# In[14]:

man1[-7:]


# Key concepts and constructs for flow of control in Python
# - Indentation defines blocks to organize code
# - Blocks begin with an operator ending in `:`
# - `if ... else ...` blocks for conditional execution
# - `for ...` blocks for iteration
# - generator expressions make iteration a very lightweight coding style.
# 
# You can totally [geek out][1] on programming with generators!
# [1]:http://www.dabeaz.com/generators/

# In[15]:

if man1[-1] == '.' :
    man1 = man1[:-1]


# In[16]:

man1


# In[17]:

for w in words :
    print w, "!"


# In[18]:

'-- '.join(words)


# In[19]:

'-- '.join(w.upper() + " really " for w in words if len(w) < 5 )


# Regular expressions, the mainstay of string programming in Python.
# 
# Interface:
# - `compile` a regular expression (string) into a pattern
# - `search` or `match` a pattern against another string, returning `Match` objects
# - `group` method gives the matched string (or matched substrings for complex patterns)
# - `span` shows where in the original string the match was found

# In[20]:

import re


# In[21]:

pat1 = r"argument"


# In[22]:

p1 = re.compile(pat1)


# In[23]:

m1 = p1.search(man1)


# In[24]:

m1


# In[25]:

m1.group()


# In[26]:

m1.span()


# In[27]:

man1[20:28]


# Key elements of regular expressions
# - fixed text
# - character classes that match a range of possibilities, like `\\s` for space, `.` for anything
# - repetitions, like `*` for any number of repetitions including 0
# - `()` to group subexpressions (which can then be repeated, reported, etc.)

# In[28]:

pat2 = r"I'd like to\s(.*)\splease"


# In[29]:

p2 = re.compile(pat2)


# In[30]:

m2 = p2.search(man1)


# In[31]:

m2


# In[32]:

m2.group()


# In[33]:

m2.group(1)


# In[34]:

"You can't " + m2.group(1) + "!"

