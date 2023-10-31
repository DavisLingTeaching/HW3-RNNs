# COSC 480C (NLP) Homework 3: Fall 2023

The due date for this homework is **Monday, November 13, 11pm ET**.
Start early!

## Introduction

This assignment is designed to guide you through the implementation and
evaluation of a neural language model using a simple recurrent neural network. 

#### Guiding Question
- How do we build neural networks? 
- How do we train a neural language model? 
- How do we evaluate our models? 

#### Learning Objectives
- Implement a recurrent neural network language model
- Train a neural language model
- Understand how to tune neural network hyperparameters 
- Evaluate a pre-trained neural language model 
- Understand how core concepts of language modeling are transferred to a neural
  network setting 

## Rubric 

1. Part 1 - 100 Total Points

    | Question | Points |
    | -------- | ------ | 
    | RNNCell init | 10 points |
    | RNNModel init | 10 points |
    | RNNCell forward | 40 points |
    | RNNModel forward | 40 points |

2. Part 2 - 50 Total Points

    | Question | Points |
    | -------- | ------ | 
    | train    | 50 points |

3. Part 3 - 50 Total Points


## Your assignment

Your task is to complete following steps:

1. Clone the homework repository

    ```
    git clone https://github.com/DavisLingTeaching/HW3-RNNs.git
    ```

2. Install pytorch ([link](https://pytorch.org/get-started/locally/))

4. Complete `model.py` described in [Part 1](#part-1)
5. Complete `train.py` described in [Part 2](#part-2)
6. Complete `ling.py` described in [Part 3](#part-3)
7. Submit your completed programs to Moodle.

## Part 1 - Implementing

### Building a RNN language model

In this portion of the assignment you will implement a recurrent neural network
language model. A description of the function and samples of its use are given in the
function docstring. **Please read the docstrings**. I have implemented some
other functions that you may find useful! 

Your task is broken into two components: implementing an RNN cell and
implementing the full model. The cell handles the computation for a single
layer, while the full model handles the full forward pass from input to output
through layers of RNN cells. 

Note that the input to your RNN model are token ids. In addition to the RNN cell
layers, you should have an embedding matrix whose weights are learned.
Additionally, you should have a decoder (with no bias) that maps from the output
of the final RNN cell to the vocab space. This mirrors the E and V matrices,
respectively, in the RNN handout from class. The handout and its key are
provided in the repo under notes. 

### Testing your RNN language model

Some (non-exhaustive) tests can be run by using test.py. 

For example, 

    python test.py --test init_cell

will test some aspects of the init of your RNNCell

## Part 2 - Training

### Training a RNN language model

In this portion you will train your RNN with some data. Starter code is included
with train.py that handles batching the data. **Please review the existing
code**. You should review the feedforward neural network lecture notes
([link](https://github.com/DavisLingTeaching/PyTorchNN)). 

## Part 3 - Evaluating 

In this portion you will evaluate a pre-trained model for linguistic knowledge.
In particular, you will evaluate a tiny version of GPT-2 called
[distilgpt2](https://huggingface.co/distilgpt2). You are tasked with
benchmarking distilgpt2's linguistic knowledge. Helper code that loads the
pre-trained model and that returns by-word surprisals/probabilities is included
in ling.py. In order to run this code, you have to install HuggingFace's
transformers library. Installation instructions can be found
[here](https://github.com/huggingface/transformers#installation).

How should you evaluate your model's linguistic knowledge? This problem is
intentionally open ended to give you practice creatively approaching problems.
As such there is no right answer, instead you should justify why your approach
makes sense and how you came to your conclusions. Please include this in a
multi-line comment at the top of your code or in a pdf, whichever you prefer. 

Still unsure how you might approach this. Here's one way we've covered in Week
11's lectures. Suppose you were interested in subject-verb agreement and whether
this model had learned the relevant linguistic behavior. You could construct
minimal pairs like the following. 

    a) The cat is so happy
    b) The cat are so happy

We could get the by\_word probabilities from distilgpt2 using the included code: 

    >>> from ling import *
    >>> model, tokenizer = load_pretrained_model()
    >>> sent_a = 'The cat is happy'
    >>> sent_b = 'The cat are happy'
    >>> a_probs = get_aligned_words_measures(sent_a, 'prob', model, tokenizer)
    >>> b_probs = get_aligned_words_measures(sent_b, 'prob', model, tokenizer)
    >>> a_probs
    [('The', 0), ('cat', 1.437380433344515e-06), ('is', 0.05121663585305214), ('happy', 0.0012213564477860928)]
    >>> b_probs
    [('The', 0), ('cat', 1.437380433344515e-06), ('are', 0.0011601115111261606), ('happy', 0.0011389038991183043)]

We see that both the verb and the whole sentence is more likely for a than for
b. If this happens with more sentences, we might conclude that distilgpt2 has
captured a basic aspect of subject-verb agreement. We can feel more confident if
we try other constructions that get at the same process. For example, 

    c) The cat near the books is so happy
    d) The cat near the books are so happy

Here it is slightly harder to keep track of the subject-verb dependency. We can
run the same code: 

    >>> from ling import *
    >>> model, tokenizer = load_pretrained_model()
    >>> sent_a = 'The cat near the books is happy'
    >>> sent_b = 'The cat near the books are happy'
    >>> a_probs = get_aligned_words_measures(sent_a, 'prob', model, tokenizer)
    >>> b_probs = get_aligned_words_measures(sent_b, 'prob', model, tokenizer)
    >>> a_probs
    [('The', 0), ('cat', 1.437380433344515e-06), ('near', 6.501279858639464e-05), ('the', 0.34228000044822693), ('books', 3.5550870961742476e-05), ('is', 0.059263549745082855), ('happy', 0.0005464301793836057)]
    >>> b_probs
    [('The', 0), ('cat', 1.437380433344515e-06), ('near', 6.501279858639464e-05), ('the', 0.34228000044822693), ('books', 3.5550870961742476e-05), ('are', 0.004924507811665535), ('happy', 0.0004336706770118326)]

We see again that the model is correct. You might try this basic approach with a
different linguistic process.

Good luck :) !
