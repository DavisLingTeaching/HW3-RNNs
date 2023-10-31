import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_pretrained_model(name:str='distilgpt2') -> tuple[GPT2LMHeadModel,
                                                          GPT2Tokenizer]:
    """Loads a pretrained auto-regressive language model from 
        HuggingFace

    Args:
        name (str): Name of model
    Returns:
        model (GPT2LMHeadModel): Pretrained model
        tokenizer (GPT2Tokenizer): Tokenizer
    """
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(name)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def get_aligned_words_measures(text: str, 
                               measure: str,
                               model: GPT2LMHeadModel, 
                               tokenizer: GPT2Tokenizer) -> list[str]:
    """ Returns words and their measure (prob|surp)
    Args:
        text (list[str]): list of sentences
        measure (str): Measure, either probability or surprisal
                        (options: prob|surp)
        model (GPT2LMHeadModel): Pretrained model
        tokenizer (GPT2Tokenizer): Tokenizer
    Returns:
        list[str]: List of words with their measures

    For example, 
    >>> model, tokenizer = load_pretrained_model()
    >>> get_aligned_words_measures('the student is happy', 
    ...        'surp', model, tokenizer)
    [('the', 0), ('student', 17.38616943359375), ('is', 6.385905742645264),
     ('happy', 9.564245223999023)]
    >>> get_aligned_words_measures('the cat is fluffy', 
    ...        'prob', model, tokenizer) 
    [('the', 0), ('cat', 2.5601848392398097e-06), ('is', 0.025296149775385857),
     ('fluffy', 0.00020585735910572112)]
    >>> get_aligned_words_measures('the cat are fluffy', 
    ...        'prob', model, tokenizer)
    [('the', 0), ('cat', 2.5601848392398097e-06), ('are', 0.0010310395155102015),
     ('fluffy', 0.00021902224398218095)]
    """
    if measure not in {'prob', 'surp'}:
        sys.stderr.write(f"{measure} not recognized\n")
        sys.exit(1)

    data = []

    ids = tokenizer(text, return_tensors='pt')
    input_ids = ids.input_ids.flatten().data
    target_ids = ids.input_ids[:,1:]

    # get output
    logits = model(**ids).logits
    output = torch.nn.functional.log_softmax(logits, dim=-1)
    if measure == 'surp':
        output = -(output/torch.log(torch.tensor(2.0)))
    else:
        output = torch.exp(output)

    # get by token measures 
    target_measures = output[:,:-1, :]
    # use gather to get the output for each target item in the batch
    target_measures = target_measures.gather(-1,
                             target_ids.unsqueeze(2)).flatten().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    words = text.split(' ')

    # A lil loop to force align words 
    current_word = words.pop(0)
    current_token = tokens.pop(0)
    measure = 0
    while len(data) != len(text.split(' ')):
        if current_word == current_token:
            data.append((current_word, measure))
            measure = 0
            if words:
                current_word = words.pop(0)
                current_token = tokens.pop(0).replace('Ġ', '')
                measure += target_measures.pop(0)
        else:
            measure += target_measures.pop(0)
            current_token += tokens.pop(0).replace('Ġ', '')

    return data

# TODO: Your code goes here


