import jieba
from collections import Counter
from torchtext.vocab import vocab

def en_tokenizer(s, punctuation_marks = None):
    '''english sentence tokenizer

    Parameters:
    ----------
    s:str
        English sentence string

    punctuation_marks:list
        Punctuation marks involved in the content to be processed by the tokenizer
        default:None

    Returns:
    --------
    r:list
    '''
    if punctuation_marks == None:
        punctuation_marks = [".", "?", "!", ",", ";", ":", "'", "-", "(", ")", "[", "]", "{", "}", "<", ">"];
    for mark in punctuation_marks:
        s = s.replace(mark, ' '+ mark);
    return s.split();

def ch_tokenizer(s, cut_by_word = False):
    '''Chinese sentence tokenizer

    Parameters:
    ----------
    s:str
        English sentence string
    
    cut_by_word:bool
        Whether to divide by word, otherwise by words

    Returns:
    --------
    r:list
    '''
    if cut_by_word:
        r = [w for w in s];
    else:
        r = ' '.join(jieba.cut(s, cut_all = False)).split();
    return r

def vocab_generator(filepath, min_freq, lang = 'en', punctuation_marks = None, cut_by_word = False, specials = None):
    '''Vocabulary generator for English and Chinese

    Parameters:
    -----------
    filepath:str

    min_freq:int
        The minimum word frequency, below this value will be deleted from the vocabulary.
    
    lang:str
        English('en') or Chinese('ch')
        default:'en'

    punctuation_marks:list
        Punctuation marks involved in the content to be processed by the tokenizer, only valid when dealing with English
        default:None

    cut_by_word:bool
        Whether to divide by word, otherwise by words, only valid when dealing with Chinese
        default:None

    specials:list
        special symbols that will be used
        default:None

    Returns:
    --------
    r:torchtext.vocab.Vocab
    '''
    if specials == None:
        specials = ['[unk]', '[pad]', '[bos]', '[eos]'];
    counter = Counter();
    with open(filepath) as f:
        for s in f:
            if lang == 'en':
                counter.update(en_tokenizer(s.strip(), punctuation_marks));
            elif lang == 'ch':
                counter.update(ch_tokenizer(s.strip(), cut_by_word));
    return vocab(counter, min_freq = min_freq, specials = specials);
