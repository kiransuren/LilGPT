class CharacterTokenizer:
    """ Simple character level tokenizer that reads a text file and returns a list of words.
    """
    
    def __init__(self):
        # member variables
        self.chars = []
        self.vocab_size = 0
        self.text = ""
        
        self.stoi = {}
        self.itos = {}

    def load_corpus(self, text_filename="input.txt"):
        """ Load the corpus from a text file and initialize the tokenizer.
        """
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
    
        # store vocabulary and vocabulary size
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        
        # create a dictionary to map character to integer (corresponding to the index in chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # create a dictionary to map integer to character
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        

    def encode(self, s):
        """ Encode a string into a list of integers based on the character mapping.
        """
        
        return [self.stoi[c] for c in s]
        

    def decode(self, i):
        """ Decode a list of integers back into a string based on the character mapping.
        """
        
        return ''.join([self.itos[i] for i in i])
