import json

from tqdm import trange
import regex as re


class BPETokenizer:
    _NUM_BYTES = 256

    def __init__(self, vocab_size: int, special_tokens: dict | None = None) -> None:
        self.vocab_size = vocab_size
        
        self.special_tokens = special_tokens if special_tokens is not None else {"<|endoftext|>": vocab_size}
        self.vocab = {i: bytes([i]) for i in range(self._NUM_BYTES)} # Initialize the vocabulary with byte tokens (0-255)
        for token, token_id in self.special_tokens.items(): # Add special tokens to the vocabulary
            self.vocab[token_id] = token.encode('utf-8')

        self.merges = {} 
        self._init_regex() # Compile the regex pattern with special tokens

    def train(self, text_file: str) -> None:
        """Train the BPE tokenizer on the input text."""
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()  

        text_chunks = self._pre_process(text)
        byte_chunks = [list(
            chunk.encode('utf-8'))
            for chunk in text_chunks
            if chunk not in self.special_tokens
        ] # Encode the text chunks into byte sequences, excluding special tokens

        for i in trange(self.vocab_size - self._NUM_BYTES, desc="Training BPE"): # Account for the initial 256 byte tokens
            best_pair = self._find_pair(byte_chunks) # Find the most frequent pair of tokens
            if not best_pair:
                break

            new_token_id = self._NUM_BYTES + i
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # merge characters of the best pair into a single new token
            self.merges[new_token_id] = best_pair # Store the merge operation
            byte_chunks = self._merge_pair(byte_chunks, best_pair, new_token_id)

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a sequence of token IDs."""
        text_chunks = self._pre_process(text)
        token_ids = []
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.special_tokens[chunk])
                continue

            byte_chunk = list(chunk.encode('utf-8'))
            while len(byte_chunk) >= 2:
                pairs = set((byte_chunk[i], byte_chunk[i + 1]) for i in range(len(byte_chunk) - 1)) # all unique pairs in the chunk
                best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf"))) # Find the pair with the lowest new token ID (added earliest)
                if best_pair not in self.merges: # If no pairs are mergeable, stop
                    break

                byte_chunk = self._merge_ids(byte_chunk, best_pair, self.merges[best_pair]) # Merge the best pair in the byte chunk
            token_ids.extend(byte_chunk)
        return token_ids
    
    def _merge_ids(self, byte_chunk: list[int], new_pair: tuple[int, int], new_id: int) -> list[int]:
        """Helper to replace occurrences of a pair with a new ID."""
        new_ids = []
        i = 0
        while i < len(byte_chunk):
            if i < len(byte_chunk) - 1 and (byte_chunk[i], byte_chunk[i + 1]) == new_pair:
                new_ids.append(new_id) # Append the new token ID for the merged pair
                i += 2
            else:
                new_ids.append(byte_chunk[i]) # Append the original token ID if no merge occurs
                i += 1
        return new_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text."""
        byte_string = b"".join(self.vocab[id] for id in token_ids)
        return byte_string.decode("utf-8")

    def save(self, path: str) -> None:
        """Store the tokenizer's vocabulary, merges and configuration to a file."""
        hex_vocab = {id: byte_value.hex() for id, byte_value in self.vocab.items()}
        state = {
            "vocab": hex_vocab,
            "merges": {" ".join((str(x) for x in k)): v for k, v in self.merges.items()}, # convert tuple keys to space-separated strings
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load the tokenizer's vocabulary, merges and configuration from a file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        instance = cls.__new__(cls) # Create an uninitialized instance of the class
        instance.vocab = {int(id): bytes.fromhex(hex_value) for id, hex_value in state["vocab"].items()}
        instance.merges = {tuple(int(x) for x in pair.split()): int(new_id) for pair, new_id in state["merges"].items()} # convert strings back to tuple keys
        instance.special_tokens = state["special_tokens"]
        instance.vocab_size = state["vocab_size"]
        instance._NUM_BYTES = 256
        instance._init_regex() # Re-initialize the regex patterns
        return instance

    def _find_pair(self, text_chunks: list[str]) -> tuple[int, int]:
        """Find the most frequent pair of tokens in the text."""
        pair_freq = {}
        for chunk in text_chunks:
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

        return max(pair_freq, key=pair_freq.get) if pair_freq else None

    def _merge_pair(self, text_chunks: list[list[int]], pair: tuple[int, int], new_token: int) -> list:
        """Merge the specified pair of tokens in the text."""
        new_text_chunks = []

        for chunk in text_chunks:
            idx = 0
            new_chunk = []
            while idx < len(chunk): # Iterate through the chunk to merge pairs
                if idx < len(chunk) - 1 and chunk[idx] == pair[0] and chunk[idx+1] == pair[1]:
                    new_chunk.append(new_token)
                    idx += 2 
                else: # If token does not math the pair or the end of the chunk is reached
                    new_chunk.append(chunk[idx])
                    idx += 1
            new_text_chunks.append(new_chunk)
        return new_text_chunks

    def _pre_process(self, text: str) -> list[str]:
        """Apply regex patterns to clean the text and text chunks."""
        if not self.special_pattern:
            return self.pattern.findall(text)
        
        text_chunks = []
        for part in re.split(self.special_pattern, text): # Split the text by special tokens
            if not part: # Skip empty parts
                continue
            if part in self.special_tokens: # If the part is a special token, add it directly
                text_chunks.append(part)
            else: # Otherwise, apply the regex pattern to extract text chunks
                text_chunks.extend(self.pattern.findall(part))
        return text_chunks

    def _init_regex(self) -> None:
        self.pattern = re.compile(r"""               # Compile the regex pattern for fast use
                    '(?i:[sdmt]|ll|ve|re)            # Match common English contractions, e.g. 's, 'd, 'll, etc.
                    |[^\r\n\p{L}\p{N}]?\p{L}+        # Match sequences of letters with an optional leading character like a space
                    |\p{N}{1,3}                      # Match sequences of 1 to 3 digits
                    |\ ?[^\s\p{L}\p{N}]+[\r\n]*      # Match characters that are letters, whitespaces or digits, followed by optional newlines, e.g. "..."
                    |\s*[\r\n]+|\s+(?!\S)|\s+        # Match newlines and whitespace sequences
                """, re.X) # Allow comments and whitespace in the regex pattern
        
        if self.special_tokens:
            escaped_specials = [re.escape(token) for token in self.special_tokens.keys()]
            self.special_pattern = re.compile(f"({'|'.join(escaped_specials)})") # Add parentheses to keep the delimiter in the split
        else:
            self.special_pattern = None
        
    @property
    def eot_token(self) -> int | None:
        """Return the end-of-text token ID."""
        if "<|endoftext|>" in self.special_tokens:
            return self.special_tokens["<|endoftext|>"]
        return None