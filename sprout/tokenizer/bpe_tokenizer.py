import regex as re


class BPETokenizer:
    PATTERN = re.compile(r""" # Compile the regex pattern for fast use
                        '(?i:[sdmt]|ll|ve|re)      # Match common English contractions, e.g. 's, 'd, 'll, etc.
                        |[^\r\n\p{L}\p{N}]?\p{L}+  # Match sequences of letters with an optional leading character like a space
                        |\p{N}{1,3}                # Match sequences of 1 to 3 digits
                        |\ ?[^\s\p{L}\p{N}]+[\r\n]* # Match characters that are letters, whitespaces or digits, followed by optional newlines, e.g. "..."
                        |\s*[\r\n]+|\s+(?!\S)|\s+  # Match newlines and whitespace sequences
                    """, re.X) # Allow comments and whitespace in the regex pattern

    def __init__(self, text_file: str, vocab_size: int) -> None:
        self.num_bytes = 256
        self.text = self._load_text(text_file)
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(self.num_bytes)} # Initialize vocab with byte values
        self.merges = {}

    def train(self):
        """Train the BPE tokenizer on the input text."""
        text_chunks = self._pre_process(self.text)
        for i in range(self.vocab_size - self.num_bytes): # Account for the initial 256 byte tokens
            best_pair = self._find_pair(text_chunks) # Find the most frequent pair of tokens
            if not best_pair:
                break

            new_token_id = self.num_bytes + i
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]] # merge characters of the best pair into a single new token
            self.merges[new_token_id] = best_pair # Store the merge operation
            text_chunks = self._merge_pair(text_chunks, best_pair, new_token_id)

    def encode(self, text: str) -> list:
        """Encode the input text into a sequence of token IDs."""
        text_chunks = self._pre_process(text)
        for new_id, pair in self.merges.items():
            text_chunks = self._merge_pair(text_chunks, pair, new_id)
        return [token for chunk in text_chunks for token in chunk] # unwrap the chunks

    def decode(self, token_ids: list) -> str:
        """Decode a sequence of token IDs back into text."""
        byte_string = b"".join(self.vocab[id] for id in token_ids)
        return byte_string.decode("utf-8")

    def _find_pair(self, text_chunks: list) -> tuple:
        """Find the most frequent pair of tokens in the text."""
        pair_freq = {}
        for chunk in text_chunks:
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

        return max(pair_freq, key=pair_freq.get) if pair_freq else None

    def _merge_pair(self, text_chunks: list, pair: tuple, new_token: int) -> list:
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

    def _pre_process(self, text: str) -> list:
        """Apply regex patterns to clean the text and turn into UTF-8."""
        chunks = re.findall(self.PATTERN, text)
        byte_chunks = [list(chunk.encode('utf-8')) for chunk in chunks]
        return byte_chunks

    def _load_text(self, text_file: str) -> str:
        """Load text from a file."""
        with open(text_file, "r", encoding="utf-8") as f:
            return f.read()