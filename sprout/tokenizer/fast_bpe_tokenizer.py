import json
import heapq
from collections import defaultdict

from tqdm import trange
import regex as re


class Node:
    __slots__ = ("prev", "val", "next") # Reduce memory overhead by defining fixed attributes

    def __init__(self, val):
        self.prev = None
        self.val = val
        self.next = None

class FastBPETokenizer:
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
        chunk_heads = self._build_linked_list(byte_chunks) # Build linked lists for the byte chunks
        pair_counts, pair_positions = self._build_stats(chunk_heads) # Initial scan to build pair frequency and position mappings

        heap = []
        for pair, count in pair_counts.items(): # Create initial frequency heap
            self._heap_push(heap, pair, count)

        for i in trange(self.vocab_size - self._NUM_BYTES, desc="Training BPE"):
            best_pair, best_count = self._heap_pop(heap, pair_counts)
            if not best_pair or best_count < 2: # If no valid pairs are left, stop training
                break

            new_token_id = self._NUM_BYTES + i
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[new_token_id] = best_pair
            self._merge_pair(best_pair, pair_counts, pair_positions, new_token_id, heap) # Merge the best pair and update stats

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a sequence of token IDs."""
        text_chunks = self._pre_process(text)
        token_ids = []
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.special_tokens[chunk])
            else:
                byte_chunk = list(chunk.encode('utf-8'))
                for new_id, pair in self.merges.items():
                    idx = 0
                    while idx < len(byte_chunk) - 1:
                        if (byte_chunk[idx], byte_chunk[idx + 1]) == pair:
                            byte_chunk[idx] = new_id
                            del byte_chunk[idx + 1]
                        else:
                            idx += 1
                token_ids.extend(byte_chunk)
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text."""
        byte_string = b"".join(self.vocab[id] for id in token_ids)
        return byte_string.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Store the tokenizer's vocabulary, merges and configuration to a file."""
        hex_vocab = {id: byte_value.hex() for id, byte_value in self.vocab.items()}
        state = {
            "vocab": hex_vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, path: str) -> "FastBPETokenizer":
        """Load the tokenizer's vocabulary, merges and configuration from a file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        instance = cls.__new__(cls) # Create an uninitialized instance of the class
        instance.vocab = {int(id): bytes.fromhex(hex_value) for id, hex_value in state["vocab"].items()}
        instance.merges = {int(new_id): tuple(pair) for new_id, pair in state["merges"].items()} # Ensure correct types
        instance.special_tokens = state["special_tokens"]
        instance.vocab_size = state["vocab_size"]
        instance._NUM_BYTES = 256
        instance._init_regex() # Re-initialize the regex patterns
        return instance
    
    def _build_linked_list(self, byte_chunks: list[list[int]]) -> list:
        """Build a doubly linked list from a list of tokens and return the heads of the linked lists for each chunk."""
        chunk_heads = []
        for chunk in byte_chunks:
            head = Node(chunk[0])
            curr = head 
            for token in chunk[1:]: # Create a linked list
                new_node = Node(token) # Create a new node
                curr.next = new_node # Link the current node to the new node
                new_node.prev = curr # Link the new node back to the current node
                curr = new_node # Move to the new node
                
            chunk_heads.append(head)
        return chunk_heads

    def _build_stats(self, chunk_heads: list[Node]) -> tuple:
        """Build a frequency dictionary for pairs of tokens and a mapping for their positions in the linked lists."""
        pair_counts = {}
        pair_positions = defaultdict(set)
        for head in chunk_heads:
            curr = head
            while curr and curr.next:
                pair = (curr.val, curr.next.val)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                pair_positions[pair].add(curr) # Only store the head node for that pair
                curr = curr.next
        return pair_counts, pair_positions

    def _replace_pair(self, node_a: Node, new_token: int) -> Node:
        """Replace a pair of tokens in the linked list with a new token."""
        node_b = node_a.next
        node_c = Node(new_token)

        # Connect the new node to the previous and next nodes in the list
        node_c.prev = node_a.prev 
        node_c.next = node_b.next 

        # Link previous and next nodes to the new node if they are not a head or tail
        if node_a.prev:
            node_a.prev.next = node_c
        if node_b.next:
            node_b.next.prev = node_c 

        # Clean up the old nodes
        node_a.prev = node_a.next = None
        node_b.prev = node_b.next = None
        return node_c

    def _heap_push(self, heap: list, pair: tuple[int, int], count: int) -> None:
        """Push a pair and its count onto the heap."""
        heapq.heappush(heap, (-count, pair)) # Negate count for max-heap behavior

    def _heap_pop(self, heap: list, pair_count: dict[tuple[int, int], int]) -> tuple:
        """Pop the heap until a non-stale pair is found."""
        while heap:
            neg_count, pair = heapq.heappop(heap)
            count = -neg_count
            if pair_count.get(pair, 0) == count: # Check if the pair count is still valid
                return pair, count
        return None, 0

    def _merge_pair(self, best_pair: tuple[int, int], pair_counts: dict[tuple[int, int], int],
                    pair_positions: dict[tuple[int, int], set[Node]], new_token_id: int, heap: list) -> list:
        """Merge the specified pair in all linked lists and update the frequency and position stats for affected pairs."""
        nodes_to_merge = list(pair_positions[best_pair]) # Get the head nodes of the best pairs
        for node_a in nodes_to_merge:
            if node_a.next is None or node_a.next.val != best_pair[1]: # If the pair is no longer valid, skip
                continue

            node_b = node_a.next
            left_neighbor = node_a.prev
            right_neighbor = node_b.next

            # Decrement old pair stats
            if left_neighbor is not None:
                left_pair = (left_neighbor, node_a)
                self._update_stats(left_pair, pair_counts, pair_positions, heap, decrement=True) # Update stats for the left pair
            if right_neighbor is not None:
                right_pair = (node_b, right_neighbor)
                self._update_stats(right_pair, pair_counts, pair_positions, heap, decrement=True) # Update stats for the right pair

            node_c = self._replace_pair(node_a, new_token_id) # New node after replacing the pair
            
            # Increment new pair stats
            if left_neighbor is not None:
                new_left_pair = (left_neighbor, node_c)
                self._update_stats(new_left_pair, pair_counts, pair_positions, heap, decrement=False)
            if right_neighbor is not None:
                new_right_pair = (node_c, right_neighbor)
                self._update_stats(new_right_pair, pair_counts, pair_positions, heap, decrement=False)

        del pair_positions[best_pair]
        del pair_counts[best_pair]

    def _update_stats(self, pair: tuple[Node, Node], pair_counts: dict[tuple[int, int], int],
                      pair_positions: dict[tuple[int, int], set[Node]], heap: list, decrement: bool) -> None:
        """Update the frequency and position stats for a given pair."""
        left_node = pair[0]
        token_pair = (pair[0].val, pair[1].val) # Get only the token values of the pair
        if decrement:
            pair_counts[token_pair] = pair_counts.get(token_pair, 0) - 1
            pair_positions[token_pair].discard(left_node) # Discard the old pair and its node position
            self._heap_push(heap, token_pair, pair_counts[token_pair]) # Push new count onto the heap
        else:
            pair_counts[token_pair] = pair_counts.get(token_pair, 0) + 1
            pair_positions[token_pair].add(left_node) # Add the position of the new pair
            self._heap_push(heap, token_pair, pair_counts[token_pair])

    def _pre_process(self, text: str) -> list[str]:
        """Apply regex patterns to clean the text and return text chunks."""
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