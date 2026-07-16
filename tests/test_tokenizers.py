import pathlib

import pytest

from sprout.tokenizer import BPETokenizer, FastBPETokenizer


STATE_PATH = pathlib.Path(__file__).parent.parent / "sprout" / "tokenizer" / "states" / "bpe_tokenizer_state.json"

@pytest.fixture(scope="session", params=[BPETokenizer, FastBPETokenizer])
def tokenizer(request):
    return request.param.load(STATE_PATH)

class TestTokenizers:
    @pytest.mark.parametrize("text", [
        "The quick brown fox jumps over the lazy dog.",
        "每个人都有他的作战策略。",
        "Hello 🌟! It's  2026.",
        "This is the eot token: <|endoftext|>"
    ])
    def test_encode_decode_round_trip(self, tokenizer, text):
        """Test that encoding and then decoding returns the original text."""
        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)
        assert decoded_text == text, f"Decoded text '{decoded_text}' does not match original '{text}'"
        assert all(token_id in tokenizer.vocab for token_id in token_ids), f"Token ID {id} in encoded output is not in the tokenizer's vocabulary"

    @pytest.mark.parametrize("tokenizer_cls", [BPETokenizer, FastBPETokenizer])
    def test_special_tokens(self, tokenizer_cls):
        """Test that special tokens are correctly stored, encoded and decoded."""
        tokens = {"<|startoftext|>": 32768, "<|pad|>": 32769, "<|unk|>": 32770}
        tokenizer = tokenizer_cls(vocab_size=32_768, special_tokens=tokens)
        for token, token_id in tokens.items():
            assert tokenizer.special_tokens[token] == token_id
            assert tokenizer.encode(token) == [token_id]
            assert tokenizer.decode([token_id]) == token

    def test_save_load_consistency(self, tokenizer, tmp_path):
        text = "The quick brown fox jumps over the lazy dog."
        ids_before = tokenizer.encode(text)
        path = tmp_path / "state.json"
        tokenizer.save(str(path))
        reloaded = type(tokenizer).load(str(path))
        assert reloaded.encode(text) == ids_before