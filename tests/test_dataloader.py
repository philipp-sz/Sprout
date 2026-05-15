import numpy as np
import pytest
from sprout.data.data_processing import TokenDataLoader


def make_shard(path, tokens: list[int]) -> None:
    np.save(path, np.array(tokens, dtype=np.uint16))


@pytest.fixture
def shard_dir(tmp_path):
    make_shard(tmp_path / "shard_000000.npy", list(range(0,   200))) # val
    make_shard(tmp_path / "shard_000001.npy", list(range(200, 400))) # train 1
    make_shard(tmp_path / "shard_000002.npy", list(range(400, 600))) # train 2
    return tmp_path

def test_batch_shape_and_offset(shard_dir):
    """Test that get_batch returns tensors of the correct shape and that y is x shifted by one token."""
    loader = TokenDataLoader(shard_dir, batch_size=2, seq_len=8, device="cpu", split="val")
    x, y = loader.get_batch()

    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
    assert (y == x + 1).all()

def test_train_val_split(shard_dir):
    """Test that the val split returns tokens from the first shard and the train split returns tokens from the subsequent shards."""
    val_loader = TokenDataLoader(shard_dir, batch_size=2, seq_len=8, device="cpu", split="val")
    x_val, _ = val_loader.get_batch()
    assert x_val.min() >= 0 and x_val.max() < 200

    train_loader = TokenDataLoader(shard_dir, batch_size=2, seq_len=8, device="cpu", split="train")
    x_train, _ = train_loader.get_batch()
    assert x_train.min() >= 200 and x_train.max() < 600

def test_epoch_wraparound(shard_dir):
    """Test if the iterator restarts correctly when the end of the shard is reached."""
    B, T = 2, 8
    loader = TokenDataLoader(shard_dir, batch_size=B, seq_len=T, device="cpu", split="val")

    batches_in_shard = (200 - 1) // (B * T) # Number of full batches that can be drawn from the shard before reaching the end
    for _ in range(batches_in_shard):
        loader.get_batch()

    x, y = loader.get_batch()

    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.min() >= 0 and x.max() < 200 # Should still be drawing from the val shard