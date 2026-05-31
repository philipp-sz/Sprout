from sprout.tokenizer import FastBPETokenizer


vocab_size = 32_767 # Leave one free for the eot token
training_data_path = "./tokenizer_data_mixed_20k.txt"
save_path = "./bpe_mixed_32k_20k_state.json"

tokenizer = FastBPETokenizer(vocab_size=vocab_size)
# tokenizer = FastBPETokenizer.load_tokenizer_state(save_path)
tokenizer.train(training_data_path)
tokenizer.save(save_path)