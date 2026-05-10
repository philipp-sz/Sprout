from sprout.tokenizer import FastBPETokenizer


vocab_size = 32_768
training_data_path = "./training_data.txt"
save_path = "./bpe_tokenizer_state2.json"

tokenizer = FastBPETokenizer(vocab_size=vocab_size)
# tokenizer = FastBPETokenizer.load_tokenizer_state(save_path)
tokenizer.train(training_data_path)
tokenizer.save(save_path)