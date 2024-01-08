import tiktoken


string_to_encode = "hii there"
# enc = tiktoken.get_encoding("cl100k_base")
# assert enc.decode(enc.encode(string_to_encode)) == string_to_encode

# # To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")

# enc = tiktoken.get_encoding("gpt2")
# assert enc.decode(enc.encode(string_to_encode)) == string_to_encode

ecoded = enc.encode("Dogs chase cats.")
print(f"Vocabulary size: ", enc.n_vocab)
print(ecoded)
for x in ecoded:
    make_list = [x]
    print(enc.decode(make_list))
# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4")