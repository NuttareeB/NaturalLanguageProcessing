from torchtext import data 
from torchtext import datasets
from matplotlib import pyplot as plt
TEXT = data.Field(lower = True) 
UD_TAGS = data.Field(unk_token = None)
fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data , valid_data , test_data = datasets.UDPOS.splits(fields)

def visualizeSentenceWithTags(example):
    print("\nToken"+"".join([" "]*(15))+"POS Tag") 
    print("---------------------------------")
    for w, t in zip(example['text'], example['udtags']): 
        print(w+"".join([" "]*(20-len(w)))+t)  

# visualizeSentenceWithTags(vars(train_data.examples[997]))


# print(len(train_data.examples))
# for i in range(2000, 3000):
#     # visualizeSentenceWithTags(vars(train_data.examples[i]))
#     # print(' '.join(vars(train_data.examples[i])['text']))
#     print(vars(train_data.examples[i]))

# build vocaburary
TEXT.build_vocab(train_data)
UD_TAGS.build_vocab(train_data)

# print(TEXT.vocab.itos[2])

# # print(vars(TEXT))
# print(vars(TEXT.vocab)['stoi'])


# TEXT.numericalize


# TEXT.numericalize("new string")
# UD_TAGS.numericalize(train_data)

# word_vocab = TEXT.vocab
# tag_vocab = UD_TAGS.vocab

# print(vars(word_vocab))
# # print(vars(tag_vocab))

# BATCH_SIZE = 64
# train_iter, val_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), 
#                                                   batch_size=BATCH_SIZE,
#                                                   sort_key=lambda x: len(x.word),
#                                                   shuffle=True,
#                                                   repeat=False,
#                                                   device="cpu")


# # print(vars(train_iter))
# for batch in train_iter:
#     print(batch.text)
#     print(batch.udtags)
#     break

# print(len(train_data.examples))
# print(len(valid_data.examples))
# print(len(test_data.examples))
# tag_dict = {}
# tag_vals = []
# for ex in train_data.examples:
#     for tag in vars(ex)['udtags']:
#         tag_vals.append(tag)
#         if tag not in tag_dict:
#             tag_dict[tag] = 1
#         else:
#             tag_dict[tag] += 1

# print(tag_dict)
# print(tag_vals)

# tag_count_per_tag = UD_TAGS.vocab.freqs.most_common()
# tags = []
# counts = []
# for tag, count in tag_count_per_tag:
#     tags.append(tag)
#     counts.append(count)

# print(counts)
# plt.hist(counts,density=False)
# plt.show()

# total_no_tag = sum([count for tag, count in tag_count_per_tag])

# print("\nTag\t\tCount\t\tPercentage\n")
# for tag, count in tag_count_per_tag:
#     print(f"{tag}\t\t{count}\t\t{count/total_no_tag*100:4.2f}%")
