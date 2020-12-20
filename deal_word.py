
word_list = set()
for line in open('data/label_train.txt','r'):
    word = line.split()[1]
    for i in word:
        word_list.add(i)
print(len(word_list))

dict = {}
i = 0
for word in word_list:
    dict[word] = i
    i+=1

f = open('./data/new_word_onehot.txt', 'w')
f.write(str(dict))
f.close()

