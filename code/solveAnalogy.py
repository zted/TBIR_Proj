filename = '../questions-words.txt'

count = 0
with open(filename, 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        words = line.split(' ')
        print(words)
        if words[0] == ':':
            continue
        a = words[0]
        b = words[1]
        c = words[2]
        answer = words[3]
        count += 1
        if count > 5:
            break

filename = '../glove.6B/glove.6B.50d.txt'

def createIndices(someFile):
    someDict = {}
    count = 0
    with open(filename, mode='r', encoding="ISO-8859-1") as f:
        lines = f.read().splitlines()
        for line in lines:
            count += 1
            word = line.split(' ')[0]
            someDict[word] = count
    return someDict




num_lines = sum(1 for line in open(filename, encoding="ISO-8859-1"))
print(num_lines)
a = [1] * (3000000)
# count = 0
# with open(filename,'r') as f:
#     for i,l in enumerate(f):
#             pass
#     print(i+1)

# def fetchVectors(a,b,c)