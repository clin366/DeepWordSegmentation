#encoding=UTF-8
import gensim
import sys

def generate_sentences(lines):
    sentences = []
    for line in lines:
        line = line.replace("\n", "")
        if line.find("\t") == -1:
            print line
            continue;
        sentence = []
        line = line.split("\t")[1]
        words = line.split(";;")
        for word in words:
            if word != '':
                sentence.append(word)
        sentences.append(sentence)
    return sentences

def main():
    file_object = open(sys.argv[1])
    write_object = open(sys.argv[2], "w")
    out_nearline = open(sys.argv[3], "w")
    oskey = sys.argv[5]
    out_dict = {}
    try:
        sentences = generate_sentences(file_object)
        model = gensim.models.Word2Vec(sentences, size=50, window=5, min_count=5, workers=20)
        model.save(sys.argv[4])
        for v in model.wv.vocab:
            results = model.most_similar(positive=[v], topn=20)
            s = v + '\t'
            for word, relation in results:
                s += word + '::' + str(relation) + ',,'
                if out_dict.has_key(word):
                    out_dict[word] = out_dict[word] + ',,' + v + '::' + str(relation)
                else:
                    out_dict[word] = v + '::' + str(relation)
            write_object.write(s + "\n")
        for key, value in out_dict.items():
            out_nearline.write(oskey + '_#%' + key +'\t' + value + '\n')
    except Exception as e:
        print e.message
    finally:
        file_object.close()
        write_object.close()
        out_nearline.close()

main()
