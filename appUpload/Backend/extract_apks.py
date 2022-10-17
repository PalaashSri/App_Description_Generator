import fnmatch
from multiprocessing import Pool
import os
import re
import subprocess
import nltk
from happytransformer import TTSettings
from PIL import Image
from nltk.corpus import words, stopwords
nltk.download("stopwords")
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.cluster import KMeans
import warnings
from happytransformer import HappyTextToText
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from time import sleep


@shared_task(bind=True)
def generate_description_from_apk(self,path):
    apk_generated_path=""
    method_names = ""
    activity_name = ""
    tokens = ""
    extracted_features = ""
    data = ""
    clustered_sentences = ""
    queries = ""
    happy_common_gen = ""
    beam_args = ""
    result = ""
    progress_recorder = ProgressRecorder(self)
    process_apk = Process_Apk()
    message="Setting up system"
    for i in range(0,100,10):
        sleep(1)
        progress_recorder.set_progress(i, 100, message)
        if i==0:
            message = "Processing apk"
        elif i==10:
            apk_generated_path = process_apk.apktools_on_apk(path)
            message = "Extracting method names"
        elif i==20:
            method_names = process_apk.extract_method_names(apk_generated_path)
            message = "Extracting activity names"
        elif i==30:
            activity_name = process_apk.extract_activity_name(apk_generated_path)
            message = "Extracting xml names"
        elif i==40:
            tokens = process_apk.extract_xml_names(apk_generated_path, method_names, activity_name)
            message = "Implementing SURF"
        elif i==50:
            extracted_features = process_apk.implement_SURF(tokens)
            message = "Converting features to vectors"
        elif i==60:
            data = process_apk.convert_features_to_vector(extracted_features)
            message = "Clustering words"
        elif i==70:
            clustered_sentences = process_apk.kmeans_to_cluster_words(data)
            message = "Extracting keywords from cluster"
        elif i==80:
            queries = process_apk.extract_keywords_from_cluster(clustered_sentences)
            message = "Generating description"
            beam_args = TTSettings(num_beams=5, min_length=1, max_length=100)
            happy_common_gen = HappyTextToText(load_path="model3/")
        elif i==90:
            # print(queries)
            input_text = "mobile application "
            for text in queries:
                input_text += " ".join(text.split()[:2])
            result = happy_common_gen.generate_text(input_text, args=beam_args).text
            print(result)
    return str(result)


class Process_Apk:

    def apktools_on_apk(self,apk_path):
        media_path=os.getcwd()+"/media/"
        os.chdir(media_path)
        print(os.getcwd())
        sp = subprocess.Popen("apktool d " + apk_path+".apk", stdin=subprocess.PIPE, shell=True, stdout=subprocess.PIPE)
        for line in iter(sp.stdout.readline, ''):
            print(line)
            if line.startswith(b'I: Copying original'):
                print("Done")
                break
        sp = subprocess.Popen("apktool b " + apk_path, stdin=subprocess.PIPE,shell=True, stdout=subprocess.PIPE)
        for line in iter(sp.stdout.readline, ''):
            print(line)
            if line.startswith(b'I: Built apk'):
                print("Done")
                break
        sp.stdin.close()  # close so that it will process

        os.chdir(os.getcwd()+"/"+apk_path)
        return os.getcwd()

    def remove_java_words(self,line):
        delete_list = ["_","-","$","bridge","From","Click","View","Layout","Drawable","View","Button","Menu","Title",
                       "Icon","All","Item","Down","Drop","<init>","<clinit>","synthetic","run","declared","-","create",
                       "execute","get","Value","value"," to"," To","String","has","Next","next","iterator"," on","set",
                       "false","true","abstract","assert","boolean","break","byte","case","catch","char","class",
                       "constructor","const","continue","default","do","double","else","enum","extends","final",
                       "finally","float","for","goto","if","implements","import","instanceof","int","interface","long",
                       "native","new","null","package","private","protected","public","return","short","static","super",
                       "switch","synchronized","this","throw","throws","transient","null","try","void","volatile",
                       "while"]

        for word in delete_list:
            line = line.replace(word, "")  # Replace all the java keyword with nothing

        return line

    def filter_word(self,word):
        i = 0
        try:
            tem = ""
            while (i < len(word)):
                if ((word[i] == ' ' and word[i + 1] <= ' ')):
                    i = i + 1
                elif (word[i] >= '0' and word[i] <= '9'):
                    i = i + 1
                elif ((word[i] == ' ' and ((word[i + 1] >= 'a' and word[i + 1] <= 'z') or (
                        word[i + 1] >= 'A' and word[i + 1] <= 'Z')) and word[i + 2] <= ' ')):
                    i = i + 2
                else:
                    tem += word[i]
                    i = i + 1
        except:
            i = 0
        return tem


    def extract_method_names(self,path):
        words = []
        for dirpath, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, '*.smali'):
                f = open(os.path.join(dirpath, filename), encoding="utf8")
                for word in f:
                    i = 8
                    if re.match("(.*)\.method(.*)", word):
                        tem = ""
                        try:
                            while (i < len(word)):
                                if (word[i] != '('):
                                    tem += word[i]
                                    i = i + 1
                                else:
                                    break
                        except IndexError:
                            i = 8
                        if (tem != "" and tem != " "):
                            words.append(tem)
        words = set(words)
        words = [self.remove_java_words(word) for word in words]
        words = [" ".join(re.findall('[A-Z][^A-Z]*', word)) for word in words]
        words = [word for word in words if word != ""]
        method_names = [self.filter_word(word) for word in words]
        return method_names

    def extract_activity_name(self,path):
        words = []
        f = open(path + "/AndroidManifest.xml", encoding="utf8")
        f = f.read()
        activity_name = re.findall(r"<activity.*? android:name=\"(.*?)\" .*?>", f)
        return activity_name

    def extract_xml_names(self,path,method_name,activity_name):
        warnings.filterwarnings('ignore')
        words = []
        for dirpath, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, 'strings.xml'):
                try:
                    i = 0
                    f = open(os.path.join(dirpath, filename), encoding='ISO-8859-1')
                    for word in f:
                        string_with_quotes = word
                        Find_double_quotes = re.compile('">([^>]*)</',
                                                        re.DOTALL | re.MULTILINE | re.IGNORECASE)  # Ignore case not needed here, but can be useful.
                        list_of_quotes = Find_double_quotes.findall(string_with_quotes)
                        list_of_quotes = str(list_of_quotes)

                        list_of_quotes = re.sub(r'[^A-Za-z\s]+', "", list_of_quotes)  # Remove any single letter
                        list_of_quotes = re.sub(r"\b[a-zA-Z]\b", "", list_of_quotes)  # Remove any single letter
                        words.append(list_of_quotes)
                except:
                    pass

        words = [word for word in words if word != "" and word != "\n"]
        xml_name = set(words)

        tokens = list(method_name)[:100] + list(activity_name) + list(xml_name)[:100]
        return tokens

    def description_token_library(self,des_paths):
        dic = set()
        for des_path in des_paths:
            with open(des_path, "r") as f:
                f = f.read()
                words = word_tokenize(f)
                words = [word for word in words if not word in stopwords.words()]
                dic.update(words)
        return dic

    def Extract_Features_with_single_POSPattern(self,pattern_1, tag_text):
        match_list = re.finditer(pattern_1, tag_text)

        app_features = []

        for match in match_list:
            app_feature = tag_text[match.start():match.end()]
            feature_words = [w.split("/")[0] for w in app_feature.split()]
            app_features.append(' '.join(feature_words))

        return (app_features)

    # function to remove stopwords
    def remove_stopwords(sefl,sen,stop_words):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    def implement_SURF(self,tokens):
        function_words = set(words.words())
        pos_patterns = [r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",  # 1
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)",  # 2
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",  # 3
                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",  # 4
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",  # 5
                        # r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN|VERB)", # 5 (old)
                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",  # 6
                        # r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN|ADJ|VERB)\s+[a-zA-Z-]+\/(NOUN)", # 6 (old)
                        r"[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/PRON\s+[a-zA-Z-]+\/(NOUN)",  # 7
                        # r"[a-zA-Z-]+\/(VERB|NOUN)\s+[a-zA-Z-]+\/PRON\s+[a-zA-Z-]+\/(NOUN)", # 7 (old)
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",  # 8
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",  # 9
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)",  # 10
                        r"[a-zA-Z-]+\/(NOUN)\s+(with|to)\/(ADP|PRT)\s+[a-zA-Z-]+\/(NOUN)",
                        # 11  (restriction prepositions)
                        # r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/ADP\s+[a-zA-Z-]+\/(NOUN)", # 11
                        # r"[a-zA-Z-]+\/(NOUN|ADJ)\s+[a-zA-Z-]+\/ADP\s+[a-zA-Z-]+\/(NOUN)", # 11 (old)
                        r"[a-zA-Z-]+\/(NOUN|VERB)\s+[a-zA-Z-]+\/(DET)\s+[a-zA-Z-]+\/(NOUN)",  # 12
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/ADP\s+[a-zA-Z-]+\/(NOUN)",  # 13
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/(NOUN)",  # 14
                        r"[a-zA-Z-]+\/ADJ\s+[a-zA-Z-]+\/CONJ\s+[a-zA-Z-]+\/ADJ",  # 15
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(PRON)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",  # 17
                        # r"[a-zA-Z-]+\/(VERB|NOUN)\s+[a-zA-Z-]+\/(PRON|DET)\s+[a-zA-Z-]+\/(ADJ|VERB|NOUN)\s+[a-zA-Z-]+\/(NOUN)", # 17 (old)
                        r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(ADP)\s+[a-zA-Z-]+\/(ADJ)\s+[a-zA-Z-]+\/(NOUN)",
                        # rule # 16
                        # r"[a-zA-Z-]+\/(VERB)\s+[a-zA-Z-]+\/(ADP)\s+[a-zA-Z-]+\/(ADJ|NOUN)\s+[a-zA-Z-]+\/(NOUN)", # rule # 16 (old)
                        r"[a-zA-Z-]+\/(NOUN)\s+[a-zA-Z-]+\/DET\s+[a-zA-Z-]+\/(ADJ|NOUN)\s+[a-zA-Z-]+\/(NOUN)"
                        # rule # 18  (RULE 18 IS REMOVED)
                        ]

        extracted_features = []
        # sentences=text.split("\n")
        for token in tokens:
            token = nltk.word_tokenize(token)
            # print(sent_tokens[:100])
            token = [w for w in token if w.lower() in function_words or not w.isalpha()]
            tag_tokens = nltk.pos_tag(token, tagset='universal')
            tag_text = ' '.join([word.lower() + "/" + tag for word, tag in tag_tokens])

            for pattern in pos_patterns:
                # store extracted features in list of app features
                # rule_name = 'POS_R%d' % (rule_counter)
                raw_features = self.Extract_Features_with_single_POSPattern(pattern, tag_text)
                if len(raw_features) != 0:
                    # app_features_pos_patterns.extend(raw_features)
                    for extracted_feature in set(raw_features):
                        extracted_features.append(extracted_feature)

        print(extracted_features[:100])
        print(len(extracted_features))
        # rule_counter = rule_counter + 1
        return extracted_features

    def convert_features_to_vector(self,extracted_features):
        # remove punctuations, numbers and special characters
        # clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        extracted_features = extracted_features[:1000]
        clean_sentences = [s.lower() for s in extracted_features]

        stop_words = stopwords.words('english')

        clean_sentences = [self.remove_stopwords(r.split(),stop_words) for r in clean_sentences]

        # Extract word vectors
        word_embeddings = {}
        os.chdir("../../appUpload/Backend/")
        #os.chdir("../")
        print(os.getcwd())
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        # similarity matrix
        sim_mat = np.zeros([len(extracted_features), len(extracted_features)])
        width, length = np.ogrid[:len(extracted_features), :len(extracted_features)]

        for i in range(len(extracted_features)):
            for j in range(len(extracted_features)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(extracted_features)), reverse=True)

        data = [word[1] for word in list(ranked_sentences[:500])]

        return data

    def kmeans_to_cluster_words(self,data):
        embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Corpus with example sentences
        corpus = data
        print("At kmeans cluster")
        corpus_embeddings = embedder.encode(corpus)

        # Perform kmean clustering
        num_clusters = 5
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        print(len(cluster_assignment))

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        return clustered_sentences

    def extract_keywords_from_cluster(self,clustered_sentences):
        queries = [" ".join(set(" ".join(c[:10]).split())) for c in clustered_sentences]
        print(queries)
        return queries


