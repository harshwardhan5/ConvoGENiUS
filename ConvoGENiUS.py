import nltk
import random
import string
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')
f = open('/content/science.yml', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()
sent_tokens = nltk.sent_tokenize(raw) #converts to list of scentences
word_tokens = nltk.word_tokenize(raw) #converts to list of words
sentToken = sent_tokens[:4]
#print(sentToken)
wordToken = word_tokens[:4]
#print(wordToken)
#preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]
def greeting(scentence):
    for word in scentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
#Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
    chatbot_response = ''
    # Append the user's response to the existing sent_tokens list
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Calculate cosine similarities
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        chatbot_response = "I am sorry! I don't understand you"
    else:
        chatbot_response = sent_tokens[idx]

    #print values of cosine similarity
    #print(vals)

    return chatbot_response
if __name__ == "__main__":
    flag = True
    print("Hello, there my name is convoGENiUS. I will answer your queries. If you want to exit, type Bye!")
    while(flag==True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response!='bye'):
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print("convoGENiUS: You're welcome!")
            else:
                if(greeting(user_response)!=None):
                    print("convoGENiUS:" +greeting(user_response))
                else:
                    print("convoGENiUS:", end='')
                    print(response(user_response))
                    if user_response in sent_tokens:
                      sent_tokens.remove(user_response)
        else:
            flag = False
            print("convoGENiUS: Bye! Have a great time!" )