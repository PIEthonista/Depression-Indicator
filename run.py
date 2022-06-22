import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import contractions
from tensorflow.keras.models import load_model

model_dir = "./Model/Hot_Stuff"
model = load_model(model_dir)

print("Model Loaded")


stopwords = set(stopwords.words('english'))


def predict(model, input, stopwords=stopwords):

    # cleaning input data
    input = input.lower()
    input = input.split(" ")
    input = [re.sub(r"[^\w\d\s\']+", "", x) for x in input]
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)

    input = [re.sub(emoj, "", x) for x in input]
    input = [x.encode("ascii", "ignore").decode() for x in input]
    input = [x for x in input if x != ""]
    input = [contractions.fix(x) for x in input]
    input = ' '.join(map(str, input))
    input = word_tokenize(input, language="english")
    input = [x for x in input if x not in stopwords]
    input = [x for x in input if not x.isdigit()]
    input = ' '.join(map(str, input))
    filtered_text = input

    x_input = np.array([input], dtype=np.string_)
    y = model.predict(x_input)
    result = ""
    if y[0][0] == y[0].max():
        result = "No Depression"
    elif y[0][1] == y[0].max():
        result = "Depression Detected"
    elif y[0][2] == y[0].max():
        result = "Suicidal Depression"

    return (result, y, filtered_text)


while True:
    x = input("Enter Text: ")
    if x == "exit":
        break
    y = predict(model, x)
    print("Filtered Text : " + y[2])
    print("Model Result  : " + y[0])
    print("No Depression       :", y[1][0][0])
    print("Depression Detected :", y[1][0][1])
    print("Suicidal Depression :", y[1][0][2])
    print("")
    print("---------------------------------------------------")

print("Completed")
