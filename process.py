"""function library for machine learning project 2"""
import json
import string
from pprint import pprint

def load_data(filename):
    "loads file in data folder as specified by filename"
    data = []
    with open("data/%s" % (filename)) as load:
        for line in load:
            data.append(json.loads(line))
    return data

def print_text_data(data):
    """prints only the text segment of each data point line by line"""
    '''
    data = []
    with open("data/%s" % (filename)) as load:
        for line in load:
            data.append(json.loads(line))
    '''
    for data_point in data:
        try:
            print(data_point["lang"].encode('utf8', 'strict') + " "
                  + data_point['text'].encode('utf8', 'strict'))
        except KeyError:
            print("no lang" + data_point['text'].encode('utf8', 'strict'))
        except UnicodeEncodeError:
            print("Can't print line")

def raw_print_json(filename):
    """prints raw data in json file"""
    with open("data/%s" % filename) as json_raw:
        dat = json.load(json_raw)
        print(dat)

def train_bag(filename):
    """Bag of Words classification model. Writes to /bag.json in model {"lang": "", words": ""}"""
    # assumes you give correct training data with lang tags
    # structured as {lang: {text: [%s, ..., %s], location:[%s, ..., %s]}}
    data = {}

    with open("data/%s" % (filename)) as json_raw: # loads entire json data
        for line_j in json_raw: # gets lines out of json data
            line = json.loads(line_j)
            # get words from json file

            # create new entry for a new language
            if line["lang"] not in data:
                # dict of language contains dict of text and location
                data[line["lang"]] = {"text":[], "location":[]}

            # words
            for word in line["text"].split():
                stripped = ""

                # strip punctuation off words
                # user names are kept as repeated interactions between users are
                # likely to be in the same language
                for chara in word:
                    if chara not in string.punctuation:
                        stripped += chara

                # add to text data if it's not already there and it's not empty
                if stripped:
                    if stripped not in data[line["lang"]]["text"]:
                        data[line["lang"]]["text"].append(stripped.lower())

            # location
            # some data doesn't have location
            try:
                loc = line["location"].lower()
                if loc == "not-given" or loc == "here":
                    loc = ""

            except KeyError:
                loc = ""

            if loc != "":
                # split on commas
                for newloc in loc.split(","):
                    # remove lead/trailing whitespace and cast to lower case
                    newloc = newloc.strip(" ").lower()

                    # remove punctuation
                    subloc = ""
                    for chara in newloc:
                        if chara not in string.punctuation:
                            subloc += chara

                    # add to location if not already in
                    if subloc not in data[line["lang"]]["location"]:
                        data[line["lang"]]["location"].append(subloc)

    # now that we're all loaded, write model to file
    # write as {"lang" : lang, "text": [%s,...,%s], "loc": [%s,...,%s]}
    model_name = "model/%s.bag.json" % filename.strip(".json")
    dest = open(model_name, 'w') # rewrites the file every time
    for lang, info in data.items():
        # lang is the label, text_loc is the sub dict of text and location
        dest.write("{")
        dest.write("\"lang\": \"%s\"" % lang)

        # each new segment is responsible writing the ", " in front of it

        if info["text"]:
            dest.write(", \"text\": [")
            dest.write(", ".join(info["text"]).encode("unicode_escape"))
            dest.write("]")

        if info["location"]:
            dest.write(", \"location\": [")
            dest.write(", ".join(info["location"]).encode("unicode_escape"))
            dest.write("]")

        dest.write("}\n")

    dest.close()

    return data
