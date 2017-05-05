"""function library for machine learning project 2"""
import json
import string
import ast

def load_data(filename):
    """loads file in data folder as specified by filename"""
    data = []
    with open("data/%s" % (filename)) as load:
        for line in load:
            data.append(json.loads(line))
    return data

def count_n_test_id(filename):
    """
    counts the total number of lines with a test id
    from this we learn there is a 1:1 correspondance of test_id to n_lines
    """
    lines = 0
    test_id = 0
    uid = 0
    with open("data/%s" % filename) as c_test:
        for line in c_test:
            lines += 1
            if "id" in line:
                test_id += 1
            if "uid" in line:
                uid += 1
    print("lines %d"%lines)
    print("test_id %d"%test_id)
    print("uid %d"%uid)


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

def train_bag(filename):
    """Bag of Words classification model. Writes to /bag.json in model {"lang": "", words": ""}"""
    # assumes you give correct training data with lang tags
    # structured as {lang: {text: [%s, ..., %s], location:[%s, ..., %s]}}
    data = {}
    #ignorable_loc = "on my the way to at of going where nearby close north south east west\
    #up down left right here there anywhere near far wherever you are"

    with open("data/%s" % (filename)) as json_raw: # loads entire json data
        for line_j in json_raw: # gets lines out of json data
            line = json.loads(line_j)
            # get words from json file

            # create new entry for a new language
            if line["lang"] not in data:
                # dict of language contains dict of text and location
                data[line["lang"]] = {"text":set(), "location":set()}

            # words
            # a conscious choice was made to strip all punctuation from words. All. Punctuation
            # too many edge cases where a word may not want to be split on . or ,
            # e.g. if people accidentally type like.this or,this but don't want to split links
            # deemed to be insignificant anyway. for now
            # significant errors in dealing with non-alphanumeric and non-spaced languages
            for word in line["text"].split():
                word = word.strip(" ").lower()
                stripped = ""

                # strip punctuation off words, already split on spaces
                # user names are kept as repeated interactions between users are
                # likely to be in the same language
                for chara in word:
                    if chara not in string.punctuation:
                        stripped += chara

                # add to text data if it's not already there and it's not empty
                if stripped:
                    if stripped not in data[line["lang"]]["text"]:
                        data[line["lang"]]["text"].add(stripped)

            # location
            # some data doesn't have location
            if "location" in line:
                loc = line["location"].lower()
                if loc == "not-given" or loc == "here":
                    loc = ""
            else:
                loc = ""

            if loc:
                # split on commas, e.g. Gorkha, Nepal and just Gorkha should be considered related
                for newloc in loc.split():
                    # remove lead/trailing whitespace and cast to lower case
                    newloc = newloc.strip(" ").lower()
                    # remove punctuation
                    subloc = ""
                    '''
                    # remove potentially irrelevant data
                    if subloc not in ignorable_loc:
                        newloc = ""
                    '''
                    for chara in newloc:
                        if chara not in string.punctuation:
                            subloc += chara

                    # add to location if not already in
                    if subloc and subloc not in data[line["lang"]]["location"]:
                        data[line["lang"]]["location"].add(subloc)

    # now that we're all loaded, write model to file
    # write as {"lang" : lang, "text": [%s,...,%s], "loc": [%s,...,%s]}
    # convert sets back into lists
    model_name = "model/%s.bag.json" % filename.strip(".json")
    dest = open(model_name, 'w') # rewrites the file every time

    for lang in data.keys():
        #data[lang]["text"] = list(data[lang]["text"])
        #data[lang]["location"] = list(data[lang]["location"])

        # lang is the label, text_loc is the sub dict of text and location
        dest.write('{')
        dest.write('"lang": "%s"' % lang)

        # each new segment is responsible writing the ", " and label in front of it

        if data[lang]["text"]:
            dest.write(', "text": ')
            json.dump(data[lang]["text"], dest)

        if data[lang]["location"]:
            dest.write(', "location": ')
            json.dump(data[lang]["location"], dest)

        dest.write("}\n")
    dest.close()
    return data

def similarity(testfile, trainfile):
    """
    compare all testing data to training data and record n-similarities to each lang
    perhaps stop if has similarity above 70% to a particular language
    perform same cleaning to testing as training (remove all punc, split on space)
    """

    #load training data first
    training_data = {}
    with open("model/%s" % trainfile) as train:
        for line in json.load(train):
            print(line)
