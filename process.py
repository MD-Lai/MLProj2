"""function library for machine learning project 2
TODO remove all folder dependencies """
import json
import string
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

def clean_word(word):
    """removes punctuation from word"""
    stripped = ""
    for chara in word.lower():
        if chara not in string.punctuation:
            stripped += chara
    return stripped

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
                data[line["lang"]] = {"text":set(), "location":set(), "displayname":set()}

            # words
            # a conscious choice was made to strip all punctuation from words. All. Punctuation
            # too many edge cases where a word may not want to be split on . or ,
            # e.g. if people accidentally type like.this or,this but don't want to split links
            # deemed to be insignificant anyway. for now
            # significant errors in dealing with non-alphanumeric and non-spaced languages
            for word in line["text"].split():
                stripped = clean_word(word)

                # add to text data if it's not already there and it's not empty
                if stripped:
                    if stripped not in data[line["lang"]]["text"]:
                        data[line["lang"]]["text"].add(stripped)

            # location
            # some data doesn't have location
            if "location" in line:
                loc = line["location"]
                if loc == "not-given":
                    loc = ""
                    # split on spaces, e.g. Gorkha, Nepal and just Gorkha should be related
                    # jeddah , saudi arabia would be split to [jeddah, saudi, arabia]
                for subloc in loc.split():
                    # split handles removing spaces b/w characters as well as stripping
                    # remove punctuation
                    subloc = clean_word(subloc)

                    # add to location if not already in
                    if subloc and subloc not in data[line["lang"]]["location"]:
                        data[line["lang"]]["location"].add(subloc)


            # display name
            if "displayname" in line:
                dispname = line["displayname"]

                for subname in dispname.split():
                    # split handles removing spaces b/w characters as well as stripping
                    # remove punctuation
                    subname = clean_word(subname)

                    # add to displaynames if not already in
                    if subname and subname not in data[line["lang"]]["displayname"]:
                        data[line["lang"]]["displayname"].add(subname)


    # now that we're all loaded, write model to file
    # write as {"lang" : lang, "text": [%s,...,%s], "loc": [%s,...,%s], "displayname": [%s,...,%s]}
    with open("model/%s.bag.json" % filename.strip(".json"), 'w') as dest:
        for lang, info in data.items():

            # lang is the label
            dest.write('{')
            dest.write('"lang": "%s"' % lang)

            # each new segment is responsible writing the ', "%s"' in front of it

            if info["text"]:
                dest.write(', "text": ')
                json.dump(list(info["text"]), dest)

            if info["location"]:
                dest.write(', "location": ')
                json.dump(list(info["location"]), dest)

            if info["displayname"]:
                dest.write(', "displayname": ')
                json.dump(list(info["displayname"]), dest)

            dest.write("}\n")

    return data

def classify_all(testfile, modelfile):
    """
    compare all testing data to training data and record n-similarities to each lang
    perhaps stop if has similarity above 70% to a particular language
    perform same cleaning to testing as training (remove all punc, split on space)
    """

    #load training data first
    model_data = {}
    with open("model/%s" % modelfile) as train:
        for line_j in train:
            line = json.loads(line_j)
            # assumes perfect input, no repeated language
            # if language not loaded, create a new bag -> this should always happen
            if line["lang"] not in model_data:
                model_data[line["lang"]] = {"text":set(), "location":set(), "displayname":set()}

                if "text" in line:
                    model_data[line["lang"]]["text"] = set(line["text"])

                if "location" in line:
                    model_data[line["lang"]]["location"] = set(line["location"])

                if "displayname" in line:
                    model_data[line["lang"]]["displayname"] = set(line["location"])

    with open("output/%s.cfd.json" % testfile.strip(".json"), 'w') as out:
        with open("test/%s" % testfile) as test:
            for line_t in test:
                line_t = json.loads(line_t)
                nmatches = classify_row(line_t, model_data)
                if len(nmatches) == 0:
                    this_lang = "unk"
                else:
                    this_lang = most_frequent(nmatches)

                out.write('{"lang": "%s", "id": "%s"}\n' % (this_lang, line_t["id"]))


def classify_row(line, model_data):
    """ classifies a test line based on given training data """
    formatted_line = {"text":set(), "location":set(), "displayname":set()}

    # decompose it into text, location, user, if available
    # test_line won't have language
    # cleaning the input line the same as all the above
    # i.e. lower case, split into words, strip punctuation

    if "text" in line:
        # lower case, strip, clean
        for word in line["text"].split():
            # split handles removing spaces b/w characters as well as stripping
            # remove punctuation
            word = clean_word(word)
                # add to text data if it's not already there and it's not empty
            if word and word not in formatted_line["text"]:
                formatted_line["text"].add(word)

    # location
    # some data doesn't have location
    if "location" in line:
        loc = line["location"]
        if loc == "not-given":
            loc = ""
            # split on spaces, e.g. Gorkha, Nepal and just Gorkha should be considered related
            # jeddah , saudi arabia would be split to [jeddah, saudi, arabia]
        for subloc in loc.split():
            # split handles removing spaces b/w characters as well as stripping
            # remove punctuation
            subloc = clean_word(subloc)

            # add to location if not already in
            if subloc and subloc not in formatted_line["location"]:
                formatted_line["location"].add(subloc)

    if "displayname" in line:
        for subname in line["displayname"].split():
            subname = clean_word(subname)

            if subname and subname not in formatted_line["displayname"]:
                formatted_line["displayname"].add(subname)


    nmatches = {} # {"%s": number} % language

    # compare each item in data to field_data
    # compare text->text location->location displayname->displayname
    # or text->text text->location text->displayname location->text etc...
    # probably just 1:1 mapping

    for field, field_data in formatted_line.items():
        for lang, data in model_data.items():
            nlang = len(field_data.intersection(data[field]))

            if nlang > 0:
                if lang not in nmatches:
                    nmatches[lang] = nlang
                else:
                    nmatches[lang] += nlang

    return nmatches
    """
    for field, field_data in formatted_line.items():
        # field is "text" "location" "displayname"
        # field_data is formatted_line[field] (which is also a set)
        for lang, model_data in model_data.items():
            # lang being the language with these words
            # data should be {"text":set(), "location":set(), "displayname":set()}
            if lang not in nmatches:
                nmatches[lang] = 0

            # what you're really doing is comparing everything in the model to items in the line
            for f_data in field_data:
                if field in model_data:
                    for
    """

def most_frequent(nmatches):
    """ returns language with the highest number of matches """
    most_frequent_lang = ""
    highest_freq = 0

    for lang, hits in nmatches.items():
        if hits > highest_freq:
            highest_freq = hits
            most_frequent_lang = lang

    return most_frequent_lang

def check_accuracy(trueclasses, classifiedfile):
    """ checks accuracy of classifier based on given files
    TODO store correct/total for each language"""
    # assumes you've ran train_bag("training.json")
    # and classify_all("unclassified.json", "training.bag.json")
    trueclasses = "verify/%s" % trueclasses
    classifiedfile = "output/%s" % classifiedfile

    totalinstances = 0
    totalcorrect = 0

    with open(trueclasses) as verify:
        with open(classifiedfile) as classified:
            for cfd, inp in zip(classified, verify):
                cfd = json.loads(cfd)
                inp = json.loads(inp)
                totalcorrect += 1 if cfd["lang"] == inp["lang"] else 0
                totalinstances += 1
    print("Accuracy %f" %  (1.0*totalcorrect/totalinstances))
