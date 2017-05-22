"""
Marvin Der Hann Lai 754672 
function library for machine learning project 2
USAGE:
On first run:
    leave out the file extension when specifying files to use, 
    assumes usage of json

    run_bag("train_file", "test_file","verify_file")
    or
    run_vec("train_file", "test_file", "verfy_file")

On subsequent runs with the same training file:

    run_bag("train_file", "test_file", "verify_file", retrain=False)
    or
    run_vec("train_file", "test_file", "verify_file", retrain=False)

This will train the model, classify test instances, verify and print accuracy
NOTE: assumes verify file and test file have same ordering of data points
"""
import json
import csv
import string
import time
import sys
import math
def clean_word(word):
    """removes punctuation from word"""
    stripped = ""
    word = word.lower()
    if not word[0] == '@' and not "http" in word:
        for chara in word:
            if chara not in string.punctuation and not chara == '\n':
                stripped += chara
    return stripped

def train_bag(trainfile):
    """Bag of Words classification model. Writes to /bag.json in model {"lang": "", words": ""}"""
    # assumes you give correct training data with lang tags
    # structured as {lang: {text: [%s, ..., %s], location:[%s, ..., %s], displayname:[%s,...,%s]}}
    data = {}
    #ignorable_loc = "on my the way to at of going where nearby close north south east west\
    #up down left right here there anywhere near far wherever you are"

    line_count = 0

    with open("%s.json" % (trainfile)) as json_raw: # loads entire json data
        for line_j in json_raw: # gets lines out of json data
            line_count += 1
            line = json.loads(line_j)
            # get words from json file

            if not line_count % 500:
                print("documenting line %d" % line_count)
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
    with open("%s.bag.json" % trainfile, 'w') as dest:
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

    #return data

def train_vec(trainfile):
    """
    Create a vectorised representation of a language.
    Meaning counts the occurunce of letters.
    """
    line_count = 0
    data = {}
    with open("%s.json" % trainfile, 'r') as raw_data:
        for line in raw_data:
            line_count += 1
            line = json.loads(line)

            if line["lang"] not in data:
                data[line["lang"]] = {}

            # Just processes words
            if not line_count % 500:
                print("documenting line %d" % line_count)
            for word in line["text"].split():
                for let in clean_word(word):
                    if let not in string.punctuation:
                        if let in data[line["lang"]]:
                            data[line["lang"]][let] += 1
                        else:
                            data[line["lang"]][let] = 1

    # no post processing, no trimming of less frequent letters

    with open("%s.vec.json" % trainfile, 'w') as vecout:
        for lang, let_count in data.items():
            vecout.write('{')
            vecout.write('"lang": "%s", "let": ' % lang)
            json.dump(let_count, vecout, sort_keys=True)
            vecout.write('}\n')

def classify_all_bag(testfile, modelfile):
    """
    compare all testing data to training data and record n-similarities to each lang
    perhaps stop if has similarity above 70% to a particular language
    perform same cleaning to testing as training (remove all punc, split on space)
    """

    #load training data first
    model_data = {}
    with open("%s.json" % modelfile) as train:
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
                    model_data[line["lang"]]["displayname"] = set(line["displayname"])

    lines = 0
    # renames "test.json" to "test.bag.cfd.csv"
    with open("%s.bag.cfd.csv" % testfile, 'w') as out:
        out.write('docid,lang\n')
        with open("%s.json" % testfile) as test:
            for line_t in test:
                line_t = json.loads(line_t)
                nmatches = possible_matches_bag(line_t, model_data)
                if len(nmatches) == 0:
                    this_lang = "unk"
                else:
                    this_lang = most_frequent(nmatches)

                #out.write('{"lang": "%s", "id": "%s"}\n' % (this_lang, line_t["id"]))
                out.write('test%04d,%s\n' % (lines, this_lang))
                #out.write('{"lang": "%s"}\n' % (this_lang))
                lines += 1


def possible_matches_bag(line, model_data):
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
            # split on spaces, e.g. Gorkha, Nepal and just Gorkha should be considered related.
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

def most_frequent(nmatches):
    """ returns language with the highest number of matches """
    most_frequent_lang = "unk"
    highest_freq = 3 # number chosen by incrementing and seeing where the accuracy level drops

    for lang, hits in nmatches.items():
        if hits >= highest_freq:
            highest_freq = hits
            most_frequent_lang = lang

    return most_frequent_lang

def get_clear_vector(vector):
    """ gets a new vector with same items as given vector with values 0'd """
    clean = {}
    for let in vector.keys():
        clean[let] = 0

    return clean

def classify_all_vec(testfile, modelfile):
    """ classifies according to the vectorised model """
    model_data = {}
    with open("%s.json" % modelfile) as model:
        # reload model data
        for line in model:
            line = json.loads(line)

            if line["lang"] not in model_data:
                model_data[line["lang"]] = line["let"]
            #print(model_data)
    # model data loaded
    # writing the file.cfd.csv as the classified values
    lines = 0
    with open("%s.vec.cfd.csv" % testfile, 'w') as out:
        out.write('docid,lang\n')
        with open("%s.json" % testfile) as test:
            for line_t in test:
                line_t = json.loads(line_t)
                matches = possible_matches_vec(line_t, model_data)
                this_lang = highest_cos(matches)

                out.write('test%04d,%s\n' % (lines, this_lang))

                lines += 1

def possible_matches_vec(line, model_data):
    """ returns list of all cosine angle between line and model_data """
    nmatches = {}

    for lang, let_dict in model_data.items():
        zero_vect = get_clear_vector(let_dict)

        for word in line["text"].split():
            for let in clean_word(word):
                if let not in string.punctuation:
                    if let in zero_vect:
                        zero_vect[let] += 1

        nmatches[lang] = let_cos(zero_vect, let_dict)
    #print(line["lang"])
    #print(nmatches)
    return nmatches

def let_cos(v1, v2):
    """ assumes v1 and v2 have same features """

    ab = 0
    a = 0
    b = 0

    for let in v1.keys():
        ab += v1[let] * v2[let]
        a += v1[let] * v1[let]
        b += v2[let] * v2[let]

    if not a or not b:
        return 0
    return 1.0 * ab/(math.sqrt(a) * math.sqrt(b))

def highest_cos(nmatches):
    """ returns the language with the highest cos score above 0.6 """
    highest_val = 0
    highest_lang = "unk"
    for lang, cos in nmatches.items():
        if cos > highest_val:
            highest_val = cos
            highest_lang = lang
    return highest_lang

def check_accuracy_csv(trueclasses, classifiedfile):
    """ check accuracy modified for csv"""
    trueclasses = "%s.json" % trueclasses
    classifiedfile = "%s.csv" % classifiedfile

    langs = {}

    totalinstances = 0
    totalcorrect = 0
    with open(trueclasses) as verify:
        with open(classifiedfile) as classified:
            cfd_row = csv.reader(classified, delimiter=',')
            next(cfd_row)
            for cfd, inp in zip(cfd_row, verify):
                inp = json.loads(inp)
                if inp["lang"] not in langs:
                    langs[inp["lang"]] = {}
                    langs[inp["lang"]]["total"] = 0
                    langs[inp["lang"]]["correct"] = 0

                if cfd[1] == inp["lang"]:
                    totalcorrect += 1
                    langs[inp["lang"]]["correct"] += 1

                langs[inp["lang"]]["total"] += 1
                totalinstances += 1

    for lang, acc in langs.items():
        print(lang, acc, "acc: %.2f%%" % (100.0* acc["correct"]/acc['total']))

    print("%d languages classified" % len(langs))
    print("Total Correct: %d" % totalcorrect)
    print("Total Instances: %d" % totalinstances)
    print("Accuracy %.2f%%" %  (100.0*totalcorrect/totalinstances))

def run_bag(train_root, classify_root, verify_root="", retrain=True, verify=True):
    ''' runs and outputs models and classifications as a file and does accuracy check
    input run(file to train on, file with correct classifications) both without .json'''
    if retrain:
        sys.stdout.write('Training...')
        train_bag(train_root)
        sys.stdout.write('Complete\n')
        sys.stdout.flush()

        time.sleep(1)

    sys.stdout.write('Classifying...')
    classify_all_bag(classify_root, train_root + '.bag')
    sys.stdout.write('Complete\n')
    sys.stdout.flush()

    time.sleep(1)

    if verify and verify_root:
        print("Accuracy")
        check_accuracy_csv(verify_root, classify_root + '.bag.cfd')
        print("Complete")

    print("Check output file %s.bag.cfd" % verify_root)

def run_vec(train_root, classify_root, verify_root="", retrain=True, verify=True):
    ''' runs and outputs models and classifications as a file and does accuracy check
    input run(file to train on, file with correct classifications) both without .json'''
    if retrain:
        sys.stdout.write('Training...')
        train_vec(train_root)
        sys.stdout.write('Complete\n')
        sys.stdout.flush()

        time.sleep(1)

    sys.stdout.write('Classifying...')
    classify_all_vec(classify_root, train_root + '.vec')
    sys.stdout.write('Complete\n')
    sys.stdout.flush()

    time.sleep(1)

    if verify and verify_root:
        print("Accuracy")
        check_accuracy_csv(verify_root, classify_root + '.vec.cfd')
        print("Complete")

    print("Check output file %s.vec.cfd" % verify_root)

