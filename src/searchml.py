# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
from __future__ import print_function
import pickle
import os
import subprocess
from optparse import OptionParser

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Parse args
def getArgs():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
        help="data file", metavar="FILE")
    parser.add_option("-s", "--treefile", dest="treefile",
        help="decision tree file", metavar="FILE")
    parser.add_option("-t", "--target",
        dest="target", default=-1,
        help="target column, by default last one in the dataset")
    parser.add_option("-r", "--nrows",
        dest="nrows", default=-1,
        help="number of rows to read")
    parser.add_option("-d", "--data",
        dest="data", help="data to test")
    parser.add_option("", "--testFile",
        dest="testFile", help="data file to test")
    parser.add_option("-v", "--visualize",
        dest="visualize", help="Create visualizaton")
    parser.add_option("-b", "--boolean",
        dest="boolean", help="Transform target to boolean")
    return parser.parse_args()

(options, args) = getArgs()
print(str(options))
print(str(args))

def get_data(dataFile):
    """Get the iris data, from local csv or pandas repo."""
    nrows = int(options.nrows) if int(options.nrows) <> -1 else None
    print("File found, reading %s rows" % nrows)
    df = pd.read_csv(dataFile, index_col=0, nrows=nrows)
    return df


def visualize_tree(treefile, tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    dotFile = treefile + ".dot"
    pdfFile = treefile + ".pdf"
    with open(dotFile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpdf", dotFile, "-o", pdfFile]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def saveTree(treefile, tree):
    with open(treefile, 'wb') as output:
        pickle.dump(tree, output)
def loadTree(treefile):
    dt = None
    if os.path.exists(treefile):
        dt = pickle.load(open(treefile, "rb"))
    return dt  

def buildTree(options, treefile, dataFile = None):
    dt = loadTree(treefile)
    if dt is not None:
        return dt
    if dataFile is None:
        raise ValueError("No data file specified")

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    files = []
    featureFrames = []
    targetFrames = []
    if os.path.isdir(dataFile):
        files = getFiles(dataFile, ".csv")
    else:
        files.append(dataFile)
    for _file in files:
        print("Loading data %s" % _file)
        (featureValues, targetValues, features, df) = loadData(_file, options)
        featureFrames.append(featureValues)
        targetFrames.append(targetValues)
    dt.fit(pd.concat(featureFrames), pd.concat(targetFrames))
    saveTree(treefile, dt)
    print("Building graph")
    visualize_tree(treefile, dt, features)
    return dt

def loadData(dataFile, options):
    df = get_data(dataFile)
    #print("* df.head()", df.head(), sep="\n", end="\n\n")
    #print("* df.tail()", df.tail(), sep="\n", end="\n\n")

    targetColumn = options.target if int(options.target) <> -1 else len(df.columns)-1
    #print("Target column %s (%s)" % (targetColumn, df.columns[targetColumn]))
    #if options.boolean is not None:
    #    print("Converting target to boolean")
    #    df[df.columns[targetColumn]] = df[df.columns[targetColumn]].astype(bool)

    features = list(df.columns[:targetColumn])
    #print("* features:", features, sep="\n")

    featureValues = df[features]
    targetValues = df[df.columns[targetColumn]]
    if options.boolean is not None:
        print("Converting target to boolean")
        targetValues = targetValues.map(lambda x: 0 if x == 0 else 1)
    return (featureValues, targetValues, features, df)

def testData(options, dt):
    print(list(dt.predict(eval(options.data))))

def getFiles(dir, extension):
    return filter(lambda _file: not os.path.isdir(_file) and _file.endswith(extension),
                  map(lambda _file: dir + "\\" + _file, os.listdir(dir)))

def testDataFile(options):
    dataFile = options.testFile
    (featureValues, targetValues, features, df) = loadData(dataFile, options)
    trees = []
    if os.path.isdir(options.treefile):
        for _file in getFiles(options.treefile, ".dt"):
            trees.append((buildTree(options, _file), os.path.basename(_file)))
    else:
        trees.append((buildTree(options, options.treefile), os.path.basename(options.treefile)))
    for tree in trees:
        print("Loading prediction tree %s" % tree[1])
        prediction = list(tree[0].predict(featureValues))
        df[tree[1]] = pd.Series(prediction, index=df.index)
    _file = dataFile + '.prediction.csv'
    print("Saving prediction %s " % _file)
    df.to_csv(_file)



if options.data <> None:
    print("Testing %s" % options.data)
    testData(options)
elif options.testFile <> None:
    print("Create prediction %s" % options.testFile)
    testDataFile(options)
elif options.filename is not None and options.treefile is not None:
    print("Building tree %s" % options.treefile)
    buildTree(options, options.treefile, options.filename)
