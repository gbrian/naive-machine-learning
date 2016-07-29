import pickle
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
from optparse import OptionParser

# Parse args
def getArgs():
    parser = OptionParser()
    parser.add_option("-t", "--treefile", dest="treefile",
        help="decision tree file", metavar="FILE")
    parser.add_option("-o", "--outfile", dest="outfile",
        help="Output file", metavar="FILE")

    return parser.parse_args()

(options, args) = getArgs()


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
def loadTree(treefile):
    dt = None
    if os.path.exists(treefile):
        dt = pickle.load(open(treefile, "rb"))
    return dt

tree = loadTree(options.treefile)
visualize_tree(tree, )