import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# viz tree
from sklearn.externals.six import StringIO  
import pydotplus




iris = load_iris()

def dump():
	print iris.feature_names
	print iris.target_names

	print iris.data[0]
	print iris.target[0]

	for i in range(len(iris.target)):
		print "Example %d: label %s, features %s" % (i, iris.target[1], iris.data[i])

#dump()
def test():
	#training data
	test_idx = [0, 50, 100]
	train_target = np.delete(iris.target, test_idx)
	train_data = np.delete(iris.data, test_idx, axis=0)
	
	# testing data
	test_target = iris.target[test_idx]
	test_data = iris.data[test_idx]
	
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(train_data, train_target)
	
	print test_target
	print clf.predict(test_data)

	dot_data = StringIO() 
	tree.export_graphviz(clf, out_file=dot_data,  
		feature_names=iris.feature_names,  
		class_names=iris.target_names,  
		filled=True, rounded=True,  
		special_characters=True)  
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
	graph.write_pdf("iris.pdf")
 	
test()
	