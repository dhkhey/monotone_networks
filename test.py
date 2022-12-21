import unittest
from monotone_networks import monotone_network
import numpy as np
from numpy.random import default_rng
import torch
from itertools import product

class Test_Monotone(unittest.TestCase):

	#TEST ONE: 
	# X = standard unit vectors of dim 1~10. 
	# Y = random numbers 
	def test_1_std_unit_vectors(self):
		#testing dimensions 1~10
		for i in range(1,11):
			X,Y = self.generate_std_unit_vector_dataset(dim=i)
			model = monotone_network(X,Y)
			#for each datapoint
			for i in range(len(X)):
				y_pred = model.forward(X[i])
				self.assertEqual(y_pred,Y[i])

	# TEST 2: vector of 1's of dim 1~10
	# e.g. dim 3
	# X = [[1,0,0], [1,1,0], [1,1,1]]
	# Y = [y1,y2,y3] where y1 < y2 < y3

	def test_2_vectors_of_ones(self):
		#testing dimensions 1~10
		for i in range(1,11):
			X,Y = self.generate_vector_of_ones_dataset(dim=i)
			model = monotone_network(X,Y)
			#for each datapoint
			for i in range(len(X)):
				y_pred = model.forward(X[i])
				self.assertEqual(y_pred,Y[i])

	# TEST 3: vector of c's of dim 1~10
	# e.g. dim 3, c = 4
	# X = [[4,0,0], [4,4,0], [4,4,4]]
	# Y = [y1,y2,y3] where y1 < y2 < y3

	def test_3_scaled_vectors(self):
		for i in range(1,11):
			X,Y = self.generate_vector_of_ones_dataset(dim=i)
			for c in range(2,10):
				X_c = c*X
				model = monotone_network(X_c,Y)
				#for each datapoint
				for i in range(len(X_c)):
					y_pred = model.forward(X_c[i])
					self.assertEqual(y_pred,Y[i])

	# TEST 4: boolean vectors of dim 1~10
	# X = dim^2 combinations of 0's and 1's. 
	# Y = 1 if all coordinates are 1. 0 otherwise. 
	def test_4_boolean_AND(self):
		for i in range(1,11):
			X = self.generate_boolean_vectors(dim=i)
			Y = [1 if all(x) else 0 for x in X]
			Y = torch.Tensor(Y)
			model = monotone_network(X,Y)
			#for each datapoint
			for i in range(len(X)):
				y_pred = model.forward(X[i])
				self.assertEqual(y_pred,Y[i])

	# TEST 5: boolean vectors of dim 1~10
	# X = dim^2 combinations of 0's and 1's. 
	# Y = 1 if at least one coordinate is 1. 0 otherwise. 
	def test_5_boolean_OR(self):
		for i in range(1,11):
			X = self.generate_boolean_vectors(dim=i)
			Y = [1 if any(x) else 0 for x in X]
			Y = torch.Tensor(Y)
			model = monotone_network(X,Y)
			#for each datapoint
			for i in range(len(X)):
				y_pred = model.forward(X[i])
				self.assertEqual(y_pred,Y[i])

	# TEST 6: boolean vectors of dim 1~10
	# X = dim^2 combinations of 0's and 1's. 
	# Y = 1 if more than half of coordinates are 1. 0 otherwise. 
	def test_6_boolean_MAJORITY(self):
		for i in range(1,11):
			X = self.generate_boolean_vectors(dim=i)
			Y = [1 if sum(x) > (len(x)/2) else 0 for x in X]
			Y = torch.Tensor(Y)
			model = monotone_network(X,Y)
			#for each datapoint
			for i in range(len(X)):
				y_pred = model.forward(X[i])
				self.assertEqual(y_pred,Y[i])


	##### DATASET GENERATION CODES #####

	# e.g dim 3 
	# X = [[1,0,0],[0,1,0],[0,0,1]]
	# Y is 3 random numbers (not ordered). 
	def generate_std_unit_vector_dataset(self,dim=-1):
		X = np.eye(dim,dim,0)
		rng = default_rng()
		Y = rng.choice(100,size=dim)
		return torch.Tensor(X),torch.Tensor(Y)


	#e.g. dim 3
	#X = [[1,0,0],[1,1,0],[1,1,1]]
	#Y = [y1,y2,y3] where y1 < y2 < y3
	def generate_vector_of_ones_dataset(self,dim=-1):
		w = np.ones((dim,dim))
		X = np.tril(w)
		rng = default_rng()
		Y = rng.choice(100,size=dim, replace=False)
		Y.sort()
		return torch.Tensor(X),torch.Tensor(Y)

	# returns list of all combinations of 0 and 1 with specified dimension.
	# e.g. dim 2
	# [[0,0],[0,1],[1,0],[1,1]]
	def generate_boolean_vectors(self,dim=-1):
		X = product([0, 1], repeat=dim)
		X = [list(tup) for tup in X]
		return torch.Tensor(X)




if __name__ == '__main__':
	unittest.main()