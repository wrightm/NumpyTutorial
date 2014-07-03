import unittest
from numpy.core.numeric import array, arange
import numpy
from numpy.linalg.linalg import inv, solve, eig
from numpy.lib.twodim_base import eye
from numpy.ma.extras import dot
from numpy.ma.core import trace

class LinearAlgebraTest(unittest.TestCase):

    def test_simple_array_operations(self):
        a = array([[1.,2.],
                   [3.,4.]])
        numpy.testing.assert_array_equal(a.transpose(), array([[1.,3.],
                                                               [2.,4.]]))
        
        numpy.testing.assert_array_almost_equal(trace(a), 5)
        
        inv_a = inv(a)
        b = array([[-2.,1.],
                   [1.5,-.5]])
        
        self.assertTrue(numpy.allclose(inv_a,b))
        
        i = dot(a,inv_a)
        numpy.testing.assert_array_almost_equal(i, eye(2), 1)
        
        numpy.testing.assert_array_almost_equal(inv_a, b)
        # system of linear equations
        a = array([[3,2,-1],
                   [2,-2,4],
                   [-1,0.5,-1]])
        b = array([1,-2,0])
        
        c = solve(a,b)
        
        d = dot(a,c)
        numpy.testing.assert_array_almost_equal(b, d, 1)
        
        a = array([[.8,.3],
                   [.2,.7]])
        eigen_values, eigen_vectors = eig(a)
        lambda_1 = eigen_values[0]
        x_1 = eigen_vectors[:,0]
        
        lambda_2 = eigen_values[1]
        x_2 = eigen_vectors[:,1]
        
    def test_the_matrix_class(self):
        a = numpy.matrix('1. 2.; 3. 4.')
        numpy.testing.assert_array_equal(a.transpose(), numpy.matrix('1. 3.; 2. 4.'))
        numpy.testing.assert_array_almost_equal(a.I, numpy.matrix('-2. 1.; 1.5 -0.5'))
        b = numpy.matrix('5. 7.')
        numpy.testing.assert_array_equal(b.transpose(), numpy.matrix('5.;7.'))
        numpy.testing.assert_array_equal(solve(a,b.transpose()), numpy.matrix('-3.;4.'))
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests([LinearAlgebraTest])
    unittest.TextTestRunner(verbosity=2).run(suite)