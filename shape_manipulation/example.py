import unittest
from numpy import random
from numpy.ma.core import floor, array
import numpy
from numpy.core.shape_base import vstack, hstack
from numpy.core.numeric import newaxis
from numpy.lib.shape_base import column_stack, hsplit


class ShapeManipulationTEST(unittest.TestCase):
    
    def test_changing_the_shape_of_an_array(self):
        a = floor(10*random.random((3,4)))
        a = array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        numpy.testing.assert_array_equal(a.ravel(), array([ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12]))
        a.shape = (6,2)
        numpy.testing.assert_array_equal(a, array([[ 1,  2],
                                                   [ 3,  4],
                                                   [ 5,  6],
                                                   [ 7,  8],
                                                   [ 9, 10],
                                                   [11, 12]]))
        numpy.testing.assert_array_equal(a.transpose(), array([[ 1,  3,  5,  7,  9, 11],
                                                               [ 2,  4,  6,  8, 10, 12]]))
        a = a.reshape(2,6)
        numpy.testing.assert_array_equal(a, array([[ 1,  2,  3,  4,  5, 6],
                                                   [ 7,  8,  9,  10, 11, 12]]))
        a = a.reshape(3,-1)
        numpy.testing.assert_array_equal(a, array([[1,2,3,4],
                                                   [5,6,7,8],
                                                   [9,10,11,12]]))
        
    def test_stacking_arrays(self):
        a = array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        b = array([[13,14,15,16],[17,18,19,20],[21,22,23,24]])
        c = vstack((a,b))
        numpy.testing.assert_array_equal(c, array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24]]))
        d = hstack((a,b))
        numpy.testing.assert_array_equal(d, array([[1,2,3,4,13,14,15,16],
                                                   [5,6,7,8,17,18,19,20],
                                                   [9,10,11,12,21,22,23,24]]))
        
    def test_column_stack_and_vstack(self):
        a=array([4.,2.])
        b=array([2.,8.])
        numpy.testing.assert_array_equal(column_stack((a[:,newaxis],b[:newaxis])), array([[4.,2.],[2.,8.]]))
        numpy.testing.assert_array_equal(vstack((a[:,newaxis],b[:,newaxis])), array([[4.],[2.],[2.],[8.]]))
        
    def test_splitting_one_array_into_smaller_arrays(self):
        a = array([[ 8.,  8.,  3.,  9.,  0.,  4.,  3.,  0.,  0.,  6.,  4.,  4.],
                   [ 0.,  3.,  2.,  9.,  6.,  0.,  4.,  5.,  7.,  5.,  1.,  4.]])
        b,c,d = hsplit(a,3) # split into three
        numpy.testing.assert_array_equal(b, array([[ 8.,  8.,  3.,  9.],
                                                   [ 0.,  3.,  2.,  9.]]))
        numpy.testing.assert_array_equal(c, array([[ 0.,  4.,  3.,  0.],
                                                   [ 6.,  0.,  4.,  5.]]))
        numpy.testing.assert_array_equal(d, array([[ 0.,  6.,  4.,  4.],
                                                   [ 7.,  5.,  1.,  4.]]))
        e,f,g = hsplit(a,(3,4)) # split after 3 and 4 column
        numpy.testing.assert_array_equal(e, array([[ 8.,  8.,  3.],
                                                   [ 0.,  3.,  2.]]))
        numpy.testing.assert_array_equal(f, array([[ 9.],
                                                   [ 9.]]))
        numpy.testing.assert_array_equal(g, array([[ 0.,  4.,  3.,  0.,  0.,  6.,  4.,  4.],
                                                   [ 6.,  0.,  4.,  5.,  7.,  5.,  1.,  4.]]))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests([ShapeManipulationTEST])
    unittest.TextTestRunner(verbosity=2).run(suite)