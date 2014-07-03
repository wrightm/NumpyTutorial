import unittest
from numpy.ma.core import array, argmin
import numpy
from numpy.core.numeric import newaxis

class BroadcastingTest(unittest.TestCase):
    
    def test_multiple_two_arrays(self):
        
        a = array([1.,2.,3.])
        b = array([2.,2.,2.])
        numpy.testing.assert_array_equal(a*b, array([2.,4.,6.]))
    
    def test_addition_two_arrays(self):
        
        a = array([[0,0,0],
                   [10,10,10],
                   [20,20,20]])
        
        b = array([1,2,3])
        c = a + b
        numpy.testing.assert_array_equal(c, array([[1,2,3],
                                                   [11,12,13],
                                                   [21,22,23]]))
        
    def test_addition_two_arrays_newaxis(self):
        a = array([0.0,10.0,20.0,30.0])
        b = array([1.0,2.0,3.0])
        c = a[:,newaxis] + b
        numpy.testing.assert_array_equal(c, array([[1.,2.,3.],
                                                   [11.,12.,13.],
                                                   [21.,22.,23.],
                                                   [31.,32.,33.]]))
        
    def test_vector_quantisation(self):
        observation = array([111.,188.])
        codes = array([[102.,203.],
                       [132.,193.],
                       [45.,155.],
                       [57.,173.]])
        
        diff = codes - observation
        dist = numpy.sqrt(numpy.sum(diff**2,axis=-1))
        self.assertEqual(argmin(dist, axis=0), 0)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests([BroadcastingTest])
    unittest.TextTestRunner(verbosity=2).run(suite)