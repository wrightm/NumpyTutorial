import unittest
from numpy.ma.core import array
import numpy


class CopiesAndViewsTest(unittest.TestCase):


    def test_view_shallow_copy(self):
        a = array([[1,2,3,4],[5,6,7,8]])
        b = a.view() # b is a view of data owed by a
        numpy.testing.assert_array_equal(b,a)
        numpy.testing.assert_array_equal(b.base,a)
        self.assertEqual(b.flags.owndata,False)
        
        b[0,3] = 10 
        self.assertEqual(a[0,3], 10)
        
    def test_slicing_shallow_copy(self):
        a = array([[1,2,3,4],[5,6,7,8]])
        b = a[:,3]
        b[:] = 10
        numpy.testing.assert_array_equal(a,array([[1,2,3,10],[5,6,7,10]]))
        
    def test_deep_copy(self):
        a = array([[1,2,3,4],[5,6,7,8]])
        b = a.copy()
        b[0,0] = 10
        numpy.testing.assert_array_equal(a,array([[1,2,3,4],[5,6,7,8]]))
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests([CopiesAndViewsTest])
    unittest.TextTestRunner(verbosity=2).run(suite)