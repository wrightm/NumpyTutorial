import unittest
from numpy.core.numeric import arange, array, zeros, ones, empty
import numpy
from numpy.core.function_base import linspace
from numpy.ma.core import sin, exp, fromfunction
from scipy.constants.constants import pi
from numpy.ma.extras import dot


class AnExampleTest(unittest.TestCase):


    def test_an_example_a(self):
        a = arange(15).reshape(3,5)
        self.assertEqual(a.shape, (3,5))
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.dtype.name, 'int64')
        self.assertEqual(a.itemsize, 8) # 64/8
        self.assertEqual(a.size, 15)
        self.assertEqual(type(a), numpy.ndarray)
        
    def test_an_example_b(self):
        b = array([6,7,8])
        self.assertEqual(type(b), numpy.ndarray)
        
class ArrayCreationTest(unittest.TestCase):

    def test_int64_ndarray(self):
        a = array([1,2,3,4])
        self.assertEqual(a.dtype, 'int64')
        
    def test_float64_ndarray(self):
        b = array([1.2,3.5,5.1])
        self.assertEqual(b.dtype, 'float64')
        
    def test_2x3_ndarray(self):
        b = array([(1.5,2,3),(4,5,6)])
        self.assertEqual(b.shape, (2,3))
        
    def test_complex_array(self):
        c = array([(1,2),(3,4)], dtype=complex)
        self.assertEqual(c.dtype, complex)
        
    def test_zeros_method(self):
        a = zeros((3,4))
        self.assertEqual(a.shape, (3,4))
        
    def test_ones_method(self):
        a = ones((2,3,4), dtype='int16')
        self.assertEqual(a.dtype, 'int16')
        
    def test_empty_method(self):
        a = empty((2,3))
        self.assertEqual(a.ndim, 2)
        
    def test_arange_method(self):
        a = arange(10,30,5)
        numpy.testing.assert_array_equal(a, array([10, 15, 20, 25]))
        
    def test_linespace_method(self):
        a = linspace(0,2,9) # array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
        x = linspace(0, 2*pi, 100)
        f = sin(x)
        self.assertEqual(len(a), 9)
        
class BasicOperationsTest(unittest.TestCase):
    
    def test_elementwise_subtraction(self):
        a = array([20,30,40,50])
        b = arange(4)
        c = a-b 
        numpy.testing.assert_array_equal(c, array([20, 29, 38, 47]))
        
    def test_elementwise_power(self):
        b = arange(4)**2
        numpy.testing.assert_array_equal(b, array([0, 1, 4, 9]))
        
    def test_elementwise_equality_less_than(self):
        a = array([20,30,40,50])
        b = a < 35
        numpy.testing.assert_array_equal(b, array([True, True, False, False], dtype=bool))
        
    def test_elementwise_product(self):
        A = array( [[1,1],
                    [0,1]] )
        B = array( [[2,0],
                    [3,4]] )
        C = A*B                         # elementwise product
        
        numpy.testing.assert_array_equal(C, array([[2, 0],
                                                   [0, 4]]))
        
    def test_matrix_product(self):
        A = array( [[1,1],
                    [0,1]] )
        B = array( [[2,0],
                    [3,4]] )
        
        C = dot(A,B)
        numpy.testing.assert_array_equal(C,array([[5, 4],
                                                  [3, 4]]))
        
    def test_elementwise_plus_operator(self):
        a = ones((2,3), dtype=int)
        a += 3
        numpy.testing.assert_array_equal(a, array([[4,4,4],
                                                   [4,4,4]]))
    
    def test_elementwise_multiply_operator(self):
        a = ones((2,3), dtype=int)
        a *= 3
        numpy.testing.assert_array_equal(a, array([[3,3,3],
                                                   [3,3,3]]))
        
    def test_upcasting_from_int_float(self):
        a = ones(3,dtype='int32')
        b = linspace(0, pi, 3)
        c = a+b
        self.assertEqual(type(c), type(b))
        
    def test_upcasting_from_float_complex(self):
        b = linspace(0, pi, 3)
        c = exp(b*1j)
        self.assertEqual(c.dtype.name, 'complex128')
        
    def test_array_unit_methods(self):
        a = array([[1,2,3], [1,2,3]])
        self.assertEqual(a.min(), 1)
        self.assertEqual(a.max(), 3)
        self.assertEqual(a.sum(), 12)
        
    def test_array_axis_sum(self):
        a = array([[1,2,3], [1,2,3]])
        numpy.testing.assert_array_equal(a.sum(axis=0), array([2,4,6])) # sum columns
        numpy.testing.assert_array_equal(a.sum(axis=1), array([6,6])) # sum row
        
    def test_array_axis_min(self):
        a = array([[1,2,3], [1,2,3]])
        numpy.testing.assert_array_equal(a.min(axis=0), array([1,2,3])) # min columns
        numpy.testing.assert_array_equal(a.min(axis=1), array([1,1])) # min row
        
    def test_array_axis_max(self):
        a = array([[1,2,3], [1,2,3]])
        numpy.testing.assert_array_equal(a.max(axis=0), array([1,2,3])) # max columns
        numpy.testing.assert_array_equal(a.max(axis=1), array([3,3])) # max row
    
    def test_array_cumsum(self):
        a = array([[1,2,3], [1,2,3]])
        numpy.testing.assert_array_equal(a.cumsum(), array([1,3,6,7,9,12]))
        numpy.testing.assert_array_equal(a.cumsum(axis=1), array([[1,3,6], [1,3,6]]))
        numpy.testing.assert_array_equal(a.cumsum(axis=0), array([[1,2,3],[2,4,6]]))
        
class IndexingSlicingIteratingTest(unittest.TestCase):
    
    def test_one_dim_indexing(self):
        a = arange(10)**3
        numpy.testing.assert_array_equal(a, array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729]))
        self.assertEqual(a[2], 8)
    
    def test_one_dim_slicing(self):
        a = arange(10)**3
        numpy.testing.assert_array_equal(a[2:5],array([ 8, 27, 64]))
        numpy.testing.assert_array_equal(a[:6:2],array([ 0, 8, 64]))
        a[:6:2] = -1000 
        numpy.testing.assert_array_equal(a[::-1],array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000]))
    
    def test_multi_dim_indexing(self):
        b = fromfunction(f,(5,4),dtype=int)
        self.assertEqual(b[2,3], 50)
        
    def test_multi_slicing_indexing(self):
        b = fromfunction(f,(5,4),dtype=int)
        numpy.testing.assert_array_equal(b[0:5, 1],array([ 10, 20, 30, 40, 50]))
        numpy.testing.assert_array_equal(b[:,1],array([ 10, 20, 30, 40, 50]))
        numpy.testing.assert_array_equal(b[1:3,:], array([[10, 20, 30, 40],
                                                          [20, 30, 40, 50]]))
        
        c = array( [ [[  0,  1,  2],
                      [ 10, 12, 13]],
                    
                    [[100,101,102],
                     [110,112,113]] ] )
        
        numpy.testing.assert_array_equal(c[0,...], array([[  0,  1,  2],
                                                          [ 10, 12, 13]]))
        numpy.testing.assert_array_equal(c[0,1,:], array([ 10, 12, 13]))
        numpy.testing.assert_array_equal(c[0,1,2], 13)
        
    def test_multi_itr_array(self):
        
        c = array( [ [[  0,  1,  2],
                      [ 10, 12, 13]],
                    
                    [[100,101,102],
                     [110,112,113]] ] )
        
        #for row in c:
        #    print row
            
        #for element in c.flat:
        #    print element
        
def f(x,y):
    return 10*(x+y)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    suite = unittest.TestSuite()
    suite.addTests([AnExampleTest,ArrayCreationTest,BasicOperationsTest,IndexingSlicingIteratingTest])
    unittest.TextTestRunner(verbosity=2).run(suite)