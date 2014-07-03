import unittest
from numpy.ma.core import arange, array, sin, zeros
import numpy
from numpy.core.function_base import linspace
from numpy.lib.index_tricks import ogrid, ix_
from numpy.core.records import recarray


class FancyIndexingAndIndexTricksTest(unittest.TestCase):


    def test_indexing_with_arrays_of_indices(self):
        a = arange(12)
        i = array([1,1,3,8,5])
        numpy.testing.assert_array_equal(a[i], i)

        j = array([[3,4],
                   [9,7]])
        
        numpy.testing.assert_array_equal(a[j],array([[3,4],
                                                     [9,7]]))
        palette = array([[0,0,0],
                         [255,0,0],
                         [0,255,0],
                         [0,0,255],
                         [255,255,255]])
        image = array([[0,1,2,0],[0,3,4,0]])
        colour_image = palette[image]

        numpy.testing.assert_array_equal(colour_image, array([[[0,0,0],
                                                               [255,0,0],
                                                               [0,255,0],
                                                               [0,0,0]],
                                                             [[0,0,0],
                                                              [0,0,255],
                                                              [255,255,255],
                                                              [0,0,0]]]))
        
        a = arange(12).reshape(3,4)
        i = array([[0,1],
                   [1,2]])
        j = array([[2,1],
                   [3,3]])
        
        numpy.testing.assert_array_equal(a[i,j], array([[2,5],
                                                        [7,11]]))
        numpy.testing.assert_array_equal(a[i,2], array([[2,6],
                                                        [6,10]]))
        
        numpy.testing.assert_array_equal(a[i,:], array([[[0,1,2,3],
                                                         [4,5,6,7]],
                                                        [[4,5,6,7],
                                                         [8,9,10,11]]]))
        
        numpy.testing.assert_array_equal(a[:,j], array([[[2,1],
                                                         [3,3]],
                                                        [[6,5],
                                                         [7,7]],
                                                        [[10,9],
                                                         [11,11]]]))

        time = linspace(20,145,5)
        data = sin(arange(20).reshape(5,4))
        ind = data.argmax(axis=0)
        time_max = time[ind]
        data_max = data[ind, xrange(data.shape[1])]
        numpy.testing.assert_array_equal(data_max, data.max(axis=0))
        
    def test_indexing_with_boolean_arrays(self):
        
        a = arange(12).reshape(3,4)
        b = a > 4
        numpy.testing.assert_array_equal(b, array([[False, False, False, False],
                                                   [False, True, True, True],
                                                   [True, True, True, True]], dtype=bool))
        a[b] = 0
        numpy.testing.assert_array_equal(a, array([[0,1,2,3],
                                                   [4,0,0,0],
                                                   [0,0,0,0]]))
        
        numpy.testing.assert_array_equal(mandelbrot(4, 4, maxit=1), array([[1,1,1,1],
                                                                           [1,1,1,1],
                                                                           [1,1,1,1],
                                                                           [1,1,1,1]]))
        
        a = arange(12).reshape(3,-1)
        b1 = array([False,True,True])
        b2 = array([True,False,True,False])
        
        numpy.testing.assert_array_equal(a[b1,:], array([[4,5,6,7],
                                                        [8,9,10,11]]))
        
        numpy.testing.assert_array_equal(a[b1], array([[4,5,6,7],
                                                       [8,9,10,11]]))
        
        numpy.testing.assert_array_equal(a[:,b2], array([[0,2],
                                                        [4,6],
                                                        [8,10]]))
        
        numpy.testing.assert_array_equal(a[b1,b2], array([4,10]))
        
    def test_ix_function(self):
        
        a = array([1,2,3,4])
        b = array([5,6,7])
        c = array([8,9,10,11,12])
        ax,bx,cx = ix_(a,b,c)
        numpy.testing.assert_array_equal(ax, array([[[1]],
                                                    [[2]],
                                                    [[3]],
                                                    [[4]]])) # 4x1x1
        numpy.testing.assert_array_equal(bx, array([[[5],
                                                     [6],
                                                     [7]]])) # 1x3x1
        
        numpy.testing.assert_array_equal(cx, array([[[8,9,10,11,12]]])) # 1x1x5
        
        self.assertEqual(ax.shape, (4,1,1))
        self.assertEqual(bx.shape, (1,3,1))
        self.assertEqual(cx.shape, (1,1,5))
        
        ax_plus_bx = ax + bx
        numpy.testing.assert_array_equal(ax_plus_bx, array([[[6],
                                                             [7],
                                                             [8]],
                                                            [[7],
                                                             [8],
                                                             [9]],
                                                            [[8],
                                                             [9],
                                                             [10]],
                                                            [[9],
                                                             [10],
                                                             [11]]]))
        ax_plus_bx_plus_cx = ax+bx+cx
        numpy.testing.assert_array_equal(ax_plus_bx_plus_cx, array([[[14,15,16,17,18],
                                                                     [15,16,17,18,19],
                                                                     [16,17,18,19,20]],
                                                                    [[15,16,17,18,19],
                                                                     [16,17,18,19,20],
                                                                     [17,18,19,20,21]],
                                                                    [[16,17,18,19,20],
                                                                     [17,18,19,20,21],
                                                                     [18,19,20,21,22]],
                                                                    [[17,18,19,20,21],
                                                                     [18,19,20,21,22],
                                                                     [19,20,21,22,23]]]))
        
    def test_shape(self):
        
        a = array([1,1,1])
        self.assertEqual(a.shape, (3,))
        a = array([[1],
                   [1]])
        self.assertEqual(a.shape, (2,1))
        a = array([[[1,1],[1,1]],
                   [[1,1],[1,1]]])
        self.assertEqual(a.shape, (2,2,2))
        a = array([[[[1],[1]],
                   [[1],[1]]]])
        self.assertEqual(a.shape, (1,2,2,1))
        
    def test_record_arrays(self):
        
        img = array([ [[0,1], [0,0]], [[0,0], [1,0]], [[0,0], [0,1]] ], dtype='float32')
        self.assertEqual(img.shape, (3,2,2))
        
        img = array([[(0,0,0), (1,0,0)], [(0,1,0), (0,0,1)]], [('r','float32'),('g','float32'),('b','float32')])
        img = array([[(0,0,0), (1,0,0)], [(0,1,0), (0,0,1)]], {'names': ('r','g','b'), 'formats': ('f4', 'f4', 'f4')})
        img = zeros((2,2), [('r','float32'),('g','float32'),('b','float32')])
        img.flat = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
        print img.view(recarray)
        
def mandelbrot(h, w, maxit=20):
    
        y,x = ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
        c = x+y*1j
        z = c

        divtime = maxit + zeros(z.shape, dtype=int)
                
        for i in xrange(maxit):
            z = z**2 + c
            diverge = z*numpy.conj(2) > 2**2
            div_now = diverge & (divtime==maxit)
            divtime[div_now] = i
            z[diverge] = 2
            
        return divtime
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test']
    suite = unittest.TestSuite()
    suite.addTests([FancyIndexingAndIndexTricksTest])
    unittest.TextTestRunner(verbosity=2).run(suite)