import unittest
import numpy as np
from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_calc_Norm(self):
        test_sample=[1,2,3,4,5]
        self.assertEqual(Decimal(calc_Norm(test_sample,p=0,infty=False)),5.0,'L0 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=1,infty=False),15.0,'L1 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=2,infty=False),7.416198487095663,'L2 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=2,infty=True),5.0,'L_max Wrong answer')
        
        test_sample=np.array([1.2])
        self.assertEqual(Decimal(calc_Norm(test_sample,p=0,infty=False)),1.0,'L0 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=1,infty=False),1.2,'L1 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=2,infty=False),1.2,'L2 Wrong answer')
        self.assertEqual(calc_Norm(test_sample,p=2,infty=True),1.2,'L_max Wrong answer')        

        
    def test_calc_Frobenius_Norm(self):
        test_sample=[[1,2,3],[4,5,6]]
        self.assertEqual(Decimal(calc_Frobenius_Norm(test_sample)),9.539392014169456,'Wrong answer')
        
    
    def test_calc_Condition_Number(self):
        test_sample=[[1,2],[2,4.0001]]
        self.assertEqual(Decimal(calc_Condition_Number(test_sample)),250008.00010058272,'Wrong answer')

        test_sample=[[1,2],[3,4]]
        self.assertEqual(Decimal(calc_Condition_Number(test_sample)),14.999999999999998,'Wrong answer')

    def test_calc_svd(self):
        test_sample =[[1,2],[3,4]]
        result = (np.array([[-0.40455358, -0.9145143 ],
                         [-0.9145143 ,  0.40455358]]),
                 np.array([5.4649857 , 0.36596619]),
                 np.array([[-0.57604844, -0.81741556],
                        [ 0.81741556, -0.57604844]]))
        self.assertEqual(calc_svd(test_sample)[0].any()==result[0].any(),True,'Wrong answer U')
        self.assertEqual(calc_svd(test_sample)[1].any()==result[1].any(),True,'Wrong answer D')
        self.assertEqual(calc_svd(test_sample)[2].any()==result[2].any(),True,'Wrong answer VT')   
        
    def test_calc_svd_decompostion(self):
        test_sample =[[1,2,3,4,5,6,7,8],
                       [2,3,4,5,6,7,8,9],
                       [3,4,5,6,7,8,9,10],
                       [4,5,6,7,8,9,10,11],
                       [5,6,7,8,9,10,11,12]]
        result = np.array([[-14.13331129,  -2.06143446],
                       [-16.81265405,  -1.15527645],
                       [-19.49199682,  -0.24911844],
                       [-22.17133959,   0.65703957],
                       [-24.85068235,   1.56319758]])
        
        self.assertEqual(calc_svd_decompostion(test_sample).any()==result.any(),True,'Wrong answer')        
        
    def test_calc_svd_reconsitution(self):
        test_sample =[[1,2,3,4,5,6,7,8],
                       [2,3,4,5,6,7,8,9],
                       [3,4,5,6,7,8,9,10],
                       [4,5,6,7,8,9,10,11],
                       [5,6,7,8,9,10,11,12]]
        result = np.array([[ 2.28811868,  2.98679854,  3.68547839,  4.38415824,  5.0828381 ,
                         5.78151795,  6.4801978 ,  7.17887766],
                           [ 2.72189206,  3.55302515,  4.38415824,  5.21529133,  6.04642442,
                         6.87755751,  7.7086906 ,  8.53982369],
                           [ 3.15566545,  4.11925177,  5.0828381 ,  6.04642442,  7.01001075,
                         7.97359707,  8.9371834 ,  9.90076972],
                           [ 3.58943883,  4.68547839,  5.78151795,  6.87755751,  7.97359707,
                         9.06963663, 10.16567619, 11.26171575],
                           [ 4.02321221,  5.25170501,  6.4801978 ,  7.7086906 ,  8.9371834 ,
                         10.16567619, 11.39416899, 12.62266179]])
        
        self.assertEqual(calc_svd_reconsitution(test_sample,topk=1).any()==result.any(),True,'Wrong answer') 
            
        
if __name__ == '__main__':
    unittest.main()