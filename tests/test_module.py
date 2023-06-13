#import numpy as np
#import pytest
#import unit_test_example as ut
#
#def test_compute_L2_norm():
#    seed = 1337
#    np.random.seed(seed)
#    
#    # Generate a test matrix
#    mat = np.random.rand(4, 5)
#
#    # Compute L2 norm using your function
#    result = ut.compute_L2_norm(mat) #+ 1.
#
#    # Compute L2 norm directly
#    expected = 2.4389162973771246 
#
#    # Check that the results are equal
#    # Single precision check
#    np.testing.assert_almost_equal(result, expected, decimal=10)
#
##----------------------------------------------------------------------
#if __name__ == "__main__":
#    test_compute_L2_norm()
