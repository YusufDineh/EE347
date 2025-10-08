import numpy as np

def test_numpy():
    # Test 1: Create a 1D NumPy array
    array_1d = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(array_1d, np.array([1, 2, 3, 4, 5])), "1D array creation failed"

    # Test 2: Create a 2D NumPy array
    array_2d = np.array([[1, 2], [3, 4]])
    assert np.array_equal(array_2d, np.array([[1, 2], [3, 4]])), "2D array creation failed"

    # Test 3: Perform element-wise addition
    array_a = np.array([1, 2, 3])
    array_b = np.array([4, 5, 6])
    result_addition = array_a + array_b
    assert np.array_equal(result_addition, np.array([5, 7, 9])), "Element-wise addition failed"

    # Test 4: Perform matrix multiplication
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    result_multiply = np.dot(matrix_a, matrix_b)
    assert np.array_equal(result_multiply, np.array([[19, 22], [43, 50]])), "Matrix multiplication failed"

    print("All NumPy tests passed!")

# Run the NumPy tests
test_numpy()