"""
Tensorflow implementation of functions to perform the Cholesky factorization
of a block tridiagonal matrix. Ported from Evan Archer's implementation here:
https://github.com/earcher/vilds/blob/master/code/lib/blk_tridiag_chol_tools.py
"""

import tensorflow as tf


def blk_tridiag_chol(A, B):
    """
    Compute the cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.
    Inputs:
    A - [Batch_size x T x n x n] tensor,
        where each A[i,:,:] is the ith block diagonal matrix
    B - [Batch_size x T-1 x n x n] tensor, where each B[i,:,:] is the ith
        (upper) 1st block off-diagonal matrix
    Outputs:
    R - python list with two elements
        * R[0] - [Batch_size x T x n x n] tensor of block diagonal elements
        of Cholesky decomposition
        * R[1] - [Batch_size x T-1 x n x n] tensor of (lower) 1st block
        off-diagonal elements of Cholesky
    """
    def _step(acc, inputs):
        """
        Compute the Cholesky decomposition of a symmetric block tridiagonal
        matrix.
        acc is the output of the previous loop
        inputs is a tuple of inputs
        """
        LL, _ = acc
        AA, BB = inputs

        CC = tf.transpose(tf.matrix_solve(LL, BB), perm=[0, 2, 1])
        DD = AA - tf.matmul(CC, tf.transpose(CC, perm=[0, 2, 1]))
        LL = tf.cholesky(DD)

        return [LL, CC]

    L = tf.cholesky(A[:, 0])
    C = tf.zeros_like(B[:, 0])

    R = tf.scan(_step,
                [tf.transpose(A[:, 1:], perm=[1, 0, 2, 3]),
                 tf.transpose(B, perm=[1, 0, 2, 3])],
                initializer=[L, C])
    R[0] = tf.concat([tf.expand_dims(L, 1),
                      tf.transpose(R[0], perm=[1, 0, 2, 3])], 1)
    R[1] = tf.transpose(R[1], perm=[1, 0, 2, 3])

    return R


def blk_chol_inv(A, B, b, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.
    Inputs:
    A - [Batch_size x T x n x n] tensor, where each A[i,:,:] is the ith block
        diagonal matrix
    B - [Batch_size x T-1 x n x n] tensor, where each B[i,:,:] is the ith
        (upper or lower) 1st block off-diagonal matrix
    b - [Batchs_size x T x n] tensor

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)
    Outputs:
    x - solution of Cx = b
    """
    def _step(acc, inputs):
        x = acc
        A, B, b = inputs

        return tf.matrix_solve(A, b - tf.matmul(B, x))

    if transpose:
        A = tf.transpose(A, perm=[0, 1, 3, 2])
        B = tf.transpose(B, perm=[0, 1, 3, 2])
    if lower:
        x0 = tf.matrix_solve(A[:, 0], b[:, 0])
        X = tf.scan(_step, [tf.transpose(A[:, 1:], perm=[1, 0, 2, 3]),
                            tf.transpose(B, perm=[1, 0, 2, 3]),
                            tf.transpose(b[:, 1:], perm=[1, 0, 2, 3])],
                    initializer=x0)
        X = tf.transpose(X, perm=[1, 0, 2, 3])
        X = tf.concat([tf.expand_dims(x0, 1), X], 1)
    else:
        xN = tf.matrix_solve(A[:, -1], b[:, -1])
        X = tf.scan(_step, [tf.transpose(A[:, :-1, ::-1], perm=[1, 0, 2, 3]),
                            tf.transpose(B[:, ::-1], perm=[1, 0, 2, 3]),
                            tf.transpose(b[:, :-1, ::-1], perm=[1, 0, 2, 3])],
                    initializer=xN)
        X = tf.transpose(X, perm=[1, 0, 2, 3])
        X = tf.concat([tf.expand_dims(xN, 1), X], 1)[:, ::-1]

    return X


def blk_chol_mtimes(A, B, x, lower=True, transpose=False):
    """
    Evaluate Cx = b, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.
    Inputs:
    A - [Batch_size x T x n x n] tensor, where each A[i,:,:] is the ith block
        diagonal matrix
    B - [Batch_size x T-1 x n x n] tensor, where each B[i,:,:] is the ith
        (upper or lower) 1st block off-diagonal matrix  

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)
    Outputs:
    b - result of Cx = b
    """
    def _lower_step(acc, input):
        A, B, x1, x2 = input

        return tf.matmul(A, x1) + tf.matmul(B, x2)

    if transpose:
        A = tf.transpose(A, perm=[0, 1, 3, 2])
        B = tf.transpose(B, perm=[0, 1, 3, 2])
    if lower:
        b0 = tf.matmul(A[:, 0], x[:, 0])
        X = tf.scan(_lower_step, [tf.transpose(A[:, 1:], perm=[1, 0, 2, 3]),
                                  tf.transpose(B, perm=[1, 0, 2, 3]),
                                  tf.transpose(x[:, 1:], perm=[1, 0, 2, 3]),
                                  tf.transpose(x[:, 0:-1], perm=[1, 0, 2, 3])],
                    initializer=b0)
        X = tf.concat([tf.expand_dims(b0, 1), X], 1)
    else:
        bN = tf.matmul(A[:, -1], x[:, -1])
        X = tf.scan(_lower_step, [tf.transpose(A[:, 0:-1], perm=[1, 0, 2, 3]),
                                  tf.transpose(B, perm=[1, 0, 2, 3]),
                                  tf.transpose(x[:, 0:-1], perm=[1, 0, 2, 3]),
                                  tf.transpose(x[:, 1:], perm=[1, 0, 2, 3])],
                    initializer=bN)
        X = tf.concat([X, tf.expand_dims(bN, 1)], 1)

    return X
