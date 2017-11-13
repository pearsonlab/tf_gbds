import tensorflow as tf


def compute_sym_blk_tridiag(AA, BB, iia=None, iib=None):
    """
    Symbolically compute block tridiagonal terms of the inverse of a
    *symmetric* block tridiagonal matrix.

    All input & output assumed to be stacked theano tensors.
    Note that the function expects the off-diagonal blocks of the upper
    triangle & returns the lower-triangle (the transpose). Matrix is assumed
    symmetric so this doesn't really matter, but be careful.

    Input:
    AA - (Batch_size x T x n x n) diagonal blocks
    BB - (Batch_size x T-1 x n x n) off-diagonal blocks (upper triangle)
    iia - (T x 1) block index of AA for the diagonal
    iib - (T-1 x 1) block index of BB for the off-diagonal

    Output:
    D  - (Batch_size x T x n x n) diagonal blocks of the inverse
    OD - (Batch_size x T-1 x n x n) off-diagonal blocks of the inverse
         (lower triangle)
    S  - (Batch_size x T-1 x n x n) intermediary matrix computation used in
         inversion algorithm

    From:
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and
    Banded Matrices"
    Note: Could be generalized to non-symmetric matrices, but it is not
    currently implemented.

    (c) Evan Archer, 2015
    """
    BB = -BB

    # Set up some parameters
    if iia is None:
        nT = tf.shape(AA)[1]
    else:
        nT = tf.shape(iia)[0]

    batch_size = tf.shape(AA)[0]
    d = tf.shape(AA)[2]

    # if we don't have special indexing requirements, just use the obvious
    # indices
    if iia is None:
        iia = tf.range(nT)
    if iib is None:
        iib = tf.range(nT - 1)

    III = tf.eye(d, dtype=tf.float32)

    initS = tf.zeros([batch_size, d, d], dtype=tf.float32)

    def compute_S(acc, inputs):
        Sp1 = acc
        idx = inputs[0]
        B_ip1 = BB[:, iib[tf.minimum(idx + 1, nT - 2)]]
        S_nTm1 = tf.matmul(BB[:, iib[-1]], tf.matrix_inverse(AA[:, iia[-1]]))
        S_i = tf.matmul(
          BB[:, iib[idx]],
          tf.matrix_inverse(AA[:, iia[tf.minimum(idx + 1, nT - 2)]] -
            tf.matmul(Sp1, tf.transpose(B_ip1, perm=[0, 2, 1]))))
        Sm = tf.where(tf.equal(tf.tile([idx], [batch_size]), nT - 2),
                      S_nTm1, S_i)

        return Sm

    S = tf.scan(compute_S, [tf.range(nT - 2, -1, -1)], initializer=initS)
    S = tf.transpose(S, perm=[1, 0, 2, 3])

    initD = tf.zeros([batch_size, d, d], dtype=tf.float32)

    def compute_D(acc, inputs):
        Dm1 = acc
        idx = inputs[0]
        D_nT = tf.matmul(tf.matrix_inverse(AA[:, iia[-1]]),
                         (III + tf.matmul(
                          tf.transpose(BB[:, iib[idx-1]], perm=[0, 2, 1]),
                          tf.matmul(Dm1, S[:, 0]))))
        D1 = (tf.matrix_inverse(AA[:, iia[0]] -
              tf.matmul(BB[:, iib[0]],
                        tf.transpose(S[:, -1], perm=[0, 2, 1]))))
        B_ip11 = BB[:, iib[tf.minimum(idx, nT - 2)]]
        S_t = tf.transpose(S[:, tf.maximum(-idx - 1, -nT + 1)],
                           perm=[0, 2, 1])
        B_t = tf.transpose(BB[:, iib[tf.minimum(idx - 1, nT - 2)]],
                           perm=[0, 2, 1])
        D = tf.where(
          tf.equal(tf.tile([idx], [batch_size]), nT - 1), D_nT,
          tf.where(tf.equal(tf.tile([idx], [batch_size]), 0), D1,
                   tf.matmul(tf.matrix_inverse(AA[:, iia[idx]] -
                             tf.matmul(B_ip11, S_t)),
                   III + tf.matmul(B_t, tf.matmul(Dm1, S[:, -idx])))))

        return D

    D = tf.scan(compute_D, [tf.range(0, nT)], initializer=initD)
    D = tf.transpose(D, perm=[1, 0, 2, 3])

    def compute_OD(acc, inputs):
        idx = inputs[0]
        OD = tf.matmul(tf.transpose(S[:, -idx - 1], perm=[0, 2, 1]),
                       D[:, idx])

        return OD

    OD = tf.scan(compute_OD, [tf.range(0, nT-1)],
                 initializer=tf.matmul(tf.transpose(S[:, -1], perm=[0, 2, 1]),
                                       D[:, 0]))

    return [D, OD, S] 


def compute_sym_blk_tridiag_inv_b(S, D, b):
    """
    Symbolically solve Cx = b for x, where C is assumed to be *symmetric* block
    matrix.

    Input:
    D  - (Batch_size x T x n x n) diagonal blocks of the inverse
    S  - (Batch_size x T-1 x n x n) intermediary matrix computation returned by
         the function compute_sym_blk_tridiag

    Output:
    x - (Batch_size x T x n) solution of Cx = b

   From:
    Jain et al, 2006
  "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded
   Matrices"

    (c) Evan Archer, 2015
    """
    batch_size = tf.shape(b)[0]
    nT = tf.shape(b)[1]
    d = tf.shape(b)[2]
    initp = tf.zeros(tf.shape(b)[2:4], dtype=tf.float32)

    def compute_p(acc, inputs):
        pp = acc
        idx = inputs[0]
        pm = tf.where(
          tf.equal(tf.tile([idx], batch_size), nT - 1), b[:, -1],
          b[:, idx] + tf.matmul(S[:, tf.maximum(-idx-1, -nT+1)], pp))

        return pm

    p = tf.scan(compute_p, [tf.range(nT - 1, -1, -1)], initializer=initp)
    p = tf.transpose(p, perm=[1, 0, 2, 3])

    def compute_q(acc, inputs):
        qm = acc
        idx = inputs[0]
        qp = tf.where(
          tf.equal(tf.tile([idx], batch_size), 0),
          tf.matmul(tf.matmul(
            tf.transpose(S[:, -1], perm=[0, 2, 1]), D[0]), b[0]),
          tf.matmul(tf.transpose(S[:, -idx - 1], perm=[0, 2, 1]),
                    qm + tf.matmul(D[:, idx], b[:, idx])))

        return qp

    q = tf.scan(compute_q, [tf.range(nT - 1)], initializer=p[0])
    q = tf.transpose(q, perm=[1, 0, 2, 3])

    def compute_y(acc, inputs):
        idx = inputs[0]
        yi = tf.where(tf.equal(tf.tile([idx], batch_size), 0),
                      tf.matmul(D[:, 0], p[:, -1]),
                      tf.where(tf.equal(tf.tile([idx], batch_size), nT - 1),
                      tf.matmul(D[:, -1], p[:, 0]) + q[:, -1],
                      tf.matmul(D[:, idx], p[:, -idx - 1]) + q[:, idx - 1]))

        return yi

    y = tf.scan(compute_y, [tf.range(nT)],
                initializer=tf.matmul(D[:, 0], p[:, -1]))
    y = tf.transpose(y, perm=[1, 0, 2, 3])

    return y
