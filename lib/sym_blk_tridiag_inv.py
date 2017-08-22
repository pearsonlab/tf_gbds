import tensorflow as tf


def compute_sym_blk_tridiag(AA, BB, iia=None, iib=None):
    '''
    Symbolically compute block tridiagonal terms of the inverse of a
    *symmetric* block tridiagonal matrix.

    All input & output assumed to be stacked theano tensors.
    Note that the function expects the off-diagonal blocks of the upper
    triangle & returns the lower-triangle (the transpose). Matrix is assumed
    symmetric so this doesn't really matter, but be careful.

    Input:
    AA - (T x n x n) diagonal blocks
    BB - (T-1 x n x n) off-diagonal blocks (upper triangle)
    iia - (T x 1) block index of AA for the diagonal
    iib - (T-1 x 1) block index of BB for the off-diagonal

    Output:
    D  - (T x n x n) diagonal blocks of the inverse
    OD - (T-1 x n x n) off-diagonal blocks of the inverse (lower triangle)
    S  - (T-1 x n x n) intermediary matrix computation used in inversion
         algorithm

    From:
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and
    Banded Matrices"
    Note: Could be generalized to non-symmetric matrices, but it is not
    currently implemented.

    (c) Evan Archer, 2015
    '''
    BB = -BB

    # Set up some parameters
    if iia is None:
        nT = tf.shape(AA)[0]
    else:
        nT = tf.shape(iia)[0]

    d = tf.shape(AA)[1]

    # if we don't have special indexing requirements, just use the obvious
    # indices
    if iia is None:
        iia = tf.range(nT)
    if iib is None:
        iib = tf.range(nT-1)

    III = tf.eye(d, dtype=tf.float32)

    initS = tf.zeros([d, d], dtype=tf.float32)

    def compute_S(acc, inputs):
        Sp1 = acc
        idx = inputs[0]
        B_ip1 = BB[iib[tf.minimum(idx+1, nT-2)]]
        S_nTm1 = tf.matmul(BB[iib[-1]], tf.matrix_inverse(AA[iia[-1]]))
        S_i = tf.matmul(BB[iib[idx]],
                        tf.matrix_inverse(AA[iia[tf.minimum(idx+1, nT-2)]]
                                          - tf.matmul(Sp1,
                                                      tf.transpose(B_ip1))))
        Sm = tf.where(tf.equal(idx, nT-2), S_nTm1, S_i)
        return Sm

    S = tf.scan(compute_S, [tf.range(nT-2, -1, -1)], initializer=initS)

    initD = tf.zeros([d, d], dtype=tf.float32)

    def compute_D(acc, inputs):
        Dm1 = acc
        idx = inputs[0]
        D_nT = tf.matmul(tf.matrix_inverse(AA[iia[-1]]),
                         III + tf.matmul(tf.transpose(BB[iib[idx-1]]),
                         tf.matmul(Dm1, S[0])))
        D1 = tf.matrix_inverse(AA[iia[0]]
                               - tf.matmul(BB[iib[0]], tf.transpose(S[-1])))
        B_ip11 = BB[iib[tf.minimum(idx, nT-2)]]
        S_t = tf.transpose(S[tf.maximum(-idx-1, -nT+1)])
        B_t = tf.transpose(BB[iib[tf.minimum(idx-1, nT-2)]])
        D = tf.where(tf.equal(idx, nT-1), D_nT,
                     tf.where(tf.equal(idx, 0), D1,
                              tf.matmul(tf.matrix_inverse(AA[iia[idx]]
                                        - tf.matmul(B_ip11, S_t)),
                              III + tf.matmul(B_t, tf.matmul(Dm1, S[-idx])))
                              )
                     )
        return D

    D = tf.scan(compute_D, [tf.range(0, nT)], initializer=initD)

    def compute_OD(acc, inputs):
        idx = inputs[0]
        OD = tf.matmul(tf.transpose(S[-idx-1]), D[idx])
        return OD

    OD = tf.scan(compute_OD, [tf.range(0, nT-1)],
                 initializer=tf.matmul(tf.transpose(S[-1]), D[0]))

    return [D, OD, S]  # , updates_D+updates_OD+updates_S]


def compute_sym_blk_tridiag_inv_b(S, D, b):
    '''
    Symbolically solve Cx = b for x, where C is assumed to be *symmetric* block
    matrix.

    Input:
    D  - (T x n x n) diagonal blocks of the inverse
    S  - (T-1 x n x n) intermediary matrix computation returned by
         the function compute_sym_blk_tridiag

    Output:
    x - (T x n) solution of Cx = b

   From:
    Jain et al, 2006
  "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded
   Matrices"

    (c) Evan Archer, 2015
    '''
    print(b)
    nT = tf.shape(b)[0]
    d = tf.shape(b)[1]
    print(d)
    initp = tf.zeros(tf.shape(b)[1:3], dtype=tf.float32)

    def compute_p(acc, inputs):
        pp = acc

        idx = inputs[0]

        pm = tf.where(tf.equal(idx, nT-1),
                      b[-1],
                      b[idx] + tf.matmul(S[tf.maximum(-idx-1, -nT+1)], pp)
                      )
        return pm

    p = tf.scan(compute_p, [tf.range(nT-1, -1, -1)], initializer=initp)

    def compute_q(acc, inputs):
        qm = acc
        idx = inputs[0]
        qp = tf.where(tf.equal(idx, 0),
                      tf.matmul(tf.matmul(tf.transpose(S[-1]), D[0]), b[0]),
                      tf.matmul(tf.transpose(S[-idx-1]), qm
                      + tf.matmul(D[idx], b[idx]))
                      )
        return qp

    q = tf.scan(compute_q, [tf.range(nT-1)], initializer=p[0])

    def compute_y(acc, inputs):
        idx = inputs[0]
        yi = tf.where(tf.equal(idx, 0),
                      tf.matmul(D[0], p[-1]),
                      tf.where(tf.equal(idx, nT-1),
                      tf.matmul(D[-1], p[0]) + q[-1],
                      tf.matmul(D[idx], p[-idx-1]) + q[idx-1]
                               )
                      )
        return yi

    y = tf.scan(compute_y, [tf.range(nT)], initializer=tf.matmul(D[0], p[-1]))
    # return [y, updates_q+updates+y]
    return y
