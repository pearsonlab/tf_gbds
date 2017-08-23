import numpy as np
import tensorflow as tf
import numpy.testing as npt

import tf_gbds.lib.blk_tridiag_chol_tools as blk

# shared testing data
prec = np.float32
npA = np.array([[1, .9], [.9, 4]]).astype(prec)
npB = (.01 * np.array([[2, 7], [7, 4]])).astype(prec)
npC = (1. * np.array([[3, 0.], [0, 1]])).astype(prec)
npD = (.01 * np.array([[7, 2], [9, 3]])).astype(prec)
npE = (.01 * np.array([[2, 0], [4, 3]])).astype(prec)
npF = (.01 * np.array([[1, 0], [2, 7]])).astype(prec)
npG = (.01 * np.array([[3, 0], [8, 1]])).astype(prec)

npZ = np.array([[0, 0], [0, 0]]).astype(prec)

npb = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).astype(prec)

lowermat = np.bmat([[npF,   npZ,   npZ,   npZ],
                    [npB.T, npC,   npZ,   npZ],
                    [npZ,   npD.T, npE,   npZ],
                    [npZ,   npZ,   npB.T, npG]])
cholmat = lowermat.dot(lowermat.T)


def test_blk_tridiag_chol():
    alist = [cholmat[i:(i+2), i:(i+2)] for i in range(0, cholmat.shape[0], 2)]
    blist = [cholmat[(i+2):(i+4), i:(i+2)].T
             for i in range(0, cholmat.shape[0] - 2, 2)]

    theDiag = tf.constant(np.array(alist))
    theOffDiag = tf.constant(np.array(blist))
    R = blk.blk_tridiag_chol(theDiag, theOffDiag)

    with tf.Session() as sess:
        R0, R1 = R[0].eval(), R[1].eval()

    for (x, y) in zip(R0, [npF, npC, npE, npG]):
        npt.assert_allclose(x, y, atol=1e-4, rtol=1e-5)

    for (x, y) in zip(R1, [npB.T, npD.T, npB.T]):
        npt.assert_allclose(x, y, atol=1e-4, rtol=1e-5)


def test_blk_chol_inv():
    xl = np.linalg.solve(lowermat, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    x = np.linalg.solve(cholmat, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    alist = [npF, npC, npE, npG]
    blist = [npB.T, npD.T, npB.T]
    theDiag = tf.constant(np.array(alist))
    theOffDiag = tf.constant(np.array(blist))
    b = tf.expand_dims(tf.constant(npb), -1)

    # now solve C * x = b by inverting one Cholesky factor of C at a time
    ib = blk.blk_chol_inv(theDiag, theOffDiag, b)
    tfx = blk.blk_chol_inv(theDiag, theOffDiag, ib, lower=False,
                           transpose=True)

    with tf.Session() as sess:
        ib_val = ib.eval()
        tfx_val = tfx.eval()
    npt.assert_allclose(ib_val.flatten(), xl, atol=1e-5, rtol=1e-4)
    npt.assert_allclose(tfx_val.flatten(), x, atol=1e-5, rtol=3e-3)


def test_blk_chol_mtimes():
    xx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    bl = lowermat.T.dot(xx)
    b = cholmat.dot(xx)

    alist = [npF, npC, npE, npG]
    blist = [npB.T, npD.T, npB.T]
    theDiag = tf.constant(np.array(alist))
    theOffDiag = tf.constant(np.array(blist))
    x = tf.expand_dims(tf.constant(npb), -1)

    ix = blk.blk_chol_mtimes(theDiag, theOffDiag, x, lower=False,
                             transpose=True)

    tfb = blk.blk_chol_mtimes(theDiag, theOffDiag, ix)
    with tf.Session() as sess:
        ix_val = ix.eval()
        tfb_val = tfb.eval()

    npt.assert_allclose(ix_val.flatten(), np.array(bl).flatten(),
                        atol=1e-5, rtol=1e-4)
    npt.assert_allclose(tfb_val.flatten(), np.array(b).flatten(),
                        atol=1e-5, rtol=1e-4)
