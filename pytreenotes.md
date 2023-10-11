Don't understand yet how to do Ax matrix vector product when A and x are both pytrees. This is needed for LinearSystem

When linearizing, one way could be to store not A = jacfwd(f, x), but A = jax.linearize(f, x). Then Ax is just A(x).

Above is a problem for feedback linearization. There, we need to compute `c.dot(np.linalg.matrix_power(A, reldeg))` and the like.

Could one just ravel all the pytrees in linearize?

Could ine just ravel the outputs of vector_field, and unravel the inputs, thus allowing arbitrary pytrees as in and output, but keep all the other machieneary strictly with ndarrays?



