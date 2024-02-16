# Charles Gil, 260970950

1. Similarly to most of the interpolations in the rest of the assignment, this is a linear interpolation. However, it
does not depend on t, rather it is a weighted average between the current and target matrices, thus creating a smooth
and constant transition between perspectives. I would call this weighted average or mixing linear interpolation.

2. No

3. I observed that when moving from the initial view to the side view, the viewer could become very large and
suddenly shrink back to the correct size. This interpolation does not correctly handle objects near the boundary of the
screen, since projection matrices are not linear transformations when it comes to perspective projections.

4. I do not see the slow-fast-slow effect for 180 degree rotations. When the quaternions are nearly opposite, the linear
interpolation can result in a situation where the interpolated quaternion has a norm close to zero. Normalizing such a
quaternion involves dividing each component by a small number, leading to a sudden increase in magnitude, resulting in
the slow-fast-slow behavior.

On the other hand, for 180-degree rotations the linear interpolation doesn't result in quaternions with a norm close to
zero. Therefore, the normalization step doesn't cause a sudden increase in magnitude, and the interpolation appears more
consistent.

5. If a non-unit-length quaternion is provided to this code, the resulting rotation matrix may not be a valid rotation
matrix. In other words, if the quaternion is not normalized, the resulting matrix won't be a pure rotation matrix, and
it might not preserve the orthogonality and length-preserving properties of rotation matrices, resulting in warping of
objects when interpolating.