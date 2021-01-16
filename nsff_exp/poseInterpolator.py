'''
Pose interpolator applies Quaternion SLERP to calculate intermediate poses given a starting point
and orientation and ending point and orientation.

Dependencies:
Q_Slerp.py - Script with function interpolate(...)
'''


from Q_Slerp import quaternion, interpolate
import numpy as np
import math
from numpy import array
from math import pi, cos, sin, acos, asin
# from PyKDL import *


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def quat2euler(q):
    ''' Return Euler angles corresponding to quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion

    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return mat2euler(nq.quat2mat(q))

def linear_translation(A, B, T):
    '''
    Interpolates between 2 start points A an dB linearly

    Input:
    A - Start point
    B - Ending point
    T - intermediate points within a range from 0 - 1, 0 representing point A and 1 representing point B

    Output
    C - intermediate pose as an array
    '''
    V_AB = B -A
    C = A + T*(V_AB)
    return C

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def linear_pose_interp(A_trans, A_rot, B_trans, B_rot, T):
    '''
    Pose interpolator that calculates intermediate poses of a vector connecting 2 points
    Interpolation is done linearly for both translation and rotation
    Translation is done assuming that point A is rotationally invariant.
    Rotation is done about the point A. Quaternion SLERP rotation is used to give the quickest rotation from A -> B
    ** "Roll" or twisting of the arm is not taken into account in this calculation. Separate interpolations have to be done for the roll angle

    Input:
    Starting points start_A [X,Y,Z,roll,pitch,yaw] list
    Ending points end_B in [X',Y',Z',roll',pitch',yaw'] list
        Yaw, Pitch, Roll calculated in radians within bounds [0, 2*pi]
        Sequence of rotation: Roll - 1st, Pitch - 2nd, Yaw - 3rd
    T = no of intermediate poses

    Output:
    list of positions and rotations stored into the variable track
        track is a dictionary with keys 'lin' and 'rot'

    track['lin'] - Linear interpolation of interval T of starting positions from A -> B
    track['rot'] - Slerp interpolation of quaternion of interval T, arranged as a list in [w x y z]
    # track['rot'] - Intermediate Yaw-Pitch-Roll poses of interval T, in sequence YPR

    '''

    track = {'lin': [], 'rot': []}

    # ra = start_A[3]; pa = start_A[4]; ya = start_A[5]  # Yaw/pitch/Roll for A and B
    # rb = end_B[3]; pb = end_B[4]; yb = end_B[5]

    # A = array(start_A[:3]); B = array(end_B[:3])
    # [vxa, vya, vza, wa] = Rotation(Vector(A_rot[0, 0], A_rot[0, 1], A_rot[0, 2]), 
                                   # Vector(A_rot[1, 0], A_rot[1, 1], A_rot[1, 2]), 
                                   # Vector(A_rot[2, 0], A_rot[2, 1], A_rot[2, 2])).GetQuaternion()  # Quaternion representation of start and end points

    # [vxb, vyb, vzb, wb] = Rotation(Vector(B_rot[0, 0], B_rot[0, 1], B_rot[0, 2]), 
                                   # Vector(B_rot[1, 0], B_rot[1, 1], B_rot[1, 2]), 
                                   # Vector(B_rot[2, 0], B_rot[2, 1], B_rot[2, 2])).GetQuaternion()

    q_a = rotmat2qvec(A_rot) 
    q_b = rotmat2qvec(B_rot) 

    # print('q_a ', q_a)
    # sys.exit()

    QA = quaternion(q_a[0], q_a[1], q_a[2], q_a[3])
    QB = quaternion(q_b[0], q_b[1], q_b[2], q_b[3])

    track['lin'] = linear_translation(A_trans, B_trans, T).tolist()
    q = interpolate(QA, QB, T)
    track['rot'] = [q.s] + (q.v).tolist()[0]    # List of quaternion [w x y z]

    # print('track ', track['rot'], track['lin'])
    # sys.exit()
    return qvec2rotmat(track['rot']), np.array(track['lin'])
    # print("Quaternion: ", track['rot'], "Y-P-R: ", quat2euler(track['rot']))
    # return track


if __name__ == "__main__":
    A = [0, 0, 0, 0, 0, 0]
    B = [1, 1, 1, -pi/2, pi/2, -pi/2]

    track = linear_pose_interp(A, B, 1)
    # print(track)
