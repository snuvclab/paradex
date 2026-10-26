"""Object-pose symmetry metadata shared by replay pipelines."""

from __future__ import annotations

import numpy as np


# Continuous rotation axes in the object-local coordinates of the meshes under
# shared_data/mesh_new. These are axis *directions* through each mesh's object
# origin. For orange, inspection of mesh_new/orange/orange.obj places the stem
# cap at local +Y (the highest-Y cap is centered near X=Z=0), so local +Y is the
# semantic centre-to-stem axis. Do not tilt it to follow sub-millimetre scan
# asymmetry. The Pringles and lemon axes are the normalized dominant PCA axes
# of their original OBJ files.
CONTINUOUS_SYMMETRY_AXIS_OBJECT = {
    "apple": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    # Orange is rotationally equivalent only around the origin-to-stem axis.
    # Keep its +Y direction fixed: the stem/top and bottom are not exchangeable.
    "orange": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "lemon": np.array([0.128844, -0.039493, 0.990878], dtype=np.float64),
    "pringles": np.array([-0.241377, 0.511070, 0.824951], dtype=np.float64),
    # mesh_new/spicemill is a rotational body elongated along local Y. Its
    # top and bottom are distinct, so longitudinal-axis flips are not allowed.
    "spicemill": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    # pastel_blue_cup is rotationally equivalent around its local Y axis. Its
    # opening and base are distinct, so longitudinal-axis flips are forbidden.
    "pastel_blue_cup": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    # Both cactus meshes are elongated, approximately rotational bodies around
    # local Y. Their two ends are different, so only spins around +Y are
    # equivalent; do not allow a longitudinal-axis flip.
    "cactus": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "cactus2": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}

# Centers of continuous-symmetry axes when the mesh origin is not on the axis.
# mesh_new/pastel_blue_cup is centered around this local-XZ location; rotating
# about the object origin would make the cup orbit by roughly 4.8 cm.
CONTINUOUS_SYMMETRY_CENTER_OBJECT = {
    "pastel_blue_cup": np.array([-0.047589, 0.0, -0.000043], dtype=np.float64),
    # The cactus OBJ origins are substantially offset from the body axes.
    # X/Z are the mesh_new bounding-box midpoints; Y is arbitrary on an axis
    # parallel to local Y and is kept at zero for a stable representation.
    "cactus": np.array([-0.083984, 0.0, -0.0203275], dtype=np.float64),
    "cactus2": np.array([0.089161, 0.0, -0.019435], dtype=np.float64),
}

# Objects with exactly two orientation-equivalent representatives: identity and
# a 180-degree proper rotation about the specified object-local axis line.
DISCRETE_HALF_TURN_SYMMETRY_OBJECT = {
    # mesh_new/fruit_cutter_base has two equivalent opposing sides around its
    # local Y axis.  Its OBJ spans Y=[0, 0.06276143], and the X/Z bounding-box
    # midpoint locates the rotation axis through the body rather than through
    # the OBJ origin.
    "fruit_cutter_base": {
        "axis": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "center": np.array([0.00007006, 0.031380715, 0.0], dtype=np.float64),
    },
    "potato_masher": {
        "axis": np.array(
            [-0.520395458, 0.167048752, 0.837426583], dtype=np.float64
        ),
        "center": np.array(
            [-0.004027027, 0.001251778, 0.006334313], dtype=np.float64
        ),
    },
}

# Rectangular cuboids with three distinct edge lengths have only four proper
# rotational symmetries: identity and 180-degree rotations about local X/Y/Z.
# Values are the rotation centers in original mesh coordinates. coffee_tin's
# OBJ origin is at one end rather than at its bounding-box center.
CUBOID_SYMMETRY_CENTER_OBJECT = {
    "coffee_tin": np.array([0.0, 0.095658, 0.0], dtype=np.float64),
}

# A cylinder whose two ends are geometrically interchangeable may also reverse
# its longitudinal axis before applying the continuous spin equivalence.
AXIS_FLIP_SYMMETRIC_OBJECTS = frozenset({"pringles"})
