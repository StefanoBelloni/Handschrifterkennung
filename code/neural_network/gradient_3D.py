import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------
# Time-dependent loss surface
# ---------------------------------

def loss(w, b, t):

    # moving minimum
    center_w = 1.5*np.sin(0.05*t)
    center_b = 1.0*np.cos(0.05*t)

    # changing curvature
    curvature = 1 + 0.3*np.sin(0.03*t)

    return (
        curvature*(w-center_w)**2
        +
        0.5*(b-center_b)**2
    )


# numerical gradient
def gradient(w,b,t):

    eps = 1e-4

    dw = (
        loss(w+eps,b,t)
        -
        loss(w-eps,b,t)
    )/(2*eps)

    db = (
        loss(w,b+eps,t)
        -
        loss(w,b-eps,t)
    )/(2*eps)

    return np.array([dw,db])


# ---------------------------------
# Gradient descent
# ---------------------------------

learning_rate = 0.03
steps = 220

w,b = -5,4

path=[]

for i in range(steps):

    path.append(
        [w,b,loss(w,b,i)]
    )

    g = gradient(w,b,i)

    w -= learning_rate*g[0]
    b -= learning_rate*g[1]


path=np.array(path)



# ---------------------------------
# Surface grid
# ---------------------------------

w_axis=np.linspace(-5,5,80)
b_axis=np.linspace(-5,5,80)

W,B=np.meshgrid(
    w_axis,
    b_axis
)



# ---------------------------------
# Plot setup
# ---------------------------------

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Loss")


ax.set_xlim(-5,5)
ax.set_ylim(-5,5)


# initial surface

surface = ax.plot_surface(
    W,
    B,
    loss(W,B,0),
    alpha=0.4
)


point, = ax.plot(
    [],
    [],
    [],
    "ro",
    markersize=8
)


trajectory, = ax.plot(
    [],
    [],
    [],
    "r",
    linewidth=3
)


gradient_line, = ax.plot(
    [],
    [],
    [],
    "k",
    linewidth=3
)



# ---------------------------------
# Animation
# ---------------------------------

def update(frame):

    global surface


    # remove old surface
    surface.remove()


    # draw new surface
    surface=ax.plot_surface(
        W,
        B,
        loss(W,B,frame),
        alpha=0.4
    )


    # current position

    w,b,z=path[frame]


    point.set_data(
        [w],
        [b]
    )

    point.set_3d_properties(
        [z]
    )


    trajectory.set_data(
        path[:frame+1,0],
        path[:frame+1,1]
    )

    trajectory.set_3d_properties(
        path[:frame+1,2]
    )


    # gradient arrow

    g=gradient(w,b,frame)

    scale=0.5


    gradient_line.set_data(
        [
            w,
            w-scale*g[0]
        ],
        [
            b,
            b-scale*g[1]
        ]
    )

    gradient_line.set_3d_properties(
        [
            z,
            loss(
                w-scale*g[0],
                b-scale*g[1],
                frame
            )
        ]
    )


    ax.set_title(
        f"Gradient Descent\nIteration {frame}"
    )


    return (
        surface,
        point,
        trajectory,
        gradient_line
    )



ani=FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=150,
    blit=False
)


plt.show()


# Save:
# ani.save(
#     "gradient_descent_3d.mp4",
#     writer="ffmpeg",
#     fps=10
# )