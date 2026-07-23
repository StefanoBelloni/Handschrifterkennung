import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --------------------------------
# Time-dependent loss function
# --------------------------------
def loss(w, b, t):
    """
    A moving loss landscape.

    The minimum slowly moves and
    the curvature changes with time.
    """

    center_w = 2*np.sin(t*0.08) + 1
    center_b = np.cos(t*0.08)

    curvature = 1 + 0.4*np.sin(t*0.1)

    return (
        curvature*(w-center_w)**2
        +
        0.5*(b-center_b)**2
    )


# Gradient (numerical)
def gradient(w,b,t):

    eps = 1e-5

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


# --------------------------------
# Gradient descent
# --------------------------------

learning_rate = 0.02   # smaller learning rate

steps = 120

w,b = -4,4

path=[]

for i in range(steps):

    t=i

    path.append([w,b])

    grad = gradient(w,b,t)

    w -= learning_rate*grad[0]
    b -= learning_rate*grad[1]


path=np.array(path)



# --------------------------------
# Grid
# --------------------------------

w_axis=np.linspace(-5,5,150)
b_axis=np.linspace(-5,5,150)

W,B=np.meshgrid(
    w_axis,
    b_axis
)



# --------------------------------
# Figure
# --------------------------------

fig,ax=plt.subplots(figsize=(7,6))


ax.set_xlabel("w")
ax.set_ylabel("b")

ax.set_title(
    "Gradient Descent on a Moving Loss Landscape"
)


# initial contour

contour=ax.contour(
    W,
    B,
    loss(W,B,0),
    levels=30
)


point,=ax.plot(
    [],
    [],
    "ro",
    markersize=10
)


trajectory,=ax.plot(
    [],
    [],
    "r-",
    linewidth=2
)


gradient_arrow=ax.quiver(
    [],
    [],
    [],
    []
)


text=ax.text(
    -4.5,
    4.5,
    ""
)



# --------------------------------
# Animation
# --------------------------------

def update(frame):

    global contour, gradient_arrow


    # remove old contour
    try:
        contour.remove()
    except:
        pass


    # redraw changing loss surface
    contour = ax.contour(
        W,
        B,
        loss(W, B, frame),
        levels=25
    )


    # current position
    w, b = path[frame]

    point.set_data(
        [w],
        [b]
    )


    trajectory.set_data(
        path[:frame+1,0],
        path[:frame+1,1]
    )


    # remove previous gradient arrow
    try:
        gradient_arrow.remove()
    except:
        pass


    # calculate gradient
    g = gradient(w,b,frame)


    # draw gradient direction
    gradient_arrow = ax.quiver(
        w,
        b,
        -g[0],
        -g[1],
        angles="xy",
        scale_units="xy",
        scale=20
    )


    text.set_text(
        f"Iteration: {frame}\n"
        f"Learning rate: {learning_rate:.3f}"
    )


    return (
        point,
        trajectory,
        gradient_arrow,
        text
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
#     "changing_gradient_descent.mp4",
#     writer="ffmpeg",
#     fps=10
# )