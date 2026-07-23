import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -------------------------------
# Tiny neural network
# -------------------------------

np.random.seed(1)

x = np.array([1.0, 0.5])
target = 1.0

W1 = np.random.randn(2,2)*0.5
W2 = np.random.randn(2)*0.5

lr = 0.2


def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward(W1,W2):

    z = W1.T @ x
    h = sigmoid(z)

    y = W2 @ h

    return h,y



def backward(W1,W2,h,y):

    error = y-target

    dW2 = error*h

    dh = error*W2

    dz = dh*h*(1-h)

    dW1=np.outer(x,dz)

    return dW1,dW2,error



# create training history

history=[]

for epoch in range(10):

    h,y=forward(W1,W2)

    dW1,dW2,error=backward(
        W1,W2,h,y
    )

    history.append(
        {
        "h":h,
        "y":y,
        "error":error,
        "dW1":dW1,
        "dW2":dW2,
        "W1":W1.copy(),
        "W2":W2.copy()
        }
    )

    W1-=lr*dW1
    W2-=lr*dW2



# -------------------------------
# Drawing
# -------------------------------

fig,ax=plt.subplots(figsize=(10,6))

ax.set_xlim(0,10)
ax.set_ylim(0,7)

ax.axis("off")


inputs=[(1,5),(1,2)]
hidden=[(5,5),(5,2)]
output=(8,3.5)


# neurons

nodes=[]

def draw_node(pos,name):

    c=plt.Circle(
        pos,
        0.45,
        facecolor="white",
        edgecolor="black",
        linewidth=2
    )

    ax.add_patch(c)

    ax.text(
        pos[0],
        pos[1],
        name,
        ha="center",
        va="center",
        fontsize=12
    )

    nodes.append(c)


for i,p in enumerate(inputs):
    draw_node(p,f"x{i+1}")

for i,p in enumerate(hidden):
    draw_node(p,f"h{i+1}")

draw_node(output,"ŷ")



# edges

edges=[]

def connect(a,b):

    line,=ax.plot(
        [a[0],b[0]],
        [a[1],b[1]],
        linewidth=2,
        color="gray"
    )

    edges.append(line)


for a in inputs:
    for b in hidden:
        connect(a,b)


for b in hidden:
    connect(b,output)



text=ax.text(
    1,
    6.5,
    "",
    fontsize=14
)


info=ax.text(
    7,
    6,
    "",
    fontsize=11
)



signal,=ax.plot(
    [],
    [],
    "o",
    markersize=15
)



# -------------------------------
# Animation
# -------------------------------

# -------------------------------
# Click-controlled animation
# -------------------------------

current_frame = [0]   # use list so it can be modified inside function


def update(frame):

    data = history[frame//4]

    phase = frame % 4

    # reset colors
    for n in nodes:
        n.set_facecolor("white")

    for e in edges:
        e.set_color("gray")
        e.set_linewidth(2)


    if phase == 0:

        text.set_text(
            "1) Forward propagation\n"
            "Input → Hidden → Prediction"
        )

        nodes[0].set_facecolor("cyan")
        nodes[1].set_facecolor("cyan")

        edges[0].set_color("blue")
        edges[1].set_color("blue")

        info.set_text(
            f"Hidden activation:\n"
            f"h={data['h']}\n"
            f"Prediction ŷ={data['y']:.3f}"
        )


    elif phase == 1:

        text.set_text(
            "2) Loss calculation\n"
            "Compare prediction with target"
        )

        nodes[-1].set_facecolor("orange")

        info.set_text(
            f"Target = {target}\n"
            f"Error = {data['error']:.3f}"
        )


    elif phase == 2:

        text.set_text(
            "3) Backpropagation\n"
            "Gradients flow backward"
        )

        for e in edges:
            e.set_color("red")
            e.set_linewidth(4)

        info.set_text(
            f"dW2 = {data['dW2']}\n\n"
            f"dW1 = {data['dW1']}"
        )


    else:

        text.set_text(
            "4) Gradient descent update\n"
            "W ← W - η∇W"
        )

        for e in edges:
            e.set_color("green")

        info.set_text(
            "Weights are updated\n"
            "and the cycle repeats"
        )


    return nodes + edges + [text, info]



# draw first frame
update(0)



def onclick(event):

    if event.inaxes != ax:
        return

    current_frame[0] += 1

    if current_frame[0] >= len(history)*4:
        current_frame[0] = 0

    update(current_frame[0])

    fig.canvas.draw_idle()



fig.canvas.mpl_connect(
    "button_press_event",
    onclick
)


plt.show()