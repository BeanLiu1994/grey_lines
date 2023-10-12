# what

Draw a photo using lines with opacity.

# how

Each line contributes some intensity to each pixel, and each pixel is from the input image. This can be treated as a constrained linear problem. Assume there are n lines and m pixels to fit. The equation should be like:

$$\arg \underset{x}{\mathop{min}} \|Ax - b\|₂, \text{s.t. } x \geq 0$$

In this equation, x is a vector and x_i means intensity for the i-th line, A is the line to pixel contribution matrix, and b is the vector of input image intensity.

# input

The input is an image and canvas configuration. In this code, there is a rectangle and circle canvas for easy use. The canvas contains dots between which lines can be drawn, and the image area indicates where the input image should be placed. You should be able to customize a canvas layout.

# limit

Ignore the non-negative constraint to improve solve speed.
If problem is too large, speed is not very fast.

# how to use

In your python>=3.11 env (venv or other), run

```
poetry install
linedraw --help
linedraw in.png out.svg
```

![in](img/in.png)
![out](img/out.svg)
![out2](img/out2.svg)