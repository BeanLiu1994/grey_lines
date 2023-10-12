# what

Draw a photo using lines with opacity.

# how

line contributes some intensity to each pixel, and pixel is from image input.

this can be treated as a constrained linear problem.

Assume there is n lines, and m pixels to fit,

the equation should be like

```
$$\text{argmin}_(x) \|\|Ax - b\|\|₂, \text{s.t. } x \geq 0$$
```

in which x is a vector and x_i means intensity for i-th line, A is line to pixel contribution matrix, and b is vector of input image intensity.

# input

input is a image and canvas configuration, in this code there is a rect and circle canvas that for easy use.

canvas contains dots between which i can draw lines, and image area indicates where the input image should be placed.

you should be able to customize a canvas layout.

# limit

ignore the constraint and solve speed is not very fast.

not built to a cli tool now.

# how to use

change load_image path in main function and

in your python>=3.11 env (venv or other), run

```
poetry install
python grey_lines/main.py
```

![in](img/in.png)
![out](img/out.svg)
![out2](img/out2.svg)