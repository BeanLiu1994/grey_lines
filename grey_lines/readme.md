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
