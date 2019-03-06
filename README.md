## Python Tool for Training RBF Surrogate Models

Evan Chodora - echodor@clemson.edu

`python rbf_surrogate.py -h` for help and options

Can be used with a variety of RBFs (see the dictionary of function names below) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

Included RBFs:
 - Linear: "linear"
 - Cubic: "cubic"
 - Multiquadratic: "multiquadratic"
 - Inverse Multiquadratic: "inverse_multiquadric"
 - Gaussian: "gaussian"
 - Thin Plate: "thin_plate"

### Training Example:

`python rbf_surrogate.py -t train -r gaussian -x x_data.dat -y y_data.dat -m model.db`

### Prediction Example - creates an output file `y_pred.dat` based on the supplied input values:

`python rbf_surrogate.py -t predict -x x_pred.dat -m model.db`

### Input and Output File Format:

Files can be supplied in a comma-separated value format for `x` and `y` with a header line.

Example (`x_data.dat`):

```
x0,x1,x2
1.34,5,5.545
```
