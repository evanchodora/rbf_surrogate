## Python Tool for Training RBF Surrogate Models

Evan Chodora - echodor@clemson.edu

`python rbf_surrogate.py -h` for help and options

Can be used with a variety of RBFs (see the dictionary of function names below) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

Makes use of the Spatial Distance calculation functions from SciPy to compute the radial distance matrices for the
radial basis function calculations.
(https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

Included RBFs:
 - Linear: "linear"
 - Cubic: "cubic"
 - Absolute Value: "absolute"
 - Multiquadratic: "multiquadratic"
 - Inverse Multiquadratic: "inverse_multiquadric"
 - Gaussian: "gaussian"
 - Thin Plate: "thin_plate"

### Training Example:

`python rbf_surrogate.py -t train -r gaussian -x x_data.dat -y y_data.dat -m model.db`

### Prediction Example
Creates an output file `y_pred.dat` based on the supplied input values:

`python rbf_surrogate.py -t predict -x x_pred.dat -m model.db`

### Input and Output File Format:
Files can be supplied in a comma-separated value format for `x` and `y` with a header line.

Example (`x_data.dat`):

```
x0,x1,x2
1.34,5,5.545
3.21,0.56,9.34
```
