What is LIP-BFGS?
-----------------

LIP-BFGS stands for Limited-memory Interior-Point 
  Broyden-Fletcher-Goldfarb-Shanno algorithm; 
  and is simply a an interior-point (IP) method which uses the 
  limited-memory BFGS (L-BFGS) algorithm.

LIP-BFGS was written in Matlab by Jesse Lu in the fall of 2011.


Why use LIP-BFGS?
-----------------

1.  LIP-BFGS can handle large problems (_x_ with millions of elements).
1.  LIP-BFGS is easy to modify. 
    The algorithm and implementation are simple and well-documented.
1.  LIP-BFGS is free. 
    Free as in public domain (see License), use and modify as you will.


Installing and running LIP-BFGS
-------------------------------

To install LIP-BFGS, simply extract all files to a directory.

To use LIP-BFGS, simply open matlab within the directory, 
  and run the _example_ program from the Matlab command line.
The _test_all_ program can also be run from the Matlab command line.


Documentation
-------------

All Matlab functions are well documented. 
To start, try typing _help lip_bfgs_ from the Matlab command line.

To understand the algorithm, look at _theory.pdf_.


What problem does LIP-BFGS solve?
---------------------------------

LIP-BFGS is designed to minimize an objective function with 
  linear equality constraints and simple bounds on the variables:

    minimize f(x)
    subject to  Ax - b = 0
                l <= x <= u.

In particular, LIP-BFGS is geared towards problems where the size of _x_
  is large (i.e. > 1 million elements).


Using LIP-BFGS
--------------

LIP-BFGS only requires the following input parameters:

1.  _x_, a vector of length _n_. The initial value for the optimization.
1.  _f(x)_, the design objective. A function returning a real scalar.
1.  _g(x)_, the gradient of _f(x)_. 
    A function which returns a vector of length _n_.
1.  _A_ and _b_, the matrix and vector defining the equality constraint.
    _A_ must be of size _p_ by _n_ and _b_ must be of length _p_.
1.  _l_ and _u_, the bound constraints on _x_. 
    Both must be real-valued and of length _n_.


Note on using complex values
----------------------------

LIP-BFGS still works with complex-valued _x_, _g(x)_, _A_, and _b_;
  however, the bounds on _x_, _l_ and _u_, only apply to the real-part of _x_
  and therefore must be real-valued.
Lastly, _f(x)_ is required to return a real-valued scalar as well.


License
-------

LIP-BFGS is public domain. Feel free to use and modify as you like.

