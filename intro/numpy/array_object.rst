..  


.. currentmodule:: numpy

The Julia Vector
======================

.. contents:: Section contents
    :local:
    :depth: 1

What are Julia vectors?
--------------------------------

Julia vectors
............

:**Julia** provides:

    - Builtin multi-dimensional arrays

    - closer to hardware (efficiency)

    - designed for scientific computation (convenience)

|

.. sourcecode:: pycon

    julia> v = [0, 1, 2, 3]
      4-element Vector{Int64}:     
       0                           
       1                           
       2                           
       3                           
    
.. tip::

    For example, a vector containing:

    * values of an experiment/simulation at discrete time steps

    * signal recorded by a measurement device, e.g. sound wave

    * pixels of an image, grey-level or colour

    * 3-D data measured at different X-Y-Z positions, e.g. MRI scan

    * ...

**Why it is useful:** Memory-efficient container that provides fast numerical
operations.

.. julia::
    julia> using BenchmarkTools
    julia> L = 1:1000
    julia> @btime [a^2 for a in L];
  1.342 µs (2 allocations: 7.97 KiB)


.. support for multidimensional arrays

.. diagram, import conventions

.. scope of this tutorial: drill in features of vector manipulation in
   Julia, and try to give some indication on how to get things done
   in good style

.. a fixed number of elements (cf. certain exceptions)
.. each element of same size and type


Julia Reference documentation
..............................

- On the web: https://docs.julialang.org/

- Interactive help:

  .. ipython::

     julia> ? Matrix
    search: Matrix BitMatrix DenseMatrix StridedMatrix AbstractMatrix

      Matrix{T} <: AbstractMatrix{T}

      Two-dimensional dense array with elements of type T, often used to represent a mathematical matrix. Alias for Array{T,2}.

      See also fill, zeros, undef and similar for creating matrices.
      
  

- Looking for something:


  .. ipython::

     julia> mat<Tab><Tab> # press tab key two times
     MathConstants  Matrix




Creating vectors
---------------

Manual construction of vectors
..............................

* **1-D**:

  .. sourcecode:: pycon

    julia> a = [0, 1, 2, 3]
   
    julia> ndims(a)
    1
    julia> size(a)
    (4,)
    julia> length(a)
    4

* **2-D, 3-D, ...**:

  .. sourcecode:: pycon

    julia> b = [0  1  2; 3 4 5]
    2×3 Matrix{Int64}:
     0  1  2
     3  4  5
    julia> ndims(b)
    2
    julia> size(b)
    (2, 3)
    julia> length(b)     # returns the total size = 2 x 3
    6

    julia> c = [1 3 5
               2 4 6;;;
               7 9 11
               8 10 12]
        2×3×2 Array{Int64, 3}:
        [:, :, 1] =
         1  3  5
         2  4  6

        [:, :, 2] =
         7   9  11
         8  10  12


.. topic:: **Exercise: Simple arrays**
    :class: green

    * Create a simple two dimensional array. First, redo the examples
      from above. And then create your own: how about odd numbers
      counting backwards on the first row, and even numbers on the second?
    * Use the functions `length`, `size` on these arrays.
      How do they relate to each other? And to the ``ndims`` function on 
      the arrays?

Functions for creating vectors
..............................

.. tip::

    In practice, we rarely enter items one by one...

* Evenly spaced:

  .. sourcecode:: pycon

    julia> a = collect(1:10)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    
    julia> b = 1:2:10 # start, step ,end (inclusive), 
    julia> b
    [1, 3, 5, 7]

* or by number of points:

  .. sourcecode:: pycon

    julia> c =  range(0,1,length = 6)   # start, end, num-points
    0.0:0.2:1.0
    julia> d = LinRange(0,1,6) #less overhead, prone to floating point errors
    6-element LinRange{Float64, Int64}:
    0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    
    
* Common arrays:

  .. sourcecode:: pycon

    julia> a = ones(3,3)
    3×3 Matrix{Float64}:
     1.0  1.0  1.0
     1.0  1.0  1.0
     1.0  1.0  1.0

    julia> b = zeros(3,3)
    3×3 Matrix{Float64}:
     0.0  0.0  0.0
     0.0  0.0  0.0
     0.0  0.0  0.0  

    julia> using LinearAlgebra

    julia> c = Matrix(1.0I,3,3) #identity matrix
    3×3 Matrix{Float64}:
     1.0  0.0  0.0
     0.0  1.0  0.0
     0.0  0.0  1.0

* :mod:`rand`: random numbers:

  .. sourcecode:: pycon
                              
  julia> using Random           
                                
  julia> Random.seed!(123)      
  TaskLocalRNG()                
                                
  julia> a = rand(5,) #uniform random numbers in the range [0,1]          
  5-element Vector{Float64}:    
   0.521213795535383            
   0.5868067574533484           
   0.8908786980927811           
   0.19090669902576285          
   0.5256623915420473 
   
   julia> b = randn(5,) #Gaussian random vector
5-element Vector{Float64}:
  0.9809798121241488
  0.0799568295050599
  1.5491245530427917
 -1.3416092408832219
  0.41216163468296796   
   
   
.. topic:: **Exercise: Creating arrays using functions**
   :class: green

   * Experiment with ``:``, ``range``, ``ones``, ``zeros``, ``I`` and
     ``diagm``.
   * Create different kinds of arrays with random numbers.
   * Try setting the seed before creating an array with random values.
   * Look at the function ``empty``. What does it do? When might this be
     useful?

.. EXE: construct 1 2 3 4 5
.. EXE: construct -5, -4, -3, -2, -1
.. EXE: construct 2 4 6 8
.. EXE: look what is in an empty() array
.. EXE: construct 15 equispaced numbers in range [0, 10]

Basic data types
----------------

You may have noticed that, in some instances, array elements are displayed with
a trailing dot (e.g. ``2.`` vs ``2``). This is due to a difference in the
data-type used:

.. sourcecode:: pycon

    julia> a = [1, 2, 3]
    julia> eltype(a)
    Int64

    julia> b = [1., 2., 3.]
    julia> eltype(b)
    Float64

.. tip::

    Different data-types allow us to store data more compactly in memory,
    but most of the time we simply work with floating point numbers.
    Note that, in the example above, Julia auto-detects the data-type
    from the input.

-----------------------------

The **default** data type is floating point:

.. sourcecode:: pycon

    julia> a = ones(3, 3)
    julia> eltype(a)
    Float64

There are also other types:

:Complex:

  .. sourcecode:: pycon

        julia> d = [1+2im, 3+4im, 5+6*1im]
        julia> eltype(d)
        Complex{Int64}
:Bool:

  .. sourcecode:: pycon

        julia> e = [true, false, false, true]
        julia> eltype(e)
        Bool

:Strings:

  .. sourcecode:: pycon

        julia> f = ["Bonjour", "Hello", "Hallo"]
        julia> eltype(f)
        String
        
:Much more:

      `Int8`,`UInt8`,`Int16`,`UInt16`,`Int32`,`UInt32`,`Int64`,`UInt64`,`Int128`,`UInt128`,`Float16`,`Float32`, and `Float64`




Basic visualization
-------------------

Now that we have our first data arrays, we are going to visualize them.

.. sourcecode:: pycon

    julia> using Plots
    julia> x = 1:100;y = 1:100; 
    julia> plot(x, y)       # line plot    # doctest: +SKIP
    

Or, if you have enabled interactive plots with ``%matplotlib``:

.. sourcecode:: pycon

    julia> plt.plot(x, y)       # line plot    # doctest: +SKIP

* **1D plotting**:

.. sourcecode:: pycon

  julia> x = np.linspace(0, 3, 20)
  julia> y = np.linspace(0, 9, 20)
  julia> plt.plot(x, y)       # line plot
  [<matplotlib.lines.Line2D object at ...>]
  julia> plt.plot(x, y, 'o')  # dot plot
  [<matplotlib.lines.Line2D object at ...>]

.. image:: auto_examples/images/sphx_glr_plot_basic1dplot_001.png
    :width: 40%
    :target: auto_examples/plot_basic1dplot.html
    :align: center

* **2D arrays** (such as images):

.. sourcecode:: pycon
  julia> using PyPlot
  julia> image = rand(1:255,10,10)
  julia> PyPlot.imshow(image)
 

.. image:: auto_examples/images/sphx_glr_plot_basic2dplot_001.png
    :width: 50%
    :target: auto_examples/plot_basic2dplot.html
    :align: center

.. topic:: **Exercise: Simple visualizations**
   :class: green

   * Plot some simple arrays: a cosine as a function of time and a 2D
     matrix.
   


Indexing and slicing
--------------------

The items of an array can be accessed and assigned to the same way as
other Python sequences (e.g. lists):

.. sourcecode:: pycon

    julia> a = np.arange(10)
    julia> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    julia> a[0], a[2], a[-1]
    (0, 2, 9)

.. warning::

   Indices begin at 0, like other Python sequences (and C/C++).
   In contrast, in Fortran or Matlab, indices begin at 1.

The usual python idiom for reversing a sequence is supported:

.. sourcecode:: pycon

   julia> a[::-1]
   array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

For multidimensional arrays, indices are tuples of integers:

.. sourcecode:: pycon

    julia> a = np.diag(np.arange(3))
    julia> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 2]])
    julia> a[1, 1]
    1
    julia> a[2, 1] = 10 # third line, second column
    julia> a
    array([[ 0,  0,  0],
           [ 0,  1,  0],
           [ 0, 10,  2]])
    julia> a[1]
    array([0, 1, 0])


.. note::

  * In 2D, the first dimension corresponds to **rows**, the second
    to **columns**.
  * for multidimensional ``a``, ``a[0]`` is interpreted by
    taking all elements in the unspecified dimensions.

**Slicing**: Arrays, like other Python sequences can also be sliced:

.. sourcecode:: pycon

    julia> a = np.arange(10)
    julia> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    julia> a[2:9:3] # [start:end:step]
    array([2, 5, 8])

Note that the last index is not included! :

.. sourcecode:: pycon

    julia> a[:4]
    array([0, 1, 2, 3])

All three slice components are not required: by default, `start` is 0,
`end` is the last and `step` is 1:

.. sourcecode:: pycon

    julia> a[1:3]
    array([1, 2])
    julia> a[::2]
    array([0, 2, 4, 6, 8])
    julia> a[3:]
    array([3, 4, 5, 6, 7, 8, 9])

A small illustrated summary of NumPy indexing and slicing...

.. only:: latex

    .. image:: ../../pyximages/numpy_indexing.pdf
        :align: center

.. only:: html

    .. image:: ../../pyximages/numpy_indexing.png
        :align: center
        :width: 70%

You can also combine assignment and slicing:

.. sourcecode:: pycon

   julia> a = np.arange(10)
   julia> a[5:] = 10
   julia> a
   array([ 0,  1,  2,  3,  4, 10, 10, 10, 10, 10])
   julia> b = np.arange(5)
   julia> a[5:] = b[::-1]
   julia> a
   array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])

.. topic:: **Exercise: Indexing and slicing**
   :class: green

   * Try the different flavours of slicing, using ``start``, ``end`` and
     ``step``: starting from a linspace, try to obtain odd numbers
     counting backwards, and even numbers counting forwards.
   * Reproduce the slices in the diagram above. You may
     use the following expression to create the array:

     .. sourcecode:: pycon

        julia> np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
        array([[ 0,  1,  2,  3,  4,  5],
               [10, 11, 12, 13, 14, 15],
               [20, 21, 22, 23, 24, 25],
               [30, 31, 32, 33, 34, 35],
               [40, 41, 42, 43, 44, 45],
               [50, 51, 52, 53, 54, 55]])

.. topic:: **Exercise: Array creation**
    :class: green

    Create the following arrays (with correct data types)::

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 2],
         [1, 6, 1, 1]]

        [[0., 0., 0., 0., 0.],
         [2., 0., 0., 0., 0.],
         [0., 3., 0., 0., 0.],
         [0., 0., 4., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 6.]]

    Par on course: 3 statements for each

    *Hint*: Individual array elements can be accessed similarly to a list,
    e.g. ``a[1]`` or ``a[1, 2]``.

    *Hint*: Examine the docstring for ``diag``.

.. topic:: Exercise: Tiling for array creation
    :class: green

    Skim through the documentation for ``np.tile``, and use this function
    to construct the array::

        [[4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1],
         [4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1]]

Copies and views
----------------

A slicing operation creates a **view** on the original array, which is
just a way of accessing array data. Thus the original array is not
copied in memory. You can use ``np.may_share_memory()`` to check if two arrays
share the same memory block. Note however, that this uses heuristics and may
give you false positives.

**When modifying the view, the original array is modified as well**:

.. sourcecode:: pycon

    julia> a = np.arange(10)
    julia> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    julia> b = a[::2]
    julia> b
    array([0, 2, 4, 6, 8])
    julia> np.may_share_memory(a, b)
    True
    julia> b[0] = 12
    julia> b
    array([12,  2,  4,  6,  8])
    julia> a   # (!)
    array([12,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    julia> a = np.arange(10)
    julia> c = a[::2].copy()  # force a copy
    julia> c[0] = 12
    julia> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    julia> np.may_share_memory(a, c)
    False



This behavior can be surprising at first sight... but it allows to save both
memory and time.


.. EXE: [1, 2, 3, 4, 5] -> [1, 2, 3]
.. EXE: [1, 2, 3, 4, 5] -> [4, 5]
.. EXE: [1, 2, 3, 4, 5] -> [1, 3, 5]
.. EXE: [1, 2, 3, 4, 5] -> [2, 4]
.. EXE: create an array [1, 1, 1, 1, 0, 0, 0]
.. EXE: create an array [0, 0, 0, 0, 1, 1, 1]
.. EXE: create an array [0, 1, 0, 1, 0, 1, 0]
.. EXE: create an array [1, 0, 1, 0, 1, 0, 1]
.. EXE: create an array [1, 0, 2, 0, 3, 0, 4]
.. CHA: archimedean sieve

.. topic:: Worked example: Prime number sieve
   :class: green

   .. image:: images/prime-sieve.png

   Compute prime numbers in 0--99, with a sieve

   * Construct a shape (100,) boolean array ``is_prime``,
     filled with True in the beginning:

   .. sourcecode:: pycon

        julia> is_prime = np.ones((100,), dtype=bool)

   * Cross out 0 and 1 which are not primes:

   .. sourcecode:: pycon

       julia> is_prime[:2] = 0

   * For each integer ``j`` starting from 2, cross out its higher multiples:

   .. sourcecode:: pycon

       julia> N_max = int(np.sqrt(len(is_prime) - 1))
       julia> for j in range(2, N_max + 1):
       ...     is_prime[2*j::j] = False

   * Skim through ``help(np.nonzero)``, and print the prime numbers

   * Follow-up:

     - Move the above code into a script file named ``prime_sieve.py``

     - Run it to check it works

     - Use the optimization suggested in `the sieve of Eratosthenes
       <https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes>`_:

      1. Skip ``j`` which are already known to not be primes

      2. The first number to cross out is :math:`j^2`

Fancy indexing
--------------

.. tip::

    NumPy arrays can be indexed with slices, but also with boolean or
    integer arrays (**masks**). This method is called *fancy indexing*.
    It creates **copies not views**.

Using boolean masks
...................

.. sourcecode:: pycon

    julia> rng = np.random.default_rng(27446968)
    julia> a = rng.integers(0, 21, 15)
    julia> a
    array([ 3, 13, 12, 10, 10, 10, 18,  4,  8,  5,  6, 11, 12, 17,  3])
    julia> (a % 3 == 0)
    array([ True, False,  True, False, False, False,  True, False, False,
           False,  True, False,  True, False,  True])
    julia> mask = (a % 3 == 0)
    julia> extract_from_a = a[mask] # or,  a[a%3==0]
    julia> extract_from_a           # extract a sub-array with the mask
    array([ 3, 12, 18,  6, 12,  3])

Indexing with a mask can be very useful to assign a new value to a sub-array:

.. sourcecode:: pycon

    julia> a[a % 3 == 0] = -1
    julia> a
    array([-1, 13, -1, 10, 10, 10, -1,  4,  8,  5, -1, 11, -1, 17, -1])


Indexing with an array of integers
..................................

.. sourcecode:: pycon

    julia> a = np.arange(0, 100, 10)
    julia> a
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

Indexing can be done with an array of integers, where the same index is repeated
several time:

.. sourcecode:: pycon

    julia> a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list
    array([20, 30, 20, 40, 20])

New values can be assigned with this kind of indexing:

.. sourcecode:: pycon

    julia> a[[9, 7]] = -100
    julia> a
    array([   0,   10,   20,   30,   40,   50,   60, -100,   80, -100])

.. tip::

  When a new array is created by indexing with an array of integers, the
  new array has the same shape as the array of integers:

  .. sourcecode:: pycon

    julia> a = np.arange(10)
    julia> idx = np.array([[3, 4], [9, 7]])
    julia> idx.shape
    (2, 2)
    julia> a[idx]
    array([[3, 4],
           [9, 7]])


____

The image below illustrates various fancy indexing applications

.. only:: latex

    .. image:: ../../pyximages/numpy_fancy_indexing.pdf
        :align: center

.. only:: html

    .. image:: ../../pyximages/numpy_fancy_indexing.png
        :align: center
        :width: 80%

.. topic:: **Exercise: Fancy indexing**
    :class: green

    * Again, reproduce the fancy indexing shown in the diagram above.
    * Use fancy indexing on the left and array creation on the right to assign
      values into an array, for instance by setting parts of the array in
      the diagram above to zero.

.. We can even use fancy indexing and :ref:`broadcasting <broadcasting>` at
.. the same time:
..
.. .. sourcecode:: pycon
..
..     julia> a = np.arange(12).reshape(3,4)
..     julia> a
..     array([[ 0,  1,  2,  3],
..            [ 4,  5,  6,  7],
..            [ 8,  9, 10, 11]])
..     julia> i = np.array([[0, 1], [1, 2]])
..     julia> a[i, 2] # same as a[i, 2*np.ones((2, 2), dtype=int)]
..     array([[ 2,  6],
..            [ 6, 10]])
