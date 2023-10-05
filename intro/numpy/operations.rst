.. currentmodule:: numpy

Numerical operations on vectors and Matrices
==============================

.. contents:: Section contents
    :local:
    :depth: 1


Elementwise operations
----------------------

Basic operations
................

With scalars:

.. sourcecode:: pycon

    julia> a =[1, 2, 3, 4]
    julia> a .+ 1 #space is necessary between a and .+
    [2, 3, 4, 5]
    julia> 2 .^a #space is necessary between 2 and .^
    [2,  4,  8, 16]

All arithmetic operates elementwise:

.. sourcecode:: pycon

	julia> b = ones(4) .+ 1     
	4-element Vector{Float64}:  
	 2.0                        
	 2.0                        
	 2.0                        
	 2.0                        
	                            
	julia> a - b                
	4-element Vector{Float64}:  
	 -1.0                       
	  0.0                       
	  1.0                       
	  2.0                       
	                            

    
    julia> a .* b
    array([2.,  4.,  6.,  8.])

    julia> j = collect(0:4)
    julia> 2 .^(j .+ 1) .- j
		5-element Vector{Int64}:
		  2
		  3
		  6
		 13
		 28
	

.. note:: **Array multiplication is matrix multiplication:**

    .. sourcecode:: pycon

        julia> c = ones(3, 3)   
		3×3 Matrix{Float64}:    
		 1.0  1.0  1.0          
		 1.0  1.0  1.0          
		 1.0  1.0  1.0          
		                        
		julia> c*c              
		3×3 Matrix{Float64}:    
		 3.0  3.0  3.0          
		 3.0  3.0  3.0          
		 3.0  3.0  3.0          
		                        

.. note:: **Elementwise multiplication:**

    .. sourcecode:: pycon

        julia> c .* c
        3×3 Matrix{Float64}:
		 3.0  3.0  3.0
		 3.0  3.0  3.0
		 3.0  3.0  3.0
		julia> d = rand(1:5,3,3)
		3×3 Matrix{Int64}:
		 5  1  3
		 1  4  3
		 4  3  4

		julia> c .*d
		3×3 Matrix{Float64}:
		 5.0  1.0  3.0
		 1.0  4.0  3.0
		 4.0  3.0  4.0
			   
.. topic:: **Exercise: Elementwise operations**
   :class: green

    * Try simple arithmetic elementwise operations: add even elements
      with odd elements
    * Generate:

      * ``[2^0, 2^1, 2^2, 2^3, 2^4]``
      * ``a_j = 2^(3*j) - j``

Other operations
................

**Comparisons:**

.. sourcecode:: pycon

    julia> a = [1, 2, 3, 4]
    julia> b = [4, 2, 2, 4]
    julia> a == b        
	false                
	                     
	julia> a .== b       
	4-element BitVector: 
	 0                   
	 1                   
	 0                   
	 1                   
	julia> a .> b             
	4-element BitVector:      
	 0                        
	 0                        
	 1                        
	 0                        
                             

**Logical operations:**

.. sourcecode:: pycon

    julia> a = Bool[1, 1, 0, 0]
    julia> b = Bool[1, 0, 1, 0]
    julia> a .| b
	4-element BitVector:
	 1
	 1
	 1
	 0	
    julia> a .& b
	4-element BitVector:
	 1
	 0
	 0
	 0

**Transcendental functions:**

.. sourcecode:: pycon

    julia> a = collect(0:4)
    julia> sin.(a)
    [0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ]
    julia> exp.(a)
    [ 1.        ,   2.71828183,   7.3890561 ,  20.08553692,  54.59815003]
    julia> log.(exp.(a))
    [0., 1., 2., 3., 4.]
	julia> sind.(a) #sine in degrees
	[0.0, 0.01745240643728351, 0.03489949670250097, 0.052335956242943835, 0.0697564737441253]


**Shape mismatches**

.. sourcecode:: pycon

    julia> a = collect(0:4)
    julia> a + [1,2]
	ERROR: DimensionMismatch: dimensions must match: a has dims (Base.OneTo(5),), b has dims (Base.OneTo(2),), mismatch at 1
	Stacktrace:
	 [1] promote_shape
	   @ .\indices.jl:178 [inlined]
	 [2] promote_shape
	   @ .\indices.jl:169 [inlined]
	 [3] +(A::Vector{Int64}, Bs::Vector{Int64})
	   @ Base .\arraymath.jl:14
	 [4] top-level scope
	   @ REPL[85]:1	

*Broadcasting?* We'll return to that :ref:`later <broadcasting>`.

**Transposition:**

.. sourcecode:: pycon

    julia> using LinearAlgebra
	julia> a = triu(ones(3, 3), 1)   # see ?triu
    julia> a = triu(ones(3, 3), 1)
	3×3 Matrix{Float64}:
	 0.0  1.0  1.0
	 0.0  0.0  1.0
	 0.0  0.0  0.0
	julia> a'
	3×3 adjoint(::Matrix{Float64}) with eltype Float64:
	 0.0  0.0  0.0
	 1.0  0.0  0.0
	 1.0  1.0  0.0



.. note:: **Linear algebra**

    The sub-module :mod:`LinearAlgebra` implements basic linear algebra, such as
    solving linear systems, singular value decomposition, etc. 
	
.. topic:: Exercise other operations
   :class: green

    * Look at the help for ``isapprox``. When might this be useful?
    * Look at the help for ``triu`` and ``tril``.


Basic reductions
----------------

Computing sums
..............

.. sourcecode:: pycon

    julia> x = [1, 2, 3, 4]
    julia> sum(x)
    10
    

.. image:: images/reductions.png
   :align: right

Sum by rows and by columns:

.. sourcecode:: pycon

    julia> x = [1, 1; 2, 2]]    
    julia> sum(A)
    6    
	julia> sum(A,dims=1)
	1×2 Matrix{Int64}:
	 3  3

	julia> sum(A,dims=2)
	2×1 Matrix{Int64}:
	 2
	 4
	julia> sum(A[1,:])
	2
	julia> sum(A[:,1])
	3

.. tip::

  Same idea in higher dimensions:

  .. sourcecode:: pycon

    julia> rng = np.random.default_rng(27446968)
    julia> x = rng.random((2, 2, 2))
    julia> x.sum(axis=2)[0, 1]
    0.73415...
    julia> x[0, 1, :].sum()
    0.73415...

Other reductions
................

--- works the same way (and take ``axis=``)

**Extrema:**

.. sourcecode:: pycon

  julia> x = [1, 3, 2]
  julia> x.min()
  1
  julia> x.max()
  3

  julia> x.argmin()  # index of minimum
  0
  julia> x.argmax()  # index of maximum
  1

**Logical operations:**

.. sourcecode:: pycon

  julia> np.all([True, True, False])
  False
  julia> np.any([True, True, False])
  True

.. note::

   Can be used for array comparisons:

   .. sourcecode:: pycon

      julia> a = np.zeros((100, 100))
      julia> np.any(a != 0)
      False
      julia> np.all(a == a)
      True

      julia> a = [1, 2, 3, 2]
      julia> b = [2, 2, 3, 2]
      julia> c = [6, 4, 4, 5]
      julia> ((a <= b) & (b <= c)).all()
      True

**Statistics:**

.. sourcecode:: pycon

  julia> x = [1, 2, 3, 1]
  julia> y = [[1, 2, 3], [5, 6, 1]]
  julia> x.mean()
  1.75
  julia> np.median(x)
  1.5
  julia> np.median(y, axis=-1) # last axis
  array([2.,  5.])

  julia> x.std()          # full population standard dev.
  0.82915619758884995


... and many more (best to learn as you go).

.. topic:: **Exercise: Reductions**
   :class: green

    * Given there is a ``sum``, what other function might you expect to see?
    * What is the difference between ``sum`` and ``cumsum``?


.. topic:: Worked Example: diffusion using a random walk algorithm

  .. image:: random_walk.png
     :align: center

  .. tip::

    Let us consider a simple 1D random walk process: at each time step a
    walker jumps right or left with equal probability.

    We are interested in finding the typical distance from the origin of a
    random walker after ``t`` left or right jumps? We are going to
    simulate many "walkers" to find this law, and we are going to do so
    using array computing tricks: we are going to create a 2D array with
    the "stories" (each walker has a story) in one direction, and the
    time in the other:

  .. only:: latex

    .. image:: random_walk_schema.png
        :align: center

  .. only:: html

    .. image:: random_walk_schema.png
        :align: center
        :width: 100%

  .. sourcecode:: pycon

   julia> n_stories = 1000 # number of walkers
   julia> t_max = 200      # time during which we follow the walker

  We randomly choose all the steps 1 or -1 of the walk:

  .. sourcecode:: pycon

   julia> t = np.arange(t_max)
   julia> rng = np.random.default_rng()
   julia> steps = 2 * rng.integers(0, 1 + 1, (n_stories, t_max)) - 1 # +1 because the high value is exclusive
   julia> np.unique(steps) # Verification: all steps are 1 or -1
   array([-1,  1])

  We build the walks by summing steps along the time:

  .. sourcecode:: pycon

   julia> positions = np.cumsum(steps, axis=1) # axis = 1: dimension of time
   julia> sq_distance = positions**2

  We get the mean in the axis of the stories:

  .. sourcecode:: pycon

   julia> mean_sq_distance = np.mean(sq_distance, axis=0)

  Plot the results:

  .. sourcecode:: pycon

   julia> plt.figure(figsize=(4, 3))
   <Figure size ... with 0 Axes>
   julia> plt.plot(t, np.sqrt(mean_sq_distance), 'g.', t, np.sqrt(t), 'y-')
   [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
   julia> plt.xlabel(r"$t$")
   Text(...'$t$')
   julia> plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")
   Text(...'$\\sqrt{\\langle (\\delta x)^2 \\rangle}$')
   julia> plt.tight_layout() # provide sufficient space for labels

  .. image:: auto_examples/images/sphx_glr_plot_randomwalk_001.png
     :width: 50%
     :target: auto_examples/plot_randomwalk.html
     :align: center

  We find a well-known result in physics: the RMS distance grows as the
  square root of the time!


.. arithmetic: sum/prod/mean/std

.. extrema: min/max

.. logical: all/any

.. the axis argument

.. EXE: verify if all elements in an array are equal to 1
.. EXE: verify if any elements in an array are equal to 1
.. EXE: load data with loadtxt from a file, and compute its basic statistics

.. CHA: implement mean and std using only sum()

.. _broadcasting:

Broadcasting
------------

* Basic operations on ``numpy`` arrays (addition, etc.) are elementwise

* This works on arrays of the same size.

    | **Nevertheless**, It's also possible to do operations on arrays of different
    | sizes if *NumPy* can transform these arrays so that they all have
    | the same size: this conversion is called **broadcasting**.

The image below gives an example of broadcasting:

.. only:: latex

    .. image:: images/numpy_broadcasting.png
        :align: center

.. only:: html

    .. image:: images/numpy_broadcasting.png
        :align: center
        :width: 100%

Let's verify:

.. sourcecode:: pycon

    julia> a = np.tile(np.arange(0, 40, 10), (3, 1)).T
    julia> a
    array([[ 0,  0,  0],
           [10, 10, 10],
           [20, 20, 20],
           [30, 30, 30]])
    julia> b = [0, 1, 2]
    julia> a + b
    array([[ 0,  1,  2],
           [10, 11, 12],
           [20, 21, 22],
           [30, 31, 32]])

We have already used broadcasting without knowing it!:

.. sourcecode:: pycon

    julia> a = np.ones((4, 5))
    julia> a[0] = 2  # we assign an array of dimension 0 to an array of dimension 1
    julia> a
    array([[2.,  2.,  2.,  2.,  2.],
           [1.,  1.,  1.,  1.,  1.],
           [1.,  1.,  1.,  1.,  1.],
           [1.,  1.,  1.,  1.,  1.]])

A useful trick:

.. sourcecode:: pycon

    julia> a = np.arange(0, 40, 10)
    julia> a.shape
    (4,)
    julia> a = a[:, np.newaxis]  # adds a new axis -> 2D array
    julia> a.shape
    (4, 1)
    julia> a
    array([[ 0],
           [10],
           [20],
           [30]])
    julia> a + b
    array([[ 0,  1,  2],
           [10, 11, 12],
           [20, 21, 22],
           [30, 31, 32]])


.. tip::

    Broadcasting seems a bit magical, but it is actually quite natural to
    use it when we want to solve a problem whose output data is an array
    with more dimensions than input data.

.. topic:: Worked Example: Broadcasting
   :class: green

   Let's construct an array of distances (in miles) between cities of
   Route 66: Chicago, Springfield, Saint-Louis, Tulsa, Oklahoma City,
   Amarillo, Santa Fe, Albuquerque, Flagstaff and Los Angeles.

   .. sourcecode:: pycon

       julia> mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544,
       ...        1913, 2448])
       julia> distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
       julia> distance_array
       array([[   0,  198,  303,  736,  871, 1175, 1475, 1544, 1913, 2448],
              [ 198,    0,  105,  538,  673,  977, 1277, 1346, 1715, 2250],
              [ 303,  105,    0,  433,  568,  872, 1172, 1241, 1610, 2145],
              [ 736,  538,  433,    0,  135,  439,  739,  808, 1177, 1712],
              [ 871,  673,  568,  135,    0,  304,  604,  673, 1042, 1577],
              [1175,  977,  872,  439,  304,    0,  300,  369,  738, 1273],
              [1475, 1277, 1172,  739,  604,  300,    0,   69,  438,  973],
              [1544, 1346, 1241,  808,  673,  369,   69,    0,  369,  904],
              [1913, 1715, 1610, 1177, 1042,  738,  438,  369,    0,  535],
              [2448, 2250, 2145, 1712, 1577, 1273,  973,  904,  535,    0]])


   .. image:: images/route66.png
      :align: center
      :scale: 60

A lot of grid-based or network-based problems can also use
broadcasting. For instance, if we want to compute the distance from
the origin of points on a 5x5 grid, we can do

.. sourcecode:: pycon

    julia> x, y = np.arange(5), np.arange(5)[:, np.newaxis]
    julia> distance = np.sqrt(x ** 2 + y ** 2)
    julia> distance
    array([[0.        ,  1.        ,  2.        ,  3.        ,  4.        ],
           [1.        ,  1.41421356,  2.23606798,  3.16227766,  4.12310563],
           [2.        ,  2.23606798,  2.82842712,  3.60555128,  4.47213595],
           [3.        ,  3.16227766,  3.60555128,  4.24264069,  5.        ],
           [4.        ,  4.12310563,  4.47213595,  5.        ,  5.65685425]])

Or in color:

.. sourcecode:: pycon

    julia> plt.pcolor(distance)
    <matplotlib.collections.PolyQuadMesh object at ...>
    julia> plt.colorbar()
    <matplotlib.colorbar.Colorbar object at ...>

.. image:: auto_examples/images/sphx_glr_plot_distances_001.png
   :width: 50%
   :target: auto_examples/plot_distances.html
   :align: center


**Remark** : the :func:`numpy.ogrid` function allows to directly create vectors x
and y of the previous example, with two "significant dimensions":

.. sourcecode:: pycon

    julia> x, y = np.ogrid[0:5, 0:5]
    julia> x, y
    (array([[0],
           [1],
           [2],
           [3],
           [4]]), array([[0, 1, 2, 3, 4]]))
    julia> x.shape, y.shape
    ((5, 1), (1, 5))
    julia> distance = np.sqrt(x ** 2 + y ** 2)

.. tip::

  So, ``np.ogrid`` is very useful as soon as we have to handle
  computations on a grid. On the other hand, ``np.mgrid`` directly
  provides matrices full of indices for cases where we can't (or don't
  want to) benefit from broadcasting:

  .. sourcecode:: pycon

    julia> x, y = np.mgrid[0:4, 0:4]
    julia> x
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [2, 2, 2, 2],
           [3, 3, 3, 3]])
    julia> y
    array([[0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3]])

.. rules

.. some usage examples: scalars, 1-d matrix products

.. newaxis

.. EXE: add 1-d array to a scalar
.. EXE: add 1-d array to a 2-d array
.. EXE: multiply matrix from the right with a diagonal array
.. CHA: constructing grids -- meshgrid using only newaxis

.. seealso::

   :ref:`broadcasting_advanced`: discussion of broadcasting in
   the :ref:`advanced_numpy` chapter.


Array shape manipulation
------------------------

Flattening
..........

.. sourcecode:: pycon

    julia> a = [[1, 2, 3], [4, 5, 6]]
    julia> a.ravel()
    array([1, 2, 3, 4, 5, 6])
    julia> a.T
    array([[1, 4],
           [2, 5],
           [3, 6]])
    julia> a.T.ravel()
    array([1, 4, 2, 5, 3, 6])

Higher dimensions: last dimensions ravel out "first".

Reshaping
.........

The inverse operation to flattening:

.. sourcecode:: pycon

    julia> a.shape
    (2, 3)
    julia> b = a.ravel()
    julia> b = b.reshape((2, 3))
    julia> b
    array([[1, 2, 3],
           [4, 5, 6]])

Or,

.. sourcecode:: pycon

    julia> a.reshape((2, -1))    # unspecified (-1) value is inferred
    array([[1, 2, 3],
           [4, 5, 6]])

.. warning::

   ``ndarray.reshape`` **may** return a view (cf ``help(np.reshape)``)),
   or copy

.. tip::

   .. sourcecode:: pycon

     julia> b[0, 0] = 99
     julia> a
     array([[99,  2,  3],
            [ 4,  5,  6]])

   Beware: reshape may also return a copy!:

   .. sourcecode:: pycon

     julia> a = np.zeros((3, 2))
     julia> b = a.T.reshape(3*2)
     julia> b[0] = 9
     julia> a
     array([[0.,  0.],
            [0.,  0.],
            [0.,  0.]])

   To understand this you need to learn more about the memory layout of a NumPy array.

Adding a dimension
..................

Indexing with the ``np.newaxis`` object allows us to add an axis to an array
(you have seen this already above in the broadcasting section):

.. sourcecode:: pycon

    julia> z = [1, 2, 3]
    julia> z
    array([1, 2, 3])

    julia> z[:, np.newaxis]
    array([[1],
           [2],
           [3]])

    julia> z[np.newaxis, :]
    array([[1, 2, 3]])



Dimension shuffling
...................

.. sourcecode:: pycon

    julia> a = np.arange(4*3*2).reshape(4, 3, 2)
    julia> a.shape
    (4, 3, 2)
    julia> a[0, 2, 1]
    5
    julia> b = a.transpose(1, 2, 0)
    julia> b.shape
    (3, 2, 4)
    julia> b[2, 1, 0]
    5

Also creates a view:

.. sourcecode:: pycon

    julia> b[2, 1, 0] = -1
    julia> a[0, 2, 1]
    -1

Resizing
........

Size of an array can be changed with ``ndarray.resize``:

.. sourcecode:: pycon

    julia> a = np.arange(4)
    julia> a.resize((8,))
    julia> a
    array([0, 1, 2, 3, 0, 0, 0, 0])

However, it must not be referred to somewhere else:

.. sourcecode:: pycon

    julia> b = a
    julia> a.resize((4,))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: cannot resize an array that references or is referenced
    by another array in this way.
    Use the np.resize function or refcheck=False

.. seealso: ``help(np.tensordot)``

.. resizing: how to do it, and *when* is it possible (not always!)

.. reshaping (demo using an image?)

.. dimension shuffling

.. when to use: some pre-made algorithm (e.g. in Fortran) accepts only
   1-D data, but you'd like to vectorize it

.. EXE: load data incrementally from a file, by appending to a resizing array
.. EXE: vectorize a pre-made routine that only accepts 1-D data
.. EXE: manipulating matrix direct product spaces back and forth (give an example from physics -- spin index and orbital indices)
.. EXE: shuffling dimensions when writing a general vectorized function
.. CHA: the mathematical 'vec' operation

.. topic:: **Exercise: Shape manipulations**
   :class: green

   * Look at the docstring for ``reshape``, especially the notes section which
     has some more information about copies and views.
   * Use ``flatten`` as an alternative to ``ravel``. What is the difference?
     (Hint: check which one returns a view and which a copy)
   * Experiment with ``transpose`` for dimension shuffling.

Sorting data
------------

Sorting along an axis:

.. sourcecode:: pycon

    julia> a = [[4, 3, 5], [1, 2, 1]]
    julia> b = np.sort(a, axis=1)
    julia> b
    array([[3, 4, 5],
           [1, 1, 2]])

.. note:: Sorts each row separately!

In-place sort:

.. sourcecode:: pycon

    julia> a.sort(axis=1)
    julia> a
    array([[3, 4, 5],
           [1, 1, 2]])

Sorting with fancy indexing:

.. sourcecode:: pycon

    julia> a = [4, 3, 1, 2]
    julia> j = np.argsort(a)
    julia> j
    array([2, 3, 1, 0])
    julia> a[j]
    array([1, 2, 3, 4])

Finding minima and maxima:

.. sourcecode:: pycon

    julia> a = [4, 3, 1, 2]
    julia> j_max = np.argmax(a)
    julia> j_min = np.argmin(a)
    julia> j_max, j_min
    (0, 2)


.. XXX: need a frame for summaries

    * Arithmetic etc. are elementwise operations
    * Basic linear algebra, ``@``
    * Reductions: ``sum(axis=1)``, ``std()``, ``all()``, ``any()``
    * Broadcasting: ``a = np.arange(4); a[:,np.newaxis] + a[np.newaxis,:]``
    * Shape manipulation: ``a.ravel()``, ``a.reshape(2, 2)``
    * Fancy indexing: ``a[a > 3]``, ``a[[2, 3]]``
    * Sorting data: ``.sort()``, ``np.sort``, ``np.argsort``, ``np.argmax``

.. topic:: **Exercise: Sorting**
   :class: green

    * Try both in-place and out-of-place sorting.
    * Try creating arrays with different dtypes and sorting them.
    * Use ``all`` or ``array_equal`` to check the results.
    * Look at ``np.random.shuffle`` for a way to create sortable input quicker.
    * Combine ``ravel``, ``sort`` and ``reshape``.
    * Look at the ``axis`` keyword for ``sort`` and rewrite the previous
      exercise.

Summary
-------

**What do you need to know to get started?**

* Know how to create arrays : ``array``, ``arange``, ``ones``,
  ``zeros``.

* Know the shape of the array with ``array.shape``, then use slicing
  to obtain different views of the array: ``array[::2]``,
  etc. Adjust the shape of the array using ``reshape`` or flatten it
  with ``ravel``.

* Obtain a subset of the elements of an array and/or modify their values
  with masks

  .. sourcecode:: pycon

     julia> a[a < 0] = 0

* Know miscellaneous operations on arrays, such as finding the mean or max
  (``array.max()``, ``array.mean()``). No need to retain everything, but
  have the reflex to search in the documentation (online docs,
  ``help()``, ``lookfor()``)!!

* For advanced use: master the indexing with arrays of integers, as well as
  broadcasting. Know more NumPy functions to handle various array
  operations.

.. topic:: **Quick read**

   If you want to do a first quick pass through the Scientific Python Lectures
   to learn the ecosystem, you can directly skip to the next chapter:
   :ref:`matplotlib`.

   The remainder of this chapter is not necessary to follow the rest of
   the intro part. But be sure to come back and finish this chapter, as
   well as to do some more :ref:`exercices <numpy_exercises>`.
