# maria
Our most basic tools for reading and using Merian science catalogs. The only dependencies of this package are:
* numpy
* pandas
* astropy

# Scope: what functions should be included in `maria`?
`maria` is meant to be a central repository for the most basic catalog manipulation tools in Merian. This repository is _not_ meant 
to host functions for scientific analysis or pixel-level operations. **No** additional dependencies should be added to this 
repository in order to maintain accessibility (that means you, `spherical_geometry`).
