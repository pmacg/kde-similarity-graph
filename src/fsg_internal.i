// %module(directors="1", threads="1") fsg_internal
%module(directors="1") fsg_internal

// Eigen / numpy stuff
%include <typemaps.i>
%include eigen.i
%include numpy.i

%{
    #include <Eigen/Core>
    #include "cpp/fsg.h"
    #include "cpp/kde.h"
    #include "cpp/utilities.h"
    #include "cpp/stag_lib/utility.h"
    #include "cpp/stag_lib/graph.h"
%}

// Deal with stag_int
typedef long long stag_int;

%init %{
    import_array();
%}

%eigen_typemaps(Eigen::VectorXd)
%eigen_typemaps(Eigen::MatrixXd)

%include <std_vector.i>
namespace std {
    // Create bindings for the std::vector types
    %template(vectori) vector<long long>;
    %template(vectord) vector<double>;
}

// Create the directors to allow overriding in python
%feature("director") KDESolver;

// Include the other fsg headers
%include "cpp/kde.h"
%include "cpp/fsg.h"
%include "cpp/utilities.h"

// Include the stag utilities to allow to convert between SprsMat and scipy
// sparse matrices.
%include "cpp/stag_lib/utility.h"

// Include a destructor for the sparse matrix type
class SprsMat {
public:
    ~SprsMat();
};
