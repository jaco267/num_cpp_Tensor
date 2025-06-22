#include "tensor.h"
#include "numcpp.h"
//* explicit init template
namespace nc{
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<long>;

template class Tensor<unsigned long>;
}