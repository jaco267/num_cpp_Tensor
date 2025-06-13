## Ndim Tensor init (zeros, ones arange,  reshape, toVec, toMat)
```sh
python run.py --opt 3 --delete False
```
- for a tensor (dimision > 2)  
- currently we only support print (info()) , indexing, slice, reshape 
- tensor (matrix) multiplication and other functions (randn,kron...) may be added in the future 
- some implementation (print tensor, strides) is based on [pynorch](https://github.com/lucasdelimanogueira/PyNorch)  
- the fancy indexing : v1.index({Slice(2,4),1,None}) is similar to [libtorch](https://docs.pytorch.org/cppdocs/notes/tensor_indexing.html), but currently we don't support broadcasting and tensor indices (for example v1.index({v2, 1}, where v2 is a vector[1,3,5] is not support yet..))
- Tensor can be init by vector and shape, a Tensor can also be convert to vector and matrix by toVec(), toMat();
### Tensor init with vector
We can create a tensor using vector (data) and the shape of the data  
This can be done by   
```cpp
vector<float> zz2 = {0.1,0.3,
                     0.2,-1.1};
vector<int> shape = {2,2};
Tensor<float> v1 {zz2,shape};
// or Tensor<float> v1 {zz2,{2,2}}; //directly
v1.info();
```

### Tensor init with zeros,arange...
```cpp
Tensor<float> T0 = ones<float> (/*shape*/{2,2});
//or Tensor<float> T0 = zeros<float> (/*shape*/{2,2});
T0.info(); 
T0 = arange<float>(-1,6); 
T0.info(); 
```

#### reshape
```cpp
vector<int> new_shape = {2,-1};
Tensor<float> newv1 = T0.reshape(new_shape);
//or newv1 = v1.reshape({1,6});
```
#### indexing with slice
- read by index  (A[:,0] in python(numpy))
```cpp
Tensor<float> vout = v1.slice(/*dim*/1,/*start*/1,/*end*/4);
//v1[:,1:4] #in python (numpy)
```

- set by index (A[1,:]=3 in python)
  - to set by index, we need to first create a input Tensor, 
and put it into the index we want to replace, the update index shape   
should be the same as input tensor shape  
  - todo broadcasting...
```cpp
vector<float> zz2 = {0.1,0.3,0.2,
                     -1.1,2.3,6.1};
Tensor<float> v1 {zz2,{2,3}};
/*
v1 = 0.1 0.3 0.2  
    -1.1 2.3 6.1
*/
vector<float> p_val_vec = {1,2,3,4}; 
Tensor<float> p_val {p_val_vec,{2,2}};
v1.slice_put(/*dim*/1,1,3, p_val);
//in python this is :  v1[:,1:3] = p_val  
// note that v1[:,1:3].shape == p_val.shape
```  

#### convert Tensor to vec or matrix  
we can convert the tensor back to vector or matrix 
```cpp
//convert Tensor to matrix (Ten0 dim must == 2)
mat<float> mm = Ten0.toMat();
//convert Tensor to vector (Ten1 dim must == 1)
vector<float> vv = Ten1.toVec();
```