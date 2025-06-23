
### simple Tensor library in c++

A simple Tensor library for N dimension tensor  
, 2D matrix (ex. vector<vector<int>>)   
and 1D vector  (ex. vector<int>)    

### exmaples 
(for more example, see z_example/ folder) 
- there is `--opt 0~6 (vec_mat)` and `--opt 10~13 (tensor)`

####  run examples

```sh
mkdir build 
cd build
cmake ..
cmake --build . --config Release  
./main -o 1

#* if you are lazy, run.py is a script to compile and run the cpp files 
pip install pyrallis  #python command line library
#* run a tensor examples
python run.py  --delete False --opt 10
```

### Ndim Tensor 

- for a tensor (dimision > 2)  
- currently we only support print (info()) ,init (zeros,ones,arange..), indexing, reshape , add, minus, multiply, divide
- tensor (matrix) multiplication and other functions (randn,kron...) may be added in the future 
- some implementation (print tensor, strides) is based on [pynorch](https://github.com/lucasdelimanogueira/PyNorch)  
- the fancy indexing : v1.index({Slice(2,4),1,None}) is similar to [libtorch](https://docs.pytorch.org/cppdocs/notes/tensor_indexing.html), but currently we don't support broadcasting and tensor indices (for example v1.index({v2, 1}, where v2 is a vector[1,3,5] is not support yet..))
- Tensor can be init by vector and shape, a Tensor can also be convert to vector and matrix by toVec(), toMat();

### Tensor init with zeros,arange...
```sh
# opt 3,4,5 are tensor examples
python run.py --opt 3 --delete False
```

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

#### indexing 
```sh
python run.py --delete False --opt 5
```

- read by index  (A[:,0] in python(numpy))
- To set by index (A[1,:]=3 in numpy), we need to first create a input Tensor, 
and put it into the index we want to replace, the update index shape   
should be the same as input tensor shape  
  - todo broadcasting...
```cpp
using namespace nc::indexing; 
vector<float> zz2 = {0.1,0.3,0.2,
                         1, 2 , 3, 

                      -1.1,2.3,6.1,
                         4, 5 , 6, 

                      0.9, 0.8,-0.5,
                        7, 8 ,  8 , 

                      -3, -2,-9,
                      10, 11, 12 };
Tensor<float> v1 {zz2,{4,2,3}};
Tensor <float> out_v1 = v1.index({Slice(2,4),1,None});
//similar to #out_v1 = v1[2:4,1,:] in python
//* we haven't support broad casting , so when updating values the shape have to be the same  
Tensor<float> put0 = zeros<float>({1,2,2});
v1.index_put({2,Slice(None),Slice(1)},put0);
// similar to # v1[2,:,1:] = 0 in python
```


#### indexing with slice ( see z_example/...md and examples/...cpp)
```sh
python run.py --delete False --opt 4 
```
- read by index  (A[:,0] in python(numpy))
```cpp
Tensor<float> vout = v1.slice(/*dim*/1,/*start*/1,/*end*/4);
//v1[:,1:4] #in python (numpy)
```
- set by index (A[1,:]=3 in python)
  - update index shape should == input tensor shape  
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