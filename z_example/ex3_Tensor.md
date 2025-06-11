#### Ndim Tensor (work in progress)
```sh
python run.py --opt 3 --delete False
```
- for a tensor (dimision > 2)  
- currently we only support print , indexing, slice, reshape 
- tensor (matrix) multiplication and other functions (zeros,randn)...may be added in the future 
- The tensor implementation is based on [pynorch](https://github.com/lucasdelimanogueira/PyNorch)  

### Tensor init  
to create a tensor, we need to create a vector (data)   
and the shape of the data  
This can be done by   
```cpp
vector<float> zz2 = {0.1,0.3,0.2,-1.1};
vector<int> shape = {2,2};
Tensor<float> v1 {zz2,shape};
```
or 
```cpp
vector<float> zz2 = {0.1,0.3,0.2,-1.1};
Tensor<float> v1 {zz2,{2,2}};
```

### reshape
```cpp
vector<int> new_shape = {2,-1};
Tensor<float> newv1 = v1.reshape(new_shape);
//or 
newv1 = v1.reshape({1,4});
```
### indexing
- read by index
```py
#in python (numpy) we can do  
print(A[:,1:4])
```
in numcpp this can be done by slice  
```cpp
Tensor<float> vout = v1.slice(/*dim*/1,/*start*/1,/*end*/4);
```
- todo (some more fancy indexing A[2:3,:,1:4]) 

- set by index 
```py
#in python (numpy) we can do  
A[:,1] = 3
```
in numcpp, we need to first create a input shape, 
and put it into the index we want to replace   
note that we need to make sure that the update index shape   
are the same as input tensor shape  
```cpp
vector<float> zz2 = {0.1,0.3,0.2,-1.1,2.3,6.1};
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

```cpp
//convert Tensor to matrix (Ten0 dim must == 2)
mat<float> mm = Ten0.toMat();
//convert Tensor to vector (Ten1 dim must == 1)
vector<float> vv = Ten1.toVec();
```