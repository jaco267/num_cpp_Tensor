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
vector<int> shape = {4,4};
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

- set by index (todo) 
```py
#in python (numpy) we can do  
A[:,1] = 3
```  
- cpp version haven't implement yet....   
