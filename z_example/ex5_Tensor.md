## Tensor indexing (work in progress)
### indexing is more convinient then slice
### warning , still doesn't support broadcasting
```sh
python run.py --delete False --opt 5 
# run numpy to check the values is correct 
python main.py --opt 5  
```


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

