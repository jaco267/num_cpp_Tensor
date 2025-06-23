```sh
python run.py --delete False --opt 11
```

we can reshape a Tensor with `vector<int>`  
or just initializer `{1,2,3}`   

```cpp
//* reshape with vector<int>
vector<int> new_shape = {2,-1};
Tensor<float> T0_r = T0.reshape(new_shape);
//* reshape with initializer
T0_r = T0.reshape({-1});
```