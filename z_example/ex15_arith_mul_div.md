## example 15 arithmetic   
- [v] same shape indexing , `(a,b,c)[:,:,:2] = (a,b,2)`
- [v] scalar broadcasting , `(a,b,c)[:,:,:2] = 2; `
- [x] tensor broadcasting , `(a,b,c)[:,:,:2] = (a,1,2)`

- todo make some syntax sugar (a+b instead of a.add(b))
```sh
python run.py --delete False --opt 15
```

```cpp
Tensor<float> t4 = t2.mul(t3);
Tensor<float> t5 = t4.div1(); //t5=1/t4;
t5 = t5.div(10);  //t5=t5/(10)
```