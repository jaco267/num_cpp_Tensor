## example 13 : slice  
slice to read tensor         
slice_put to update tensor   
- [v] same shape indexing , `(a,b,c)[:,:,:2] = (a,b,2)`
- [x] scalar broadcasting , `(a,b,c)[:,:,:2] = 2; `
- [x] tensor broadcasting , `(a,b,c)[:,:,:2] = (a,1,2)`

- todo  add scalar broadcasting on slice (should be easy to add)
```sh
python run.py --delete False --opt 13
```