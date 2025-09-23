#include "nc_ten/ten_print.h"
namespace nc{
void print_nc_Slice_idx(nc_Slice_Index index){
  std::visit(PrintVisitor{}, index);
}
void print_nc_Slice_idx(const vector<nc_Slice_Index> & index_list){
  for(int i=0; i<(int)index_list.size(); i++){
    print_nc_Slice_idx(index_list[i]);
    cout<<",";  
  }
  cout<<endl;
}
}