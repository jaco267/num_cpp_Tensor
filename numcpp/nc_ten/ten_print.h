#pragma once
#include "numcpp.h"
#include "tensor.h"

namespace nc{
struct PrintVisitor {
    void operator()(const nc::indexing::Slice& s) const { std::cout << s; }
    void operator()(int i) const { std::cout << i; }
    void operator()(std::monostate) const { std::cout << "null"; }
};
void print_nc_Slice_idx(nc_Slice_Index index);
void print_nc_Slice_idx(const vector<nc_Slice_Index> & index);
}