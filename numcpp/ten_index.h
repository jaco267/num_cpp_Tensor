#pragma once
#include "numcpp.h"  
#include "tensor.h"
#include <variant>
//https://www.reddit.com/r/cpp_questions/comments/hu7cen/is_it_possible_to_have_an_empty_variant/

//* if you f12 torch::indexing::Slice you can see it uses optional
using nc_Index = std::variant<
  std::monostate, 
  int
>;  //monostate is None

namespace nc::indexing{
struct NoneType {
    constexpr operator std::monostate() const { return {}; }
};

inline constexpr NoneType None{};
class Slice{
public:
  Slice(){};
  Slice(nc_Index start=std::monostate{}, 
        nc_Index end=std::monostate{},
        nc_Index step=std::monostate{}){
    start_ = start;
    end_ = end; 
    step_ = step;
    //https://youtu.be/qCc_Vqg3hJk?t=339
    if (auto value = std::get_if<int> (&start)){
        int& v = *value; 
        start_end_step_vec.push_back(v);
        None_vec_.push_back(0);
    }else if (std::get_if<std::monostate>(&start)) {
        start_end_step_vec.push_back(0); //* default is 0
        None_vec_.push_back(1);
    }
    if (auto value = std::get_if<int> (&end)){
        int& v = *value; 
        start_end_step_vec.push_back(v);
        None_vec_.push_back(0);
    }else if (std::get_if<std::monostate>(&end)) {
        start_end_step_vec.push_back(-1);
        None_vec_.push_back(1);
    }
    if (auto value = std::get_if<int> (&step)){
        int& v = *value; 
        start_end_step_vec.push_back(v);
        None_vec_.push_back(0);
    }else if (std::get_if<std::monostate>(&step)) {
        start_end_step_vec.push_back(1);//* default step is 1
        None_vec_.push_back(1);
    }
    // cout<<"start"<<start<<endl;
  };

  // Define how Slice is printed
  friend std::ostream& operator<<(std::ostream& os, const Slice& slice) {
      os << "Slice(";
      print_vec(slice.start_end_step_vec,0);
      os << ")";
      return os;
  }
private:
  nc_Index start_;
  nc_Index end_;
  nc_Index step_;
public:
  vector<int> start_end_step_vec;  //-1 means that pos is none
  vector<int> None_vec_; //1-> None ,0->not None
};
}