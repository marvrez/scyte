#ifndef LAYER_H
#define LAYER_H

#include "mat.h"

#include <vector>
#include <string>

class Layer {
public:
    Layer() = default;
    virtual ~Layer() = 0;

    virtual void forward(const std::vector<Mat>& bottom, const std::vector<Mat>& top) = 0;
    virtual void backward(const std::vector<Mat*>& top, const std::vector<bool>& propagate_down, const std::vector<Mat*>& bottom) = 0;
    virtual const char* get_type() const = 0;

    // layer name
    std::string name;
    // indices which this layer needs as input
    std::vector<int> bottoms;
    // indices which this layer produces as output
    std::vector<int> tops;
};

#endif
