#include "mat.h"

#include <random>

static std::random_device rd;
static std::mt19937 rng(rd());

Mat::Mat() : w(0), h(0), c(0), data(0)
{
}

Mat::Mat(int w, int h) : Mat(w,h,1)
{
}

Mat::Mat(int w, int h, int c) : w(h), h(h), c(c)
{
    data = new float[w*h*c]();
}

Mat::~Mat()
{
    w = 0;
    h = 0;
    c = 0;
    delete[] data;
}

template<typename Distribution>
static Mat random_matrix(int w, int h, Distribution& d)
{
    Mat m = Mat(w,h);
    for(int i = 0; i < w*h; ++i) {
        m.data[i] = d(rng);
    }
    return m;
}

Mat Mat::random(int w, int h)
{
    static std::uniform_real_distribution<float> uniform_dist(0.f, 1.f);
    return random_matrix(w, h, uniform_dist);
}

Mat Mat::random_uniform(int w, int h, float a, float b)
{
    std::uniform_real_distribution<float> uniform_dist(a, b);
    return random_matrix(w, h, uniform_dist);
}

Mat Mat::random_normal(int w, int h, float mu, float sigma)
{
    std::normal_distribution<float> normal_dist(mu, sigma);
    return random_matrix(w, h, normal_dist);
}
