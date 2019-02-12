#ifndef MAT_H
#define MAT_H

class Mat {
public:
    Mat();
    Mat(int w, int h);
    Mat(int w, int h, int c);

    static Mat random(int w, int h);
    static Mat random_uniform(int w, int h, float a, float b);
    static Mat random_normal(int w, int h, float mu, float sigma);

    ~Mat();

    // dimensions
    int w;
    int h;
    int c;

    float* data;
};

#endif
