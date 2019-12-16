#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
    int c, h, w;
    float* data;
} image;

image make_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image make_image_from_hwc_bytes(int w, int h, int c, unsigned char* bytes);
image copy_image(image m);
void free_image(image* m);

// load functions
image load_image(const char* filename, int num_channels);
image load_image_from_memory(const unsigned char* buffer, int buf_len, int num_channels);
image load_image_rgb(const char* filename);
image load_image_grayscale(const char* filename);

// save functions
int save_image_png(image m, const char* filename);
int save_image_jpg(image m, const char* filename, int quality);

image get_channel(image m, int c);

// colorspace functions
image rgb_to_grayscale(image m);
image grayscale_to_rgb(image m, float r, float g, float b);

// image operations
void fill_image(image* m, float s);
void clamp_image(image* m);
void translate_image(image* m, float s);
void scale_image(image* m, float s);
void normalize_image(image* m);
void transpose_image(image* m);
void flip_image(image* m);

unsigned char* get_image_data_hwc(image m);

#endif
