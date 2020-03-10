#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <stdio.h>

#include "mobilenetssd.id.h"

#include <vector>
// ncnn

#include <sys/time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "WaterdemoNcnn", "%.2fms   %s", elasped, comment);
}

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Mat mobilenetssd_param;
static ncnn::Mat mobilenetssd_bin;
//static std::vector<std::string> squeezenet_words;
static ncnn::Net mobilenetssd;

std::vector<Object> objects;

static void draw_objects(cv::Mat& image, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"};

//    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);




        __android_log_print(ANDROID_LOG_DEBUG, "waterNcnn", "detect: %d = %.5f at %.2f %.2f %.2f x %.2f  %s",
                            obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, text);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

//    cv::imshow("image", image);
//    cv::waitKey(0);
}

extern "C" {



JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL Java_com_example_syj_mobilenetssd_MobilenetSSD_Init(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;



    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;


    mobilenetssd.opt = opt;

    // init param
    {
        int len = env->GetArrayLength(param);
        mobilenetssd_param.create(len, (size_t)1u);
        env->GetByteArrayRegion(param, 0, len, (jbyte*)mobilenetssd_param);
        int ret = mobilenetssd.load_param((const unsigned char*)mobilenetssd_param);
        __android_log_print(ANDROID_LOG_DEBUG, "waterNcnn", "load_param %d %d", ret, len);
    }

    // init bin
    {
        int len = env->GetArrayLength(bin);
        mobilenetssd_bin.create(len, (size_t)1u);
        env->GetByteArrayRegion(bin, 0, len, (jbyte*)mobilenetssd_bin);
        int ret = mobilenetssd.load_model((const unsigned char*)mobilenetssd_bin);
        __android_log_print(ANDROID_LOG_DEBUG, "waterNcnn", "load_model %d %d", ret, len);
    }

    // init words
//    {
//        int len = env->GetArrayLength(words);
//        std::string words_buffer;
//        words_buffer.resize(len);
//        env->GetByteArrayRegion(words, 0, len, (jbyte*)words_buffer.data());
//        squeezenet_words = split_string(words_buffer, "\n");
//    }

    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_example_syj_mobilenetssd_MobilenetSSD_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    bench_start();

    // ncnn from bitmap
    ncnn::Mat in;
    ncnn::Mat i2;

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int target_size = 300;
    int img_w = info.width;
    int img_h = info.height;
//        if (width != 300 || height != 300)
//            return NULL;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

//    void* indata;
//    AndroidBitmap_lockPixels(env, bitmap, &indata);
//
//    in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2BGR, target_size, target_size);
//    i2 = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2BGR, img_w, img_h);


    i2 = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);
    in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, target_size, target_size);

    cv::Mat out_cvmat(img_h, img_w,CV_8UC3,cv::Scalar(0,1,2));
    i2.to_pixels(out_cvmat.data, ncnn::Mat::PIXEL_BGR); //8U3,0-255

//    AndroidBitmap_unlockPixels(env, bitmap);


    // squeezenet


    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenetssd.create_extractor();

    ex.set_vulkan_compute(use_gpu);

    ex.input(mobilenetssd_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(mobilenetssd_param_id::BLOB_detection_out, out);

    objects.clear();
    std::string result_str = "success";

    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    draw_objects(out_cvmat, objects);


    out = ncnn::Mat::from_pixels(out_cvmat.data, ncnn::Mat::PIXEL_BGR, out_cvmat.cols, out_cvmat.rows);
    out.to_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);





    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());

    bench_end("detect");

    return result;
}

}