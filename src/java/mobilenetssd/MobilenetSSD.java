package com.example.syj.mobilenetssd;

import android.graphics.Bitmap;

public class MobilenetSSD {
    static {
        System.loadLibrary("native-lib");
    }
    //    public static native int add(int i,int j);
//    public static native String stringFromJNI();
//
    public native boolean Init(byte[] param, byte[] bin);

    public native String Detect(Bitmap bitmap, boolean use_gpu);
}
