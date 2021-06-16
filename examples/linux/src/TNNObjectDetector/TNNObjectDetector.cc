// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "object_detector_ssd.h"
#include "macro.h"
#include "utils/utils.h"
#include "tnn_sdk_sample.h"

#include "../flags.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

int main(int argc, char **argv) {
    std::cout << "===>Object Detecting Beginning" << std::endl;
    if (!ParseAndCheckCommandLine(argc, argv)) {
        ShowUsage(argv[0]);
        return -1;
    }

    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    std::cout << "FLAGS_p.c_str():" << FLAGS_p.c_str() << std::endl;
//    if (proto_content.length() > 0) {
//        std::cout << "proto_content:" << proto_content << std::endl;
//    }else{
//        std::cout << "proto_content is null" << std::endl;
//    }

    auto model_content = fdLoadFile(FLAGS_m.c_str());
    std::cout << "FLAGS_m.c_str():" << FLAGS_m.c_str() << std::endl;
//    if (model_content.length() > 0) {
//        std::cout << "model_content:" << model_content << std::endl;
//    }else{
//        std::cout << "model_content is null" << std::endl;
//    }
//    std::cout << "model_content:" << model_content << std::endl;

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();// option是个地址
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsTensorRT;
        #elif _OPENVINO_
            option->compute_units = TNN_NS::TNNComputeUnitsOpenvino;
        #endif
    }

//    std::cout << "option->proto_content:" << option->proto_content << std::endl;
    // compute_units是0 答：TNNComputeUnitsCPU = 0
//    std::cout << "option->compute_units:" << option->compute_units << std::endl;
    char img_buff[256];
    // img_buff和input_imgfn为空
//    std::cout << "img_buff" << img_buff << std::endl;
    char* input_imgfn = img_buff;
//    std::cout << "input_imgfn:" << input_imgfn << std::endl;
    strncpy(input_imgfn, FLAGS_i.c_str(), 256);

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    std::vector<int> nchw = {1, 3, image_height, image_width};

    if (!data) {
        fprintf(stderr, "Object-Detector open file %s failed.\n", input_imgfn);
    }

    auto predictor = std::make_shared<TNN_NS::ObjectDetectorSSD>();
    // 问题出在这？
    // 答：option有问题


    // 或者问题出在option的地址越界之类的问题
    // 问题来了：怎么判断是否内存越界？
    // 问题出在这？Init()这里 TNN_NS::ObjectDetectorSSD的Init有问题
//    std::cout << "------ Now before Init ------" << std::endl;
//    std::cout << "option before Init:" << option << std::endl;
    auto status = predictor->Init(option);
//    LOGE("instance.net init failed %d", (int)status);
//    std::cout << "status:" << status << std::endl;
    // 问题就是出在这
    if (status != TNN_NS::TNN_OK) {
        LOGE("instance.net init failed %d", (int)status);
        std::cout << "Predictor Initing failed, please check the option parameters" << std::endl;
    }

    std::shared_ptr<TNN_NS::TNNSDKOutput> sdk_output = nullptr;

    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
    auto resize_mat = predictor->ProcessSDKInputMat(image_mat, "data_input");
    CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNN_NS::TNNSDKInput>(resize_mat), sdk_output));

    CHECK_TNN_STATUS(predictor->ProcessSDKOutput(sdk_output));
    std::vector<TNN_NS::ObjectInfo> object_list;
    if (sdk_output && dynamic_cast<TNN_NS::ObjectDetectorSSDOutput *>(sdk_output.get())) {
        auto obj_output = dynamic_cast<TNN_NS::ObjectDetectorSSDOutput *>(sdk_output.get());
        object_list = obj_output->object_list;
    }

    const int image_orig_height = int(image_height);
    const int image_orig_width  = int(image_width);
    const auto& target_dims     = predictor->GetInputShape();
    const int target_height     = target_dims[2];
    const int target_width      = target_dims[3];
    float scale_x               = image_orig_width  / (float)target_width;
    float scale_y               = image_orig_height / (float)target_height;

    uint8_t *ifm_buf = new uint8_t[image_orig_width*image_orig_height*4];
    for (int i = 0; i < image_orig_height * image_orig_width; i++) {
        ifm_buf[i * 4] = data[i * 3];
        ifm_buf[i * 4 + 1] = data[i * 3 + 1];
        ifm_buf[i * 4 + 2] = data[i * 3 + 2];
        ifm_buf[i * 4 + 3] = 255;
    }
    for (int i = 0; i < object_list.size(); i++) {
        auto object = object_list[i];
        TNN_NS::Rectangle((void*)ifm_buf, image_orig_height, image_orig_width, object.x1, object.y1,
                           object.x2, object.y2, scale_x, scale_y);
    }

    char buff[256];
    sprintf(buff, "%s.png", "object-detector_predictions");
    int success = stbi_write_bmp(buff, image_orig_width, image_orig_height, 4, ifm_buf);
    if (!success) return -1;

    fprintf(stdout, "Object-Detector Done.\nNumber of objects: %d\n", int(object_list.size()));
    fprintf(stdout, "Save result image:%s\n", buff);
    delete [] ifm_buf;
    free(data);

    return 0;
}