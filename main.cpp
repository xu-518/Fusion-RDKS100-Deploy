#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "dnn/hb_dnn.h"
#include "hb_sys.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // ==========================================
    // 0. 全局参数设置
    // ==========================================
    string model_path = "/home/sunrise/Xs/fusion/fusion_s1002.hbm";
    string ir_dir = "/home/sunrise/Xs/fusion/ir_images/";
    string vis_dir = "/home/sunrise/Xs/fusion/vis_images/";
    string out_dir = "/home/sunrise/Xs/fusion/fused_output/";

    int height = 256;
    int width = 256;
    int image_area = height * width;

    // ==========================================
    // 1. 获取图片路径列表
    // ==========================================
    vector<cv::String> ir_paths;
    glob(ir_dir + "*.jpg", ir_paths, false);

    if (ir_paths.empty()) {
        cerr << "报错：未能找到红外图片！" << endl;
        return -1;
    }

    // ==========================================
    // 2. 初始化地平线 BPU 模型
    // ==========================================
    hbDNNPackedHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    if (hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1) != 0) {
        cerr << "模型加载失败" << endl;
        return -1;
    }

    const char** model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

    hbDNNHandle_t dnn_handle;
    if (hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]) != 0) {
        cerr << "获取模型句柄失败" << endl;
        hbDNNRelease(packed_dnn_handle);
        return -1;
    }

    int input_count = 0, output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);

    cout << "模型输入节点数: " << input_count << ", 输出节点数: " << output_count << endl;

    // ==========================================
    // 3. 准备输入输出 Tensor 内存（修正 tensorType）
    // ==========================================
    cout << "\n--- 结构体大小检查 ---" << endl;
    cout << "sizeof(hbDNNTensor)=" << sizeof(hbDNNTensor) << endl;
    cout << "sizeof(hbSysMem)=" << sizeof(hbSysMem) << endl;
    cout << "----------------------\n" << endl;
    vector<hbDNNTensor> input_tensors(input_count);
    for (int i = 0; i < input_count; ++i) {

        hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i);
                cout << "[Input " << i << " 原始属性]" << endl;
        cout << "  tensorType=" << input_tensors[i].properties.tensorType << endl;
        cout << "  alignedByteSize=" << input_tensors[i].properties.alignedByteSize << endl;
        cout << "  validShape.numDimensions=" << input_tensors[i].properties.validShape.numDimensions << endl;

        cout << "Input " << i << " 原始 tensorType: " << input_tensors[i].properties.tensorType
             << ", alignedByteSize: " << input_tensors[i].properties.alignedByteSize << endl;

        if (input_tensors[i].properties.alignedByteSize == image_area * sizeof(float) &&
            input_tensors[i].properties.tensorType == 0) {
            cout << "检测到输入实际应为浮点类型，将 tensorType 从 0 改为 7" << endl;
            input_tensors[i].properties.tensorType = 7;  // HB_DNN_TENSOR_TYPE_F32
        }

        int mem_size = input_tensors[i].properties.alignedByteSize;
        if (mem_size == 0) {
            cerr << "Warning: Input " << i << " alignedByteSize is 0, using default size "
                 << image_area * sizeof(float) << endl;
            mem_size = image_area * sizeof(float);
        }
        hbUCPMallocCached(&input_tensors[i].sysMem, mem_size, 0);
        cout << "[Input " << i << " 内存分配后]" << endl;
        cout << "  virAddr=" << input_tensors[i].sysMem.virAddr << endl;
        cout << "  phyAddr=" << input_tensors[i].sysMem.phyAddr << "\n" << endl;
    }

    vector<hbDNNTensor> output_tensors(output_count);
    for (int i = 0; i < output_count; ++i) {
        hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i);
        cout << "[Output " << i << " 原始属性]" << endl;
        cout << "  tensorType=" << output_tensors[i].properties.tensorType << endl;
        cout << "  alignedByteSize=" << output_tensors[i].properties.alignedByteSize << endl;

        if (output_tensors[i].properties.alignedByteSize == image_area * sizeof(float) &&
            output_tensors[i].properties.tensorType == 0) {
            cout << "检测到输出实际应为浮点类型，将 tensorType 从 0 改为 7" << endl;
            output_tensors[i].properties.tensorType = 7;
        }

        int mem_size = output_tensors[i].properties.alignedByteSize;
        if (mem_size == 0) {
            mem_size = image_area * sizeof(float);
        }
        hbUCPMallocCached(&output_tensors[i].sysMem, mem_size, 0);
        cout << "[Output " << i << " 内存分配后]" << endl;
        cout << "  virAddr=" << output_tensors[i].sysMem.virAddr << endl;
        cout << "  phyAddr=" << output_tensors[i].sysMem.phyAddr << "\n" << endl;
    }

    // ==========================================
    // 4. 核心批处理循环
    // ==========================================
    for (size_t i = 0; i < ir_paths.size(); ++i) {
        string current_ir_path = ir_paths[i];
        size_t last_slash = current_ir_path.find_last_of("/\\");
        string ir_filename = current_ir_path.substr(last_slash + 1);

        string vis_filename = ir_filename;
        size_t pos = vis_filename.find("IR");
        if (pos != string::npos) {
            vis_filename.replace(pos, 2, "VIS");
        }

        string current_vis_path = vis_dir + vis_filename;
        string current_out_path = out_dir + "fused_" + ir_filename;

        Mat ir_mat = imread(current_ir_path, IMREAD_GRAYSCALE);
        Mat vis_color = imread(current_vis_path, IMREAD_COLOR);

        if (ir_mat.empty() || vis_color.empty()) {
            cerr << "跳过缺失的图片对：" << ir_filename << endl;
            continue;
        }

        vector<Mat> bgr_channels;
        split(vis_color, bgr_channels);
        Mat vis_mat = bgr_channels[2];

        resize(ir_mat, ir_mat, Size(width, height), 0, 0, INTER_LINEAR);
        resize(vis_mat, vis_mat, Size(width, height), 0, 0, INTER_LINEAR);

        Mat ir_float, vis_float;
        ir_mat.convertTo(ir_float, CV_32FC1, 1.0 / 255.0);
        vis_mat.convertTo(vis_float, CV_32FC1, 1.0 / 255.0);
        memcpy(input_tensors[0].sysMem.virAddr, ir_float.data, image_area * sizeof(float));
        memcpy(input_tensors[1].sysMem.virAddr, vis_float.data, image_area * sizeof(float));

        hbUCPMemFlush(&input_tensors[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        hbUCPMemFlush(&input_tensors[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);

        // ==========================================
        // 执行推理 (UCP 方式)
        // ==========================================
        hbUCPTaskHandle_t task_handle = nullptr;

        const int ABI_TENSOR_SIZE = 176;

        // 1. 打包输入 Tensor 数组
        alignas(8) char packed_inputs[ABI_TENSOR_SIZE * 10]; 
        for (int k = 0; k < input_count; ++k) {
          
            memcpy(packed_inputs + k * ABI_TENSOR_SIZE, &input_tensors[k], ABI_TENSOR_SIZE);
        }

        // 2. 打包输出 Tensor 数组 (虽然你只有1个输出，但为了严谨一起打包)
        alignas(8) char packed_outputs[ABI_TENSOR_SIZE * 10];
        for (int k = 0; k < output_count; ++k) {
            memcpy(packed_outputs + k * ABI_TENSOR_SIZE, &output_tensors[k], ABI_TENSOR_SIZE);
        }

        // 3. 调用推理 API
        int ret = hbDNNInferV2(&task_handle, 
                               reinterpret_cast<hbDNNTensor*>(packed_outputs), 
                               reinterpret_cast<const hbDNNTensor*>(packed_inputs), 
                               dnn_handle);

        if (ret != 0) {
            cerr << "hbDNNInferV2 推理失败，错误码: " << ret << endl;
            continue;
        }

        // 4. 提交任务并等待完成
        hbUCPSchedParam ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
        
        hbUCPSubmitTask(task_handle, &ctrl_param);
        hbUCPWaitTaskDone(task_handle, 0);
        hbUCPReleaseTask(task_handle);


        hbUCPMemFlush(&output_tensors[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        float* output_data = reinterpret_cast<float*>(output_tensors[0].sysMem.virAddr);

        Mat fused_mat(height, width, CV_8UC1);
        for (int p = 0; p < image_area; ++p) {
            float val = output_data[p] * 255.0f;
            fused_mat.data[p] = saturate_cast<uchar>(val);
        }

        imwrite(current_out_path, fused_mat);
        cout << "成功处理: " << ir_filename << " -> " << current_out_path << endl;
    }

    // ==========================================
    // 5. 释放资源
    // ==========================================
    for (auto& tensor : input_tensors) {
        hbUCPFree(&tensor.sysMem);
    }
    for (auto& tensor : output_tensors) {
        hbUCPFree(&tensor.sysMem);
    }
    hbDNNRelease(packed_dnn_handle);

    return 0;
}