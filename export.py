import torch
import torch.nn as nn
from backbone.Gen_mamba3_1 import E_TransG

def export_to_onnx():
    print("===========================================")
    print("1. 实例化完整模型 E_TransG (原版结构)...")
    print("===========================================")
    device = "cpu" 
    
    # 实例化完全体模型
    model = E_TransG().to(device)
    
    pkl_path = "mamba3_1_FLIRtest2_3000.pkl"
    print(f"2. 加载原始权重: {pkl_path}")
    
    checkpoint = torch.load(pkl_path, map_location=device)
    
    model.load_state_dict(checkpoint["weight"])
    model.eval()
    
    def set_bn_eval(m):
        if isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()
            m.track_running_stats = False # 彻底禁掉训练逻辑

    model.apply(set_bn_eval)
    print("===========================================")
    print("3. 构造端侧静态虚拟输入 (256x256)...")
    print("===========================================")
    dummy_ir = torch.randn(1, 1, 256, 256).to(device)
    dummy_vis = torch.randn(1, 1, 256, 256).to(device)

    onnx_file_name = "fusion_s1002.onnx"
    print("===========================================")
    print(f"4. 呼叫地平线底层引擎，开始导出: {onnx_file_name}...")
    print("===========================================")

    torch.onnx.export(
        model, 
        (dummy_ir, dummy_vis),             
        onnx_file_name,
        export_params=True,
        opset_version=11,                 
        do_constant_folding=True,          
        input_names=['ir_image', 'vis_image'], 
        output_names=['fused_image']
    )
    print("===========================================")
    print("🎉 大功告成！D-Robotics 官方环境 ONNX 导出成功！")
    print("===========================================")

if __name__ == '__main__':
    export_to_onnx()