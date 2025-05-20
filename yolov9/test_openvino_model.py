from openvino.runtime import Core
import numpy as np

# 模型路徑（請依實際修改）
MODEL_XML = "/home/user/yolo-openpose-app/yolov9/yolov9_openvino/last.xml"
INPUT_SHAPE = (1, 3, 640, 640)

# 產生假影像輸入
dummy_input = np.random.rand(*INPUT_SHAPE).astype(np.float32)

# 初始化 OpenVINO Core
core = Core()
model = core.read_model(model=MODEL_XML)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 執行推論
print("🔎 執行模型推論...")
results = compiled_model([dummy_input])[output_layer]

# 顯示輸出形狀
print("✅ 模型推論成功！")
print(f"輸入形狀：{dummy_input.shape}")
print(f"輸出形狀：{results.shape}")

# 顯示前幾筆資料（方便檢查）
print("前 5 筆輸出資料（前 6 維度）：")
print(results[:5, :6])
