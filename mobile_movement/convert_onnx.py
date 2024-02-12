from movement_prediction_model import *
import torch
import os
import onnx
import onnxruntime
import numpy as np
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType



root_path = '/home/rgu/Documents/MARIO_mobile'

# CNN1D model 
data_path = os.path.join(root_path, 'magnitude_training', 'CNN_merge','best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
model = MovPredictCNN1D(input_size=23, hidden_size=64, output_size=5).to(device)
model.load_state_dict(torch.load(data_path))
# torch.save(model.state_dict(), os.path.join(root_path,'model.pt'))
model.eval()
input_magnitude = torch.randn(1, 23).to(device)
torch_out = model(input_magnitude)
torch.onnx.export(model, input_magnitude, os.path.join(root_path, 'best_model.onnx'), input_names=['input_magnitude'], output_names=['output_magnitude'], verbose=True)

onnx_model = onnx.load(os.path.join(root_path, 'best_model.onnx'))
# onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(os.path.join(root_path, 'best_model.onnx'), providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_magnitude)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print(f'PyTorch output: {to_numpy(torch_out)}, \nONNX Runtime output: {ort_outs[0]}')
# x = ort_outs[0]
# print(x[0][0])
# print(x[0][1])
# print(x[0][2])
# print(x[0][3])
# print(x[0][4])


# SKLearn random forest model   
data_path = os.path.join(root_path, 'random_forest_model.pkl')
model = joblib.load(data_path)
initial_type = [("float_input", FloatTensorType([1, 23]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open(os.path.join(root_path,'random_forest_model.onnx'), "wb") as f:
    f.write(onx.SerializeToString())
ort_session = onnxruntime.InferenceSession(os.path.join(root_path, 'random_forest_model.onnx'), providers=["CPUExecutionProvider"])
X_test = np.random.rand(1, 23)
input_name = ort_session.get_inputs()[0].name
label_name = ort_session.get_outputs()[0].name
ort_outs = ort_session.run([label_name], {input_name: X_test.astype(np.float32)})
print(f'input: {X_test}, \noutput: {ort_outs[0]}')
