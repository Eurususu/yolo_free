# import torch
# # 无dynamic导出，依然以TSM为例
# input_names = ['images']
# output_names = ['output']
# x = torch.randn(1, 48, 64, 64)
# torch.onnx.export(model, x, 'TSM.onnx', input_names=input_names, output_names=output_names,
# verbose=False, opset_version=13,
# do_constant_folding=True,training=torch.onnx.TrainingMode.EVAL)
#
# # dynamic导出
# input_names = ['images']
# output_names = ['output']
# x = torch.randn(1, 48, 64, 64)
# # 这里的0表示第一个维度为动态，batch表示对这个维度起的名字
# dynamic_axes_0 = {
#     'input': {0: 'batch'},
#     'output': {0: 'batch'}
# }
# torch.onnx.export(model, x, 'TSM.onnx', input_names=input_names, output_names=output_names,
# verbose=False, opset_version=13,
# do_constant_folding=True,training=torch.onnx.TrainingMode.EVAL,
# dynamic_axes=dynamic_axes_0)