# import torch
# # ��dynamic��������Ȼ��TSMΪ��
# input_names = ['images']
# output_names = ['output']
# x = torch.randn(1, 48, 64, 64)
# torch.onnx.export(model, x, 'TSM.onnx', input_names=input_names, output_names=output_names,
# verbose=False, opset_version=13,
# do_constant_folding=True,training=torch.onnx.TrainingMode.EVAL)
#
# # dynamic����
# input_names = ['images']
# output_names = ['output']
# x = torch.randn(1, 48, 64, 64)
# # �����0��ʾ��һ��ά��Ϊ��̬��batch��ʾ�����ά���������
# dynamic_axes_0 = {
#     'input': {0: 'batch'},
#     'output': {0: 'batch'}
# }
# torch.onnx.export(model, x, 'TSM.onnx', input_names=input_names, output_names=output_names,
# verbose=False, opset_version=13,
# do_constant_folding=True,training=torch.onnx.TrainingMode.EVAL,
# dynamic_axes=dynamic_axes_0)