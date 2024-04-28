from tensorflow.python import pywrap_tensorflow
# 以下XXX为模型保存chechpoint文件的相对路径，如"model/bestParMTCR_gateBehavior.ckpt"
checkpoint_path = "/home/liuyue/PycharmProjects/EasyRec/experiments/ckpt/taobao"  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)      # 参数名
    print(reader.get_tensor(key)) 
