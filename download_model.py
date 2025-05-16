import gdown

FILE_ID_ONNX = "1RQkUdY20r-VMkhUID_gmiZip6LvZuAco"
FILE_ID_REGNET = "1HMajjuiZjH4I93Pn9EsijmUBScMSKeLy"
MODEL_PATH_REGNET = "regnet94valacc.pth"
MODEL_PATH_ONNX = "u2net_human_seg.onnx"


gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID_REGNET}", MODEL_PATH_REGNET, quiet=False)
gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID_ONNX}", MODEL_PATH_ONNX, quiet=False)