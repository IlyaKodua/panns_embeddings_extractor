import glob
import pickle
import os

from utils_for_read import*

def to_embedding(file_name):

	emb = audio_tagging(model_type="ResNet38", checkpoint_path="ResNet38_mAP=0.434.pth", audio_path=file_name)
	return emb





data_dir = "data/"
dir_out = "out"

os.mkdir(dir_out)

list_files = glob.glob(data_dir + '**/*.wav', recursive=True)

for i, filename in enumerate(list_files):
	print(int((i+1)/len(list_files)*100), " %")
	arr_dict = dict()
		
	arr_dict["data"] = to_embedding(filename)

	file_name_split = filename.split("/")

	wav_name = file_name_split[-1]


	# arr_dict["filename"] = wav_name

	if 'anomaly' in file_name_split[-1] :
		arr_dict["labels"] = 1
	else:
		arr_dict["labels"] = 0
	name = file_name_split[-1].split(".")[0]
	with open(dir_out + "/" + file_name_split[-3] + "_" + file_name_split[-2] + "_" + name +".pkl", 'wb') as f:
		pickle.dump(arr_dict, f)
