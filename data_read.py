import glob
import pickle
import os

import scipy.io.wavfile
import openl3

def to_embedding(file_name):
	# print(file_name)
	Fs, x = scipy.io.wavfile.read(file_name)
	x = x.astype("float")
	emb, ts = openl3.get_audio_embedding(x, Fs, content_type="env",
                               input_repr="mel128", embedding_size=512)
	
	return emb[:,None,None,:]





data_dir = "/home/liya/research/sound_data/"
dir_out = "out"

os.mkdir(dir_out)

list_files = glob.glob(data_dir + '**/*.wav', recursive=True)

for i, filename in enumerate(list_files):
	print(int((i+1)/len(list_files)*100), " %")
	arr_dict = dict()
		
	arr_dict["data"] = to_embedding(filename)

	file_name_split = filename.split("/")

	if 'anomaly' in file_name_split[-1] :
		arr_dict["labels"] = 1
	else:
		arr_dict["labels"] = 0
	name = file_name_split[-1].split(".")[0]
	with open(dir_out + "/" + file_name_split[-3] + "_" + file_name_split[-2] + "_" + name +".pkl", 'wb') as f:
		pickle.dump(arr_dict, f)
