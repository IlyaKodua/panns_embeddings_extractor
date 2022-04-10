import glob
import pickle
import os
import time
from utils_for_read import*


def to_embedding(file_name):

	emb = audio_tagging(model_type="ResNet38", checkpoint_path="ResNet38_mAP=0.434.pth", audio_path=file_name)
	return emb





data_dir = "/home/liya/research/dcase_2022_data/"
dir_out = "out"

os.mkdir(dir_out)

list_files = glob.glob(data_dir + '**/*.wav', recursive=True)

persent = 0
timer = time.time()
print("Start!!!")
for i, filename in enumerate(list_files):

	cycl_perent = int((i+1)/len(list_files)*100)

	if(cycl_perent != persent):
		persent = cycl_perent
		print(persent, " %")
		print(time.time() - timer, " s")
		timer = time.time()

		
	data = to_embedding(filename)

	file_name_split = filename.split("/")

	wav_name = file_name_split[-1]

	name = file_name_split[-1].split(".")[0]
	with open(dir_out + "/" + name +".pkl", 'wb') as f:
		pickle.dump(data, f)
