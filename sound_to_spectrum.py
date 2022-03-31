import scipy.io.wavfile
import openl3

def to_embedding(file_name):
	# print(file_name)
	Fs, x = scipy.io.wavfile.read(file_name)
	x = x.astype("float")
	emb, ts = openl3.get_audio_embedding(x, Fs, content_type="env",
                               input_repr="mel128", embedding_size=512)
	
	return emb[:,None,None,:]



