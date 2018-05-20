import os

from six import text_type
import myg2p

word = "experimental"
try:
    if not issubclass(type(word), text_type):
        word = text_type(word, encoding="utf-8", errors="replace")
except EOFError:
    pass

model = myg2p.startpoint(model_dir = 'D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress', interactive=True)
model.interactive()


#g2p_model.interactive()
#g2p_model.decode(output_file_path=model_flags.output)

#g2p_seq2seq.app.FLAGS
#model_params = g2p_seq2seq.params.Params('D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress', 'D:\Projects\g2p-models\test-list.txt', g2p_seq2seq.app.FLAGS)
#model = g2p_seq2seq.g2p.G2PModel(model_params)
#result = model.decode('D:\Projects\g2p-models\test-output.txt')
#print(result)
#command = 'g2p-seq2seq --decode D:\Projects\g2p-models\test-list.txt --model_dir D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress --output D:\Projects\g2p-models\test-output.txt'
#cmd = command.encode()
#subprocess.run(cmd.decode())
#dec = os.path.abspath('D:\Projects\g2p-models\test-list.txt')
#mod = os.path.abspath('D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress')
#os.system('g2p-seq2seq', decode = dec, model_dir = mod)
#os.system('g2p-seq2seq --decode ' + dec + ' --model_dir ' + mod)
#os.system('g2p-seq2seq', decode = os.path('D:\Projects\g2p-models\test-list.txt'), model_dir = os.path('D:\Projects\g2p-models\g2p-seq2seq-model-6.2-cmudict-nostress'))
