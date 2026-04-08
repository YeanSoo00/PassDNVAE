import os
import json, torch, argparse, time
from model import PassDNVAE

def idx2str(sample, i2w, pad, sos, eos):
    out=[]
    for t in sample:
        t=int(t)
        if t==eos: break
        if t in (pad,sos): continue
        out.append(i2w.get(str(t),""))
    return "".join(out)


def main(args):
    #vocab=json.load(open(f"{args.data_dir}/{args.vocab_file}","r"))
    with open(f"{args.data_dir}/{args.vocab_file}", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        w2i,i2w=vocab['w2i'], vocab['i2w']

    pad,sos,eos=w2i['<pad>'],w2i['<sos>'],w2i['<eos>']
    model=PassDNVAE(
        vocab_size=len(w2i),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        sos_idx=sos,eos_idx=eos,pad_idx=pad,
        max_sequence_length=args.max_sequence_length
    )
    state=torch.load(args.model_path,map_location="cuda")
    model.load_state_dict(state)
    model=model.cuda().eval()

    start=time.time()
    total_generated=0

    with open(args.output_file,'w',encoding='utf-8') as f:

        for _ in range(args.repeat):  
            samples,_=model.inference(n=args.num_samples)  
            samples=samples.tolist()

            for s in samples:
                f.write( idx2str(s,i2w,pad,sos,eos)+"\n" )
                total_generated+=1

    print(f" Finish! Total password generation {total_generated:,}")
    print(f" During: {time.time()-start:.2f} sec")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='./data')
    parser.add_argument('--vocab_file',default='rockyou-vocab.json')  #4iq-4class8-20%-vocab

    parser.add_argument('--model_path',default='./rockyou0/best_model.pt')     
    parser.add_argument('--output_file',default='./PassDNVAE_10M.txt')
    parser.add_argument('--num_samples',type=int,default=10000) 
    parser.add_argument('--repeat',type=int,default=100000)       

    parser.add_argument('--embedding_size',type=int,default=300)
    parser.add_argument('--hidden_size',type=int,default=256)
    parser.add_argument('--latent_size',type=int,default=64)
    parser.add_argument('--word_dropout',type=float,default=0.0)
    parser.add_argument('--embedding_dropout',type=float,default=0.1)
    parser.add_argument('--max_sequence_length',type=int,default=12)

    args=parser.parse_args()
    main(args)
