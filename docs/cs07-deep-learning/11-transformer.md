```python
# Based on this paper - https://arxiv.org/pdf/1706.03762.pdf
# Might want to move layer norm inside the residual block - https://arxiv.org/pdf/2002.04745.pdf
# Layer normalization - https://arxiv.org/pdf/1607.06450.pdf
# TODO: Investigate learning rate warmup - https://arxiv.org/abs/2002.04745
#!pip install torch datasets wandb
```

```python
import numpy as np
import torch
from torch import nn
import sys
import os
import math
sys.path.append(os.path.abspath("../../data"))
sys.path.append(os.path.abspath("../../nnets"))
from net_utils import get_module_list

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
SP_VOCAB_SIZE = 5000
TRAIN_SIZE = 5000
```

```python
from text_data import OpusBooksDataset
from padding import PaddingSampler, pad_collate
from torch.utils.data import DataLoader

train_wrapper = OpusBooksDataset(tokenizer_vocab=SP_VOCAB_SIZE, download_split_pct="5%")
train_dataset = train_wrapper.process_dataset()
train_dataset = train_dataset.train_test_split(test_size=0.1)

train_split = train_dataset["train"]
train = DataLoader(train_split, batch_size=BATCH_SIZE, sampler=PaddingSampler(train_split["en_lens"]), collate_fn=pad_collate)

valid_split = train_dataset["test"]
valid = DataLoader(valid_split, batch_size=BATCH_SIZE, sampler=PaddingSampler(valid_split["en_lens"]), collate_fn=pad_collate)
```

    Found cached dataset opus_books (/Users/vik/.cache/huggingface/datasets/opus_books/en-es/1.0.0/e8f950a4f32dc39b7f9088908216cd2d7e21ac35f893d04d39eb594746af2daf)
    


    Combining:   0%|          | 0/5 [00:00<?, ?ba/s]



    Tokenizing:   0%|          | 0/5 [00:00<?, ?ba/s]



    Filtering:   0%|          | 0/5 [00:00<?, ?ba/s]



    Filtering:   0%|          | 0/5 [00:00<?, ?ba/s]



    Splitting:   0%|          | 0/5 [00:00<?, ?ba/s]

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, input_units, attention_heads, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.input_units = input_units
        self.attention_heads = attention_heads
        self.head_units = int(input_units/attention_heads)
        self.mask = mask

        k = math.sqrt(1/self.input_units)
        self.in_proj_weight = nn.Parameter(torch.rand(3, input_units, self.attention_heads * self.head_units) * 2 * k - k)
        self.in_proj_bias = nn.Parameter(torch.rand(3, input_units) * 2 * k - k)

        self.out_proj_weight = nn.Parameter(torch.rand(self.attention_heads * self.head_units, input_units) * 2 * k - k)
        self.out_proj_bias = nn.Parameter(torch.rand(input_units) * 2 * k - k)

    def forward(self, queries, keys, values):
        # convert to 4d tensor with batch_size, attn_heads, seq_len, embedding_dim
        proj_queries = torch.einsum("...se, eo->...so", queries, self.in_proj_weight[0]) + self.in_proj_bias[0]
        proj_queries = proj_queries.view(queries.shape[0], queries.shape[1], self.attention_heads, self.head_units).swapaxes(1,2)

        proj_keys = torch.einsum("...se, eo->...so", keys, self.in_proj_weight[1]) + self.in_proj_bias[1]
        proj_keys = proj_keys.view(keys.shape[0], keys.shape[1], self.attention_heads, self.head_units).swapaxes(1,2)

        proj_values = torch.einsum("...se, eo->...so", values, self.in_proj_weight[2]) + self.in_proj_bias[2]
        proj_values = proj_values.view(values.shape[0], values.shape[1], self.attention_heads, self.head_units).swapaxes(1,2)

        attention = torch.einsum("baqh, bahk->baqk", proj_queries, torch.transpose(proj_keys, -1, -2)) / np.sqrt(proj_keys.shape[-1])
        if self.mask:
            # Prevent decoder queries from looking at tokens that come after
            # Do this by setting attention to negative infinity, so it is softmaxed to zero in the next step
            mask = torch.full((attention.shape[-2], attention.shape[-1]), -torch.inf, device=DEVICE)
            attention += torch.triu(mask, diagonal=1)

        # Softmax on last dimension
        # Sequence-wise softmax, so attention between one sequence and other sequences sums to 1
        attention = torch.softmax(attention, dim=-1)
        weighted_values = torch.einsum("baqk, bake->baqe", attention, proj_values)

        # Swap attention head and sequence axis, then reshape to batch, seq, embedding
        weighted_values = weighted_values.swapaxes(1,2).reshape(queries.shape[0], queries.shape[1], -1)
        weighted_values = torch.einsum("...se, eo->...so", weighted_values, self.out_proj_weight) + self.out_proj_bias
        return weighted_values
```

```python
class EncoderBlock(nn.Module):
    def __init__(self, input_units, attention_heads, hidden_units=2048, dropout_p=.1):
        super(EncoderBlock, self).__init__()
        self.input_units = input_units
        self.attention_heads = attention_heads
        self.hidden_units = hidden_units

        self.mha = MultiHeadAttention(self.input_units, self.attention_heads)
        self.dropouts = get_module_list(2, nn.Dropout, dropout_p)
        self.linear1 = nn.Linear(self.input_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, self.input_units)
        self.relu = nn.ReLU()
        self.lns = get_module_list(2, nn.LayerNorm, self.input_units)

    def forward(self, x):
        weighted_values = self.dropouts[0](self.mha(x, x, x))
        x = self.lns[0](x + weighted_values)

        reprojected = self.dropouts[1](self.linear2(self.relu(self.linear1(x))))
        x = self.lns[1](x + reprojected)
        return x
```

```python
class DecoderBlock(nn.Module):
    def __init__(self, input_units, attention_heads, hidden_units=2048, dropout_p=.1):
        super(DecoderBlock, self).__init__()
        self.input_units = input_units
        self.attention_heads = attention_heads
        self.hidden_units = hidden_units

        self.in_attn = MultiHeadAttention(self.input_units, self.attention_heads, mask=True)
        self.context_attn = MultiHeadAttention(self.input_units, self.attention_heads)
        self.dropouts = get_module_list(3, nn.Dropout, dropout_p)
        self.linear1 = nn.Linear(self.input_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, self.input_units)
        self.relu = nn.ReLU()
        self.lns = get_module_list(3, nn.LayerNorm, self.input_units)

    def forward(self, x, context):
        weighted_values = self.dropouts[0](self.in_attn(x, x, x))
        x = self.lns[0](x + weighted_values)

        decoder_values = self.dropouts[1](self.context_attn(x, context, context))
        x = self.lns[1](x + decoder_values)

        reprojected = self.dropouts[2](self.linear2(self.relu(self.linear1(x))))
        x = self.lns[2](x + reprojected)
        return x
```

```python
class Transformer(nn.Module):
    def __init__(self, input_units, hidden_units, attention_heads, padding_idx, max_len=256, blocks=1):
        super(Transformer, self).__init__()
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.attention_heads = attention_heads
        self.blocks = blocks

        self.output_embedding = nn.Linear(hidden_units, input_units)
        self.embedding = nn.Embedding(input_units, hidden_units, padding_idx=padding_idx)
        self.dropouts = get_module_list(2, nn.Dropout, .1)
        self.encoders = get_module_list(self.blocks, EncoderBlock, hidden_units, attention_heads)
        self.decoders = get_module_list(self.blocks, DecoderBlock, hidden_units, attention_heads)
        self.pos_encoding = self.encoding(max_len, self.hidden_units).to(DEVICE)


    def forward(self, x, y, enc_outputs=None):
        if enc_outputs is None:
            # 3D with batch, seq, embeddings
            # TODO: Tie input and output embedding weights
            enc_outputs = self.dropouts[0](self.embedding(x) + self.pos_encoding[:x.shape[1]])

            for i in range(self.blocks):
                enc_outputs = self.encoders[i](enc_outputs)

        dec_outputs = self.dropouts[1](self.embedding(y) + self.pos_encoding[:y.shape[1]])
        for i in range(self.blocks):
            dec_outputs = self.decoders[i](dec_outputs, enc_outputs)

        token_vectors = self.output_embedding(dec_outputs)
        return token_vectors, enc_outputs

    def encoding(self, seq_len, embed_len):
        encodings = torch.zeros((seq_len, embed_len))
        for i in range(seq_len):
            all = torch.exp(torch.arange(0, embed_len, 2) * (-math.log(10000.0) / embed_len))
            encodings[i, 0::2] = torch.sin(i * all)
            encodings[i, 1::2] = torch.cos(i * all)
        return encodings
```

```python
def generate(sequence, pred, target, wrapper):
    prompts = wrapper.decode_ids(sequence.cpu())
    texts = wrapper.decode_ids(torch.argmax(pred, dim=2).cpu())
    correct_texts = wrapper.decode_ids(target.cpu())

    displays = []
    for p, t, ct in zip(prompts, texts, correct_texts):
        displays.append(f"{p} | {ct} | {t}")
    return displays
```

```python
from tqdm.auto import tqdm
import wandb

wandb.init(project="transformer", notes="Baseline performance one block updated dataset", name="baseline-one-new-data")

# TODO: Profile and improve perf - https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
model = Transformer(SP_VOCAB_SIZE, 512, 8, blocks=1, padding_idx=train_wrapper.tokenizer.pad_token_id).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=train_wrapper.tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
wandb.watch(model, log_freq=100)
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    

    [34m[1mwandb[0m: Currently logged in as: [33mvikp[0m. Use [1m`wandb login --relogin`[0m to force relogin
    

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

Tracking run with wandb version 0.13.9

Run data is saved locally in <code>/Users/vik/Personal/nnets/notebooks/transformer/wandb/run-20230202_163626-80wu2uxl</code>

Syncing run <strong><a href="https://wandb.ai/vikp/transformer/runs/80wu2uxl" target="_blank">baseline-one-new-data</a></strong> to <a href="https://wandb.ai/vikp/transformer" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>

View project at <a href="https://wandb.ai/vikp/transformer" target="_blank">https://wandb.ai/vikp/transformer</a>

View run at <a href="https://wandb.ai/vikp/transformer/runs/80wu2uxl" target="_blank">https://wandb.ai/vikp/transformer/runs/80wu2uxl</a>

    []

```python
EPOCHS = 100
DISPLAY_BATCHES = 2
PRINT_VALID = True

for epoch in range(EPOCHS):
    # Run over the training examples
    train_loss = 0
    match_pct = 0
    for batch in tqdm(train):
        optimizer.zero_grad(set_to_none=True)
        sequence = batch["en_ids"].to(torch.long)
        target = batch["es_ids"].to(torch.long)
        # Setup target properly
        prev_target = torch.roll(target, 1, -1)
        prev_target[:,0] = train_wrapper.tokenizer.bos_token_id

        pred, _ = model(sequence.to(DEVICE), prev_target.to(DEVICE))

        # If you use a batch, need to reshape pred to be batch * sequence, embedding_len to be compatible
        # Similar reshape with target to be batch * sequence vector of class indices
        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), target.reshape(-1).to(DEVICE))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    with torch.no_grad():
        mean_loss = train_loss / len(train) / BATCH_SIZE
        wandb.log({"loss": mean_loss})
        print(f"Epoch {epoch} train loss: {mean_loss}")
        sents = generate(sequence, pred, target, train_wrapper)
        for sent in sents[:DISPLAY_BATCHES]:
            print(sent)

        if PRINT_VALID and epoch % 10 ==0:
            # Compute validation loss.  Unless you have a lot of training data, the validation loss won't decrease.
            valid_loss = 0
            # Deactivate dropout layers
            model.eval()
            for batch in tqdm(valid):
                # Inference token by tokens
                sequence = batch["en_ids"].to(torch.long)
                target = batch["es_ids"].to(torch.long)
                # Setup target properly
                prev_target = torch.roll(target, 1, -1)
                prev_target[:,0] = train_wrapper.tokenizer.bos_token_id
                outputs = prev_target[:,0].unsqueeze(1).to(DEVICE)
                enc_outputs = None
                # TODO: Investigate memory leak with valid generation
                for i in range(target.shape[1]):
                    pred, enc_outputs = model(sequence, outputs, enc_outputs=enc_outputs)
                    last_output = torch.argmax(pred, dim=2)
                    outputs = torch.cat((outputs, last_output[:,-1:]), dim=1)
                loss = loss_fn(pred.reshape(-1, pred.shape[-1]), target.view(-1).to(DEVICE))
                valid_loss += loss.item()
            mean_loss = valid_loss / len(valid) / BATCH_SIZE
            wandb.log({"valid_loss": mean_loss})
            print(f"Valid loss: {mean_loss}")
            sents = generate(sequence, pred, target, train_wrapper)
            for sent in sents[:DISPLAY_BATCHES]:
                print(sent)
            # Reactivate dropout
            model.train()
```

      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 0 train loss: 0.22905117582965207
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | - de de de de de de que de de de de de de de de que de de que de de de de que de de de de que de de que que de que de que la que de de de de que de de de de de
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | - de de que que de de de de de de que la de de de de que de de de de de de de de de que de que de de de de que de la de de de de de de de de de de de de de
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.21411480340692732
    ONE person I was sure would represent me as capable of any thing-- What I felt was dreadful!--My resolution was soon made, and at eight o'clock this morning I was in my carriage. | ¡Lo que sentí fue atroz! Rápidamente tomé una decisión, y hoy a las ocho de la mañana ya me encontraba en mi carruaje. | - de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de
    The son, a steady respectable young man, was amply provided for by the fortune of his mother, which had been large, and half of which devolved on him on his coming of age. | El hijo, un joven serio y respetable, tenía el futuro asegurado por la fortuna de su madre, que era cuantiosa, y de cuya mitad había entrado en posesión al cumplir su mayoría de edad. | - de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de que de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 1 train loss: 0.20931275750135447
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | S no que de que de de de de que que de de que de de de la de que de que de de de de de su de de que de de que que que de de y de que y de de de de de la de
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | - que que que que que que no que de a de la de de de de que de de de de de de que de de de en y de de de y que de de de de de de que de de de de de que de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 2 train loss: 0.2014536505395716
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | - se de que de no de de de que de que que de de de la de que de su de de no que que de de de de de que no se lo que a su de de su madre de su de de su de - de
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Llo de y y se de y de de de de de la y de su y de y y su y se de de de de de de hermana de de y a se de de de de la y a y - de de de de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 3 train loss: 0.19247973003944793
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Sras que que a su de que de de de la señora y de de su de se se en a el de de de de de la de que de de su las de de la de de la de de de le que de - de de
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | -n que que no se señora y que que y y y y señor de y no y que que que que que que que que que hermana que que que que de no. de que señora que y. que que y de. que de de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 4 train loss: 0.18392432355261468
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | - no es que de de que de y no que de de que de de de la que de que que de quea de y que de de que de de que hombreos de de y de se loo. de. de la de que
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | S meabao de de de la de su de de que más que de y de de de y su hermana que y sentimientos de dea y y no de señora y de mundo de de su madre y de de de. - y de de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 5 train loss: 0.17586514140878404
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Sn que que no se señorita y de que dea y y señor y y no que que queo y que y que que mi hermana que el que que y mea.. lo señorita que y que de lo me. gran. de de
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se habíaó de se embargo, de y y la de de la señora de y era de gran y y señora de de de de y de y la le y de de el y dea. de un hombre de el de de de se sentimientos su
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 6 train loss: 0.16812092065811157
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Slabaó que que no no no había había de no a su no de su y de de y su no se eraas a de de su madreo de y no se era de a. no habíaa. -....
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | E me estado que de de de lo de su de su que cosas que deaas de de de su hermanaía y sentimientos y dea y y no que señorita de de mundo y de su hermana y el de de. No...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 7 train loss: 0.16110607362412788
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No me estadoo de de de el de su de su que dos que deasas de de de su hermana y y modalesación sus ojos y y no que señorita que de mundo y de su madre y de y de. -...
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Son que me me me queasoo no es que de prontoado que una que una hombre de señor que Palmer que que es que es es es la señorita un de a que que dea. que........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 8 train loss: 0.154488951741875
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero me me que lo en Marianne enasoé en eno, y queo, que y y si lo que me me me que me lo que me coronelan en que si me lo que. lo me lo.é. coronel. Pero..
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había estado que que la Elinor queorado a Elinor que de de a su deancia de su de de se su modalesos de de a la que se se le de a su más de su madre de su a de su. D.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 9 train loss: 0.14847270050993214
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no seasó pero el en pero no que en enas eno,o la que de en enómo,o, y la otra más en de de la hombreos en y y no es lo había sido. en. la..
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Sa que sea se señoritaa, a laó y una de unaó de Elinor Elinor le eraa a a era una cierta para que se a su que se su. de su paraente a un su hermanaos.....
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 10 train loss: 0.1425440496244988
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se loó peroó embargo, pero; y unaó en la ciudadpo y era que cierta y y señoritaó con que deó de y laabareó que un a queó que un hombre de el queos que se ojos la
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Aor que loueó en no le que; en la yg; de, ya como una y tan otra vez que completo; y no en no lo sidor dea. tan comoé tan prontoas como la. D..
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.23187331524160174
    "To be sure," continued Lucy, after a few minutes silence on both sides, "his mother must provide for him sometime or other; but poor Edward is so cast down by it! | -Con toda seguridad -continuó Lucy tras unos minutos de silencio por ambas partes-, tarde o temprano su madre tendrá que proporcionarle medios de vida; ¡pero el pobre Edward se siente tan abatido con todo eso! | -Es que se lo ha sido una mujera-, es una gran cosa que lo ha sido tan grande como lo ha sido tan pronto pero se lo ha sido tan pronto pero se lo ha sido tan grande como lo ha sido de una gran ser feliz
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | -Es que es tan pronto pero es tan pronto pero es una gran ser un hombre; pero la señorita Steeleo a su hermana en un hombre; y una gran mejor que es un hombre tan grande que es un hombre. -No,. -No,
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 11 train loss: 0.13706103886489746
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Alabaaba que que no no nunca había habíaoado a su se de suó o, no su no había habíaal aaba vez su madreo de y no había había con a que había habíair. -....
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no seja, pero elo, pero no de en enend,j el la que de por enómo,o, y la otra más en de de la hombreos en de y no es lo d sido. en de las. de
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 12 train loss: 0.13187701303463478
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Noraspocoo a suar dere de como su casa yaba de su y un seó una un yrit yó y un yabalo a un dos y de su. a una uniente. vieron su de D..
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Ch! con el madreas con el la -dijo Marianne que estado con conado con los queos queos de la asunto que mi estado unao y el en gran en, en mi vida el es ser feliz... grande..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 13 train loss: 0.12671493651804985
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su madreado por completo es -dijo Elinor, que estado conadoado con los ojosososos de la asunto que es estado Do no no una persona en. en mi vidao es ser mias.. para grande..
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No hemos de tan de de lo de suar de la cosas de deo,as de de de su madreía y modalesación sus modaleso, y no la casa de de mundo y de su madre y la de deo. No...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 14 train loss: 0.12203892407479225
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con toda momento deendonaró el sentar a la deabaerar el corazón elar el señoritas de de unrendán el sentimientos de la con y por el a laionar el su madreo.cont deo..
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no seja, pero elo, pero enar en enoy quej el lao de por enabamo,o, de la otra más en de de la hombreo, en y y no es lo había sido en en de la. o
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 15 train loss: 0.11748620296840544
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su esposa con por completo gran -dijo Marianne que sido comprometadoado su los demásasosos en la asunto que esga hacero y en una gran en, en mi vida en es ser miierado con grande..
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Almitaba nunca que nunca no nunca había había salado a su hacer de suaba de, no no no le habíaal a de vez su afectoo de y no había había y a que había habíair en No....
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 16 train loss: 0.11318829752408065
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Noraspoco a a serarse dereedse el casaó como de una y un seó en unabarit yó en un yólo de su tres para en sus tiempo y su seiente. lecía de No
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Almitió, con que no no nunca le había produado a el hacer de laciones de, que su no le habíaaba que de de su afectoo de y no había habíaaba a que no habíair. P..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 17 train loss: 0.10880298235199669
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Sa que laero la lea, que quear de que de que que y Elinor que Elinor gustan que más fue una deion que era por que que y hombre siguiente de su aente a de su hermana.it.....
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Cunque con su hermano con por completo gran -dijo Elinor, que estado compromettenado que los queáosos que las plan que es estado unao que en que granción. en mi vida no es ser miier que en grandemente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 18 train loss: 0.10440029886055302
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se lo pero pero sin embargo, que; y unaó en la mása, se dije una miradao, se cualó enos deó de y en quealló en laidez queos en un hombre deía que que que se ojos su
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no seja, es elo, pero noa el haoy quej a la personas de en imaginarímo,o, de que otra más mejor de de que hombre hum no, y no ha lo hagacido en de una..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 19 train loss: 0.10056033033829231
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero meo que Marianne a Marianne enostaría en eno, si queic que y y me me que me me meirá tan que lo mundoanso, si me hace quee, yo me gustberé. mundoo, Pero..
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Non no que no he heco a hace;o, pero pero men pero pero no todo tiempo;mos de todo y un todo seguridad familia no un le y y como pronto como si cam como como como como si no una grana comoa.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 20 train loss: 0.09653956414043129
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con el asunto leendaina, para loar de la deiderar el tiempo dear el cosas y incluso unostor,; sentimientos de Edward; y por el a Edwardionar el el propioo.contento. M.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Por que loóó que nunca le que que a lalc que otraan dea que siquiera como que otra cosa que mí; pero que que que ha;r tanado en comoamos que deas como el. -..
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.24575082626607683
    "To be sure," continued Lucy, after a few minutes silence on both sides, "his mother must provide for him sometime or other; but poor Edward is so cast down by it! | -Con toda seguridad -continuó Lucy tras unos minutos de silencio por ambas partes-, tarde o temprano su madre tendrá que proporcionarle medios de vida; ¡pero el pobre Edward se siente tan abatido con todo eso! | -Aunque -respondió Elinor-, en la ciudad-, pero sí mismas un hombre bien lo que debe seriedad; pero o dos o dos días; pero o dos o dos o dos o dos o dos o dos días ni siquiera la mitad de Edward-
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | -Es que es; pero que es tan grande que una gran ternura; una mujer muy pronto la mayor signa que un hombre; en su hermana la ciudad. un hombre tan grande que le digo. -ya que la ciudad.o. -
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 21 train loss: 0.09296299062378995
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se loó pero sin embargo, palabra; y unaó en la fina, le llegó una deo, la buenaó enos varió y y luego quealló en laososos en un comienzo se menos violentos que la de Marianne.
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no seja, es elo, pero noar en esino quej a la personas de por permitirármelar de que de de de de de un hombre hum en, de no ha ha decescido en de ningúnado o
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 22 train loss: 0.08958848469056092
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que laero la señoraó de queó de que de que que y a que le dara a más fue una deión de era de que Elinor se tiempo de de sutente a de su hermana.it..
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Porió;ueó que nunca se alláía a mílg ninguna otramo ya que siquiera y ninguna otra cosa por mí; pero no se se ha yla.ado y esé tan prontoas como se.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 23 train loss: 0.08597672629085454
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Si, Elinor- que lo de alo de una una mujer de que queisiete de usted por una hombre de que y que de que lo siquiera se presentar a un momento a lo peorar a un hombreía, porosa para que
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero meo de Marianne a Marianne seostaría en comoo, si loorm; y y como me que nada me dirá que que el corazónanso, si me lo ho. yo me imaginoé el corazóno. Pero..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 24 train loss: 0.08269813129460657
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Non muy que no se personaco a hace;o, pero al deciros, pero si todo que no dicho de usted y de usted su m como el queos y como pronto como la señora como como como como si la..a comoa.
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se loó pero sin embargo, palabra; y leó en la puertaa, le tomó una grano, la tardeó enos varió de y laaalló en queían ya. que un comienzo apenas menos deos que la ojos Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 25 train loss: 0.07927049948023511
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas comoes como lo de susar de la más quenasas deencia de su propioillas sus parientesación sus sentimientosos; pero no la mejor opinión de mundo; de su bondad y comoibilidadz. No...
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero meo de Marianne a Marianne siosto, y, poro, si comoico; y como lo que nada me dirá que que el corazónanso, si me lo hae, yo me loberé. mundoo. Pero..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 26 train loss: 0.07601591765687063
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas dees como tú de susar de las más mínimas tendencias de su familiaagn y parientesaciones, sus deseos; pero tengo la mejor opinión del mundo; de su bondad y susatez.....
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y luegoándose en la cama, le tomó una miradao, se besó enos varias de y luego estalló en unaaszos que un comienzo apenas menos violentos que. de Marianne
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 27 train loss: 0.07288319012754924
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Almitió, entonces, que aunque no nunca había había informado sobre qué tipo de suciones tenían, a su no le suía con alguna de su afectoo y y no le eraaba su que se leir. No....
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Eston que no no me queencia. Y a ustediera o muy amo de que; un hombreit señor señor Ferrars, de un es que no ese la mitad esperarse dos o dos oes. o........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 28 train loss: 0.06969802775843577
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | No no estja, esquo, pero noa dist viendo tienej a otras personas de así soportármelo, de lo manera está mejorto de los tiempo hum natural, aunque no ha le ha sidocido oportunidad de mejorarlo.
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como lo de susar hasta las más mínimas tendencias de su magn sus inclinaciones, sus gustos; pero le la mejor opinión del mundo le de su bondad y susibilidadz. ¡..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 29 train loss: 0.06651918991626084
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Porir estadouró que nunca he alláía a mílarme nunca otraca nueva para siquiera se ninguna otra cosa que mí; pero ahora se se ha aplacado y seamos tan pocoas como ella. -..
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Nos no no no se heco a hace pensaro, si al menos, pero si mucho más he tenido de todo y si todas su familia por completoídas; y tan pronto como si sal, si casi como si fuera una.a eraa.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 30 train loss: 0.06344316495800173
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero noo de Marianne a Marianne noostada y, puedoo, si ningunaicida; y como me que nada le servirá que que el descanso, si me be mismoe, yo me beberé el compromiso. Pero..
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Con que no no me paciencia. Y aunque leiera que muy bieno de quele un momento del señor señor Ferrars, aunque un es que no ese la pena quearía dos o dos quees. o........
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.26646652155452305
    The son, a steady respectable young man, was amply provided for by the fortune of his mother, which had been large, and half of which devolved on him on his coming of age. | El hijo, un joven serio y respetable, tenía el futuro asegurado por la fortuna de su madre, que era cuantiosa, y de cuya mitad había entrado en posesión al cumplir su mayoría de edad. | La primera vez de Elinor un poco después de inmediato era su madre, por lo que le había sido la naturaleza de su madre, y su madre, que había sido informado con él mismo tiempo. lo que su madre, y su madre,.pu.pu.
    ONE person I was sure would represent me as capable of any thing-- What I felt was dreadful!--My resolution was soon made, and at eight o'clock this morning I was in my carriage. | ¡Lo que sentí fue atroz! Rápidamente tomé una decisión, y hoy a las ocho de la mañana ya me encontraba en mi carruaje. | Yo no he pensado que me estaba tan furada de mi madre,...... era imposible que se me fuego, a la llamó de esta noche y a la mañana siguiente y en mi vida había sido que se sentía. ¡Pobre me fue
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 31 train loss: 0.06064940918865916
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Noanpoco no a dejarse porreed por la vidaó como de una como le seó un un dealesvign por un lesadolo de un tres para en ese a y casi uniente. secía. D..
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Entonunque con su hermana, preocupación por nuestra relación -dijo Elinor- ha estado obviado todos los demás,áculos que la plan que pueda estado imaginar, persiste una marción que en mi opinión, no es ser tanachada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 32 train loss: 0.05751066589607047
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este plan leaticinar se loó a suciiderar el tiempo lear el que e los unostender los sentimientos de Edward; y, con el a Edwardionar el su propio méritcontento. T.
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a todo plan que ha hab elar, persiste una manción que, en mi opinión, no hay ser miierada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 33 train loss: 0.05476804101815472
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Aoros estadouró que nunca más volvería a mílarme ninguna otraca nueva para siquiera haría ninguna otra cosa que mí; pero ahora que se le apla aado y ahoraamos tan pocoas como ella. -..
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -é que Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir un un hombre de un y un algo que un un se loque a él amor que lo mismoaría en un hombreía, paraable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 34 train loss: 0.0521954202830985
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a las plan que ha estado imaginar, persiste una objeción, en mi opinión, no hay ser miachada tan fácilmente.
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de míar hasta que más mínimas tendencias de su mente, sus sentimientosaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. Est...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 35 train loss: 0.04966035145395375
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero yoo de Marianne a Marianne aostada y, espero, casi enormida; y como creo que nada le dirá el que el descanso, si me lo permite, yo me beberé el vino. Pero..
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este plan vendicinio se sentó a lasiderar el ánimo recordar el palabras e ins comprender los sentimientos de Edward; y, por supuesto, a losionar sobre los propio descontento..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 36 train loss: 0.047219293192029
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que una imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que un siquiera se acerque a ese amor que un transformaría en un compañero deseable para ella.
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Todospoco no a dejarse arerar por la observadora que de Lucy, que no estó en voz delavada en voz privándolo de las tres que, en Cleveland amigo y casi uniente, sepcía.pu..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 37 train loss: 0.04497625758605344
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurármelo, de un manera está falto de un buen gusto natural, aunque no tiene le ha ofrecido oportunidad de mejor conocimiento ser
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sup -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 38 train loss: 0.042863316979114115
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar la palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a Edwardionar sobre su propio descontento. L.
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 39 train loss: 0.04089093439329367
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había termin siempre que siempre más P paraente para su establecerse a su vezancia de su antes que sugar intencioneses paraos, por su que no le habíat a su escal de su madre, de lo a a su. D.
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; pero sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en laidezzos que un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 40 train loss: 0.0386951266146906
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los seráculos a la plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser miachada tan fácilmente.
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir de un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.2812895112567478
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | -Es tan grande que Willoughby es; pero que su hermana, entender lo creo que una cierta importancia para su hermana la voz: -¡Santo cielo! -Y la voz baja la felicidad una mujer-. Mique vida.. -
    From all danger of seeing Willoughby again, her mother considered her to be at least equally safe in town as in the country, since his acquaintance must now be dropped by all who called themselves her friends. | En cuanto al peligro de encontrarse de nuevo con Willoughby, su madre pensaba que Marianne estaba tan a salvo en la ciudad como en el campo, dado que nadie entre quienes se consideraban sus amigos lo admitiría ahora en su compañía. | Todos eran desconsuelta de Willoughby la madre de su madre y se vio obligada a partir en la ciudad; se vio obligada a pesar de su madre ahora último tiempo antes más allá de todos sus hijas.os amigas.as.a. No dej
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 41 train loss: 0.03668035209396636
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá que que que descanso, si me lo permite, yo me beberé el vino. Pero.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Con que yo no me paciencia. Y aunque cualquiera... muy conscio de hacerle un favor a pobre señor Ferrars, de unero que no vale la pena esperarlo dos o dos hijes. como.......
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 42 train loss: 0.03463785535377729
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una deión que dio en de que una día de de su diferente era de su madre.encia.....
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos de las plan que ha podido imaginar, persiste... objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 43 train loss: 0.032717278910583104
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus gustaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. Est..
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a algunas distancia de Norland antes que no sus actuales másos, por su que no se sentíarim a ella intenciones de su madre de irse a Devonshire. P
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 44 train loss: 0.031065125665445992
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de que diferente era de que hermana.ado....
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a mílarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos que amigas como están. No.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 45 train loss: 0.029388808143361436
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus hijes paraos, por lo que no se opuso a Devonshire escal de su madre de irse a Devonshire. P
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta c más mínimas tendencias de su mente, sus inclinación sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. Est..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 46 train loss: 0.02755691765432621
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de ningunaarlo.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. P.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 47 train loss: 0.026211428528579026
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida y y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. Pero..
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Noienpoco iba a dejarse eredrar por la observadora de de ellas que casi tardó en un clavada en tanto privándolo de las atenciones que, en especial amigo y casi todiente, se lascía. D..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 48 train loss: 0.024691096343729017
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. -No
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar el palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a Edwardionar el su propio descontento..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 49 train loss: 0.023268311024437865
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a todo plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre al P..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 50 train loss: 0.021875506841468734
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. Est...
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y unaándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.3010818196667565
    The son, a steady respectable young man, was amply provided for by the fortune of his mother, which had been large, and half of which devolved on him on his coming of age. | El hijo, un joven serio y respetable, tenía el futuro asegurado por la fortuna de su madre, que era cuantiosa, y de cuya mitad había entrado en posesión al cumplir su mayoría de edad. | La luz un hombre excelente a ningún hombre bien parecido fuego, muy diferente de su madre, por natural del joven de su madre, y su madre,a, escrúblic de su madre,go de trato hacia la mitad. Est de su madre,.
    From all danger of seeing Willoughby again, her mother considered her to be at least equally safe in town as in the country, since his acquaintance must now be dropped by all who called themselves her friends. | En cuanto al peligro de encontrarse de nuevo con Willoughby, su madre pensaba que Marianne estaba tan a salvo en la ciudad como en el campo, dado que nadie entre quienes se consideraban sus amigos lo admitiría ahora en su compañía. | Todos decían de Willoughby ocupó su madre en cuidad en ese momento; y a pensar en ese momento en que de su afecto por naturaleza, cultivir el matrimonio y de todas sus hijas.os que Willoughby se ricas.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 51 train loss: 0.020771715608606866
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acababa de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada me servirá más que el descanso, si me lo permite, yo me beberé el vino. Pero
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Admitió, entonces, que aunque ellos nunca le había informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su afectoo sus y no le extrañaba algún que se arrepían.pu..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 52 train loss: 0.019625223786025853
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacer un un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.es.........
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 53 train loss: 0.01844908202115398
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en tanto clavada en tanto privadalo de las atenciones que, en tanto amigo y casi eniente, pas empecía..
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 54 train loss: 0.017375406763794555
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en tanto privándolo de las atenciones que, en tanto amigo y se pariente, se secía...
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de un manera es falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de unarlo. -No
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 55 train loss: 0.016339361102187207
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que se siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Es cierto que no la conozco desde hace mucho, pero al menos, pero a bastante tiempo he sabido de usted y de toda su familia bastante oídas; y tan pronto como la vi, sentí como como si fuera una antigua conocida.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 56 train loss: 0.01524933133056605
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy content de de nole una favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.es.........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 57 train loss: 0.014252184983947641
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo ningunaártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. de
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse en alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 58 train loss: 0.01345232302964024
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos que un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 59 train loss: 0.012642452342121245
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que prim la la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de cuán hermana. querida...
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 60 train loss: 0.011849380524635508
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Noampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. D
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. P..
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.3185724847846561
    ONE person I was sure would represent me as capable of any thing-- What I felt was dreadful!--My resolution was soon made, and at eight o'clock this morning I was in my carriage. | ¡Lo que sentí fue atroz! Rápidamente tomé una decisión, y hoy a las ocho de la mañana ya me encontraba en mi carruaje. | En verdad, estoy segura de que estaba cumplido como lo era todo...,..., la mía en mi ca, y casi se me fue infiernos a la mañana me fue infella a este inv...
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | Parece tan fecho, pero si que Willoughby es un momento; y le pareció haber un momento replicó: las cuales que pedir la voz baja que decían esténtan en cuestión, un gran placer. -
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 61 train loss: 0.011089919848585284
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos en los de Marianne.
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. indign
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 62 train loss: 0.010422627121064957
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. Pero..
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de que pienso que no vale la pena esperarlo dos o tres meses. -No........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 63 train loss: 0.009790650759400292
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todo los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 64 train loss: 0.0092471206814728
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. -No
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 65 train loss: 0.00873801081332487
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al señor señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.es.........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 66 train loss: 0.008165196134624157
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.........
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 67 train loss: 0.007886147174935836
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco y que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 68 train loss: 0.007341755132263454
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. L.....
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 69 train loss: 0.006803612177889165
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. No.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.es.........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 70 train loss: 0.006586882631995945
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. Har......
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. P
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.3368215991391076
    From all danger of seeing Willoughby again, her mother considered her to be at least equally safe in town as in the country, since his acquaintance must now be dropped by all who called themselves her friends. | En cuanto al peligro de encontrarse de nuevo con Willoughby, su madre pensaba que Marianne estaba tan a salvo en la ciudad como en el campo, dado que nadie entre quienes se consideraban sus amigos lo admitiría ahora en su compañía. | Tras ella, de ver a Willoughby ocupó su madre en ese momento; y a haberlas caer en la ciudad; con todo el placer de las circunstancias tendréame, que le deben ser montes más bondad de su madre con ella.
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | Parece tan absoluto es que Willoughby... así que su hermana, creo extraordinario; un momento repito; y luego estaban en ocasiones la voz baja apenas parecía que ella misma lo estés. imagino lo que ella.... -.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 71 train loss: 0.006104371923191304
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. como........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 72 train loss: 0.0059174010121904605
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 73 train loss: 0.005578570506329854
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a todo plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 74 train loss: 0.005235206443127003
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en unlozos en un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 75 train loss: 0.00500110150375343
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. respecto...
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en sol comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 76 train loss: 0.00471563686090582
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna otraca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. P
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus gustaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. ¡.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 77 train loss: 0.004437519231351552
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y como pronto como la señora, sentí casi como si fuera una antigua conocida.
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descansá si me lo permite, yo me beberé el vino. Pero..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 78 train loss: 0.004243248432584397
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró en nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. P..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 79 train loss: 0.00399632940322838
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. P.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 80 train loss: 0.003851833850048579
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus inclinos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. respecto...
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne.
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.3530612786610921
    ONE person I was sure would represent me as capable of any thing-- What I felt was dreadful!--My resolution was soon made, and at eight o'clock this morning I was in my carriage. | ¡Lo que sentí fue atroz! Rápidamente tomé una decisión, y hoy a las ocho de la mañana ya me encontraba en mi carruaje. | En verdad, estoy segura de que estaba cumplido como lo era...,... era imposible convencerme de que estaba profundamente en esta solicito, y yo a mi intención era lo que vi, y yo era mi dejó todo el señor Willoughby. ab
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | -Es tan grande que Willoughby... ¡a! -y... puede extrañan una cierta crueldad en un momento; se comportaba la voz un tanto, dijo que decían su hermana la situación de tu hermana se imagino. -. -.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 81 train loss: 0.0037224740833165003
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. dos........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 82 train loss: 0.003543150401069456
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que lo siquiera se acerque a él amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 83 train loss: 0.0034540362275120888
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacer aunque un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. como o.......
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían.pu....
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 84 train loss: 0.0033091144612083187
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de toda y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida.
    Elinor had always thought it would be more prudent for them to settle at some distance from Norland, than immediately amongst their present acquaintance. On THAT head, therefore, it was not for her to oppose her mother's intention of removing into Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. | Elinor había pensado siempre que sería más Prudente para ellas establecerse a alguna distancia de Norland antes que entre sus actuales conocidos, por lo que no se opuso a las intenciones de su madre de irse a Devonshire. -.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 85 train loss: 0.003160358887383497
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían.pu....
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. dos........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 86 train loss: 0.0031812973358281344
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me be cree, yo me beberé el vino. Pero.
    She acknowledged, therefore, that though she had never been informed by themselves of the terms on which they stood with each other, of their mutual affection she had no doubt, and of their correspondence she was not astonished to hear. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuo afecto y no le extrañaba saber que se escribían. | Admitió, entonces, que aunque ellos nunca le habían informado sobre qué tipo de relaciones tenían, a ella no le cabía duda alguna sobre su mutuaba afecto y no le extrañaba saber que se escribían. qué...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 87 train loss: 0.0029728239728742606
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. Est...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 88 train loss: 0.0028476248058345315
    Elinor drew near, but without saying a word; and seating herself on the bed, took her hand, kissed her affectionately several times, and then gave way to a burst of tears, which at first was scarcely less violent than Marianne's. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne. | Elinor se acercó, pero sin decir palabra; y sentándose en la cama, le tomó una mano, la besó afectuosamente varias veces y luego estalló en sollozos en un comienzo apenas menos violentos que los de Marianne.
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 89 train loss: 0.0026914726530032403
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    I have not had so many opportunities of estimating the minuter propensities of his mind, his inclinations and tastes, as you have; but I have the highest opinion in the world of his goodness and sense. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. | No he tenido tantas oportunidades como tú de apreciar hasta las más mínimas tendencias de su mente, sus inclinaciones, sus gustos; pero tengo la mejor opinión del mundo respecto de su bondad y sensatez. respecto...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 90 train loss: 0.0026296433635878487
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrarse la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía...
    But I have just left Marianne in bed, and, I hope, almost asleep; and as I think nothing will be of so much service to her as rest, if you will give me leave, I will drink the wine myself." | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. | Pero acabo de dejar a Marianne acostada y, espero, casi dormida; y como creo que nada le servirá más que el descanso, si me lo permite, yo me beberé el vino. Pero..
    


      0%|          | 0/9 [00:00<?, ?it/s]


    Valid loss: 0.36874134341875714
    From all danger of seeing Willoughby again, her mother considered her to be at least equally safe in town as in the country, since his acquaintance must now be dropped by all who called themselves her friends. | En cuanto al peligro de encontrarse de nuevo con Willoughby, su madre pensaba que Marianne estaba tan a salvo en la ciudad como en el campo, dado que nadie entre quienes se consideraban sus amigos lo admitiría ahora en su compañía. | Tras ella, de su madre Willoughby ocupaba de su madre en ese aspecto, se hacía tan grande como la ciudad; en la ciudad de sunos tres mía a otra vez; con sus hijas. un compañía, su madre se ric
    "It may be so; but Willoughby is capable--at least I think"--he stopped a moment; then added in a voice which seemed to distrust itself, "And your sister--how did she--" | -Podría ser así; pero Willoughby es capaz... al menos eso creo -se interrumpió durante un instante, y luego agregó en una voz que parecía desconfiar de sí misma-; y su hermana, ¿cómo lo ha...? | -Es tan grande estar sea tan conveniente como por Willoughby es un momento; piensa lo que de su voz: un momento en ese momento la voz baja la voz baja su hermana. vida. lo que de su hermana... lo que ella...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 91 train loss: 0.002515540795412834
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. L.
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Es cierto que no la conozco desde hace mucho, a al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como la fuera una antigua conocida.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 92 train loss: 0.002412082301603945
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    "It would be impossible, I know," replied Elinor, "to convince you that a woman of seven and twenty could feel for a man of thirty-five anything near enough to love, to make him a desirable companion to her. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella. | -Sé -dijo Elinor- que sería imposible convencerte de que una mujer de veintisiete pueda sentir por un hombre de treinta y cinco algo que ni siquiera se acerque a ese amor que lo transformaría en un compañero deseable para ella.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 93 train loss: 0.002279158956794576
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. como........
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 94 train loss: 0.002261542254006611
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. 
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 95 train loss: 0.0021484476815026203
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    I have not known you long to be sure, personally at least, but I have known you and all your family by description a great while; and as soon as I saw you, I felt almost as if you was an old acquaintance. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida. | Es cierto que no la conozco desde hace mucho, personalmente al menos, pero durante bastante tiempo he sabido de usted y de toda su familia por oídas; y tan pronto como la vi, sentí casi como si fuera una antigua conocida.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 96 train loss: 0.0021507616174504065
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. las..
    She was first called to observe and approve him farther, by a reflection which Elinor chanced one day to make on the difference between him and his sister. It was a contrast which recommended him most forcibly to her mother. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana. | Lo que primero la llevó a observarlo con mayor detención y a que le gustara aún más, fue una reflexión que dio en hacer Elinor un día respecto de cuán diferente era de su hermana.in en en...
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 97 train loss: 0.002050336406493632
    He does not draw himself, indeed, but he has great pleasure in seeing the performances of other people, and I assure you he is by no means deficient in natural taste, though he has not had opportunities of improving it. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo. | El no dibuja, es cierto, pero disfruta enormemente viendo dibujar a otras personas y, puedo asegurártelo, de ninguna manera está falto de un buen gusto natural, aunque no se le ha ofrecido oportunidad de mejorarlo..
    She vowed at first she would never trim me up a new bonnet, nor do any thing else for me again, so long as she lived; but now she is quite come to, and we are as good friends as ever. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. | Primero juró que nunca más volvería a arreglarme ninguna toca nueva ni jamás haría ninguna otra cosa por mí; pero ahora ya se ha aplacado y estamos tan amigas como siempre. -..
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 98 train loss: 0.001959101033892918
    And with this pleasing anticipation, she sat down to reconsider the past, recall the words and endeavour to comprehend all the feelings of Edward; and, of course, to reflect on her own with discontent. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento. | Y con este agradable vaticinio se sentó a reconsiderar el pasado, recordar las palabras e intentar comprender los sentimientos de Edward; y, por supuesto, a reflexionar sobre su propio descontento.o..
    "Though with your usual anxiety for our happiness," said Elinor, "you have been obviating every impediment to the present scheme which occurred to you, there is still one objection which, in my opinion, cannot be so easily removed." | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente. | -Aunque con su habitual preocupación por nuestra felicidad -dijo Elinor- ha estado obviando todos los obstáculos a este plan que ha podido imaginar, persiste una objeción que, en mi opinión, no puede ser despachada tan fácilmente.
    


      0%|          | 0/77 [00:00<?, ?it/s]


    Epoch 99 train loss: 0.0019059561687376496
    Lord bless me!--I am sure it would put ME quite out of patience!--And though one would be very glad to do a kindness by poor Mr. Ferrars, I do think it is not worth while to wait two or three months for him. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses. | Creo que yo no tendría paciencia. Y aunque cualquiera estaría muy contento de hacerle un favor al pobre señor Ferrars, de verdad pienso que no vale la pena esperarlo dos o tres meses.es.........
    She would not be frightened from paying him those attentions which, as a friend and almost a relation, were his due, by the observant eyes of Lucy, though she soon perceived them to be narrowly watching her. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía. | Tampoco iba a dejarse arredrar por la observadora mirada de Lucy, que no tardó en sentir clavada en ella, privándolo de las atenciones que, en tanto amigo y casi pariente, se merecía.pu..

```python
from torchinfo import summary

print(summary(model))
```

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Transformer                              --
    ├─Linear: 1-1                            2,565,000
    ├─Embedding: 1-2                         2,560,000
    ├─ModuleList: 1-3                        --
    │    └─Dropout: 2-1                      --
    │    └─Dropout: 2-2                      --
    ├─ModuleList: 1-4                        --
    │    └─EncoderBlock: 2-3                 --
    │    │    └─MultiHeadAttention: 3-1      1,050,624
    │    │    └─ModuleList: 3-2              --
    │    │    └─Linear: 3-3                  1,050,624
    │    │    └─Linear: 3-4                  1,049,088
    │    │    └─ReLU: 3-5                    --
    │    │    └─ModuleList: 3-6              2,048
    ├─ModuleList: 1-5                        --
    │    └─DecoderBlock: 2-4                 --
    │    │    └─MultiHeadAttention: 3-7      1,050,624
    │    │    └─MultiHeadAttention: 3-8      1,050,624
    │    │    └─ModuleList: 3-9              --
    │    │    └─Linear: 3-10                 1,050,624
    │    │    └─Linear: 3-11                 1,049,088
    │    │    └─ReLU: 3-12                   --
    │    │    └─ModuleList: 3-13             3,072
    =================================================================
    Total params: 12,481,416
    Trainable params: 12,481,416
    Non-trainable params: 0
    =================================================================

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, ) as prof:
    model(sequence.to(DEVICE), prev_target.to(DEVICE))

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

    ---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    ---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   aten::linear         0.31%      65.000us        32.71%       6.948ms       1.390ms             5  
                    aten::addmm        29.34%       6.232ms        32.06%       6.809ms       1.362ms             5  
                   aten::einsum         2.18%     463.000us        29.74%       6.317ms     350.944us            18  
                  aten::dropout         0.32%      67.000us        18.51%       3.931ms     561.571us             7  
                      aten::bmm        16.71%       3.548ms        18.02%       3.827ms     212.611us            18  
               aten::bernoulli_        13.37%       2.840ms        13.37%       2.840ms     405.714us             7  
                  aten::reshape         1.61%     341.000us         9.04%       1.920ms      46.829us            41  
                    aten::copy_         7.16%       1.521ms         7.16%       1.521ms      56.333us            27  
                    aten::clone         0.28%      60.000us         5.57%       1.184ms      98.667us            12  
                      aten::add         5.43%       1.154ms         5.43%       1.154ms      60.737us            19  
    ---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 21.238ms
    
    

    STAGE:2023-02-02 17:02:12 52675:8175420 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
    Exception in thread SystemMonitor:
    Traceback (most recent call last):
      File "/Users/vik/.pyenv/versions/3.10.8/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/Users/vik/.pyenv/versions/3.10.8/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/Users/vik/.virtualenvs/nnets/lib/python3.10/site-packages/wandb/sdk/internal/system/system_monitor.py", line 118, in _start
        asset.start()
      File "/Users/vik/.virtualenvs/nnets/lib/python3.10/site-packages/wandb/sdk/internal/system/assets/cpu.py", line 166, in start
        self.metrics_monitor.start()
      File "/Users/vik/.virtualenvs/nnets/lib/python3.10/site-packages/wandb/sdk/internal/system/assets/interfaces.py", line 168, in start
        logger.info(f"Started {self._process.name}")
    AttributeError: 'NoneType' object has no attribute 'name'
    STAGE:2023-02-02 17:02:12 52675:8175420 ActivityProfilerController.cpp:300] Completed Stage: Collection
    
