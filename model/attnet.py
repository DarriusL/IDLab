# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var, util, callback

logger = glb_var.get_value('logger')


def attn_pad_msk(seq, expand_len, mask_item = [0]):
    '''Return the mask for pad

    Parameters:
    -----------

    expand_len: int
        expand the size of seq from (batch_size, len_seq) to (batch_size, expand_len, len_seq)
        (firstly, (batch_size, 1, len_seq))
    
    seq: torch.Tensor
        The template used to generate the mask
        (batch_size, seq_len)

    for_msk: list
        The value to be masked
    
    Returns:
    --------

    pad_mask: torch.Tensor
        The expanded mask of pad
        (batch_size, expand_len, seq_len)

    '''
    #seq:(bach_size, seq_len)
    #msk:(bach_size, seq_len) -> (bach_size, 1, seq_len)
    try:
        msk_num = len(mask_item);
    except:
        logger.error('The mask must be of type list');
        callback.CustomException('TypeError');
    msk = seq.data.eq(mask_item[0]).unsqueeze(1).repeat(1, expand_len,  1);
    for i in range(1, msk_num - 1):
        msk = torch.bitwise_or(
            msk,
            seq.data.eq(mask_item[i]).unsqueeze(1).repeat(1, expand_len,  1)
        )
    return msk.to(glb_var.get_value('device'));

def attn_subsequence_mask(seq):
    '''Retuen the mask for subsequence(upper triangular matrix)

    Parameters:
    -----------

    seq: torch.Tensor
        (batch_size, seq_len)

    Returns:
    --------

    subsequence_mask: torch.Tensor
        (batch_size, tgt_len, seq_len)
    '''
    return torch.triu(torch.ones((seq.shape[0], seq.shape[1], seq.shape[1])), diagonal = 1).eq(1).to(glb_var.get_value('device'));

class PositionEncoding(torch.nn.Module):
    '''generate the positional encoding for Transformer

    Parameters:
    -----------

    d_model: int 
        Dimensions of the encoding of the input encoding vector

    max_len: int
        the maximum length of the sequence
    
    Methods:
    ---------

    forward(self, x):
        forward propagation of net
        -x: torch.Tensor
            (batchsize, seq_len, d)
    '''
    def __init__(self, d, max_len = 10000) -> None:
        super().__init__();

        #position encode:(max_len, d)
        pe = torch.zeros(max_len, d);
        #postion: (max_len, 1)
        pos = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1);
        #weight: (d//2)
        w = torch.pow(10000, -torch.arange(0, d, 2).float() / d).unsqueeze(0);
        #generater position code
        pe[:, 0::2] = torch.sin(torch.matmul(pos, w));
        pe[:, 1::2] = torch.cos(torch.matmul(pos, w));
        #set as buffer:(max_len, d) -> (1, max_len, d)
        self.register_buffer('pe', pe.unsqueeze(0));

    def forward(self, x):

        return x + self.pe[:, 0:x.shape[1], :];

class LearnablePositionEncoding(torch.nn.Module):
    def __init__(self, d, max_len = 10000) -> None:
        super().__init__()
        self.Embed = torch.nn.Embedding(max_len, d)
        self.register_buffer('Pos_idx', torch.arange(max_len));

    def forward(self, x):
        return self.Embed(self.Pos_idx[:x.shape[1]].unsqueeze(0).repeat(x.shape[0], 1));

class PositionwiseFeedForwardNet(torch.nn.Module):
    '''Position-wise Feed-Forward Network for Transformer

    Parameters:
    -----------
    d:int
    Dimension of embedded victor

    d_fc:int
    number of nodes in Linear Network 
    '''
    def __init__(self, d, d_fc) -> None:
        super().__init__()

        self.Net = torch.nn.Sequential(
            torch.nn.Linear(d, d_fc, bias = True),
            torch.nn.ReLU(),
            torch.nn.Linear(d_fc, d, bias = True)
        )
    def forward(self, inputs):
        #inputs:(batch_size, seq_len, d)
        #outputs:(batch_size, seq_len, d)
        return self.Net(inputs);

class ScaledDotProductAttention(torch.nn.Module):
    '''Scaled Dot Product Attention use separately from multihead

    Parameters:
    -----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    Methods:
    --------
    forward(self, input_Q, input_K, input_V)
        argin:
            -input_Q: (batch_size, len_q, d)
            -input_K: (batch_size, len_k, d)
            -input_V: (batch_size, len_v, d)
            -attn_mask: (batch_size, len_q, len_k)
                default:None
        argout:
            -attention: (batch_size, len_q, d)
    
    '''
    def __init__(self, d, d_q, d_k) -> None:
        super().__init__();
        self.W_Q = torch.nn.Linear(d, d_q, bias = False);
        self.W_K = torch.nn.Linear(d, d_k, bias = False);
        self.Softmax = torch.nn.Softmax(dim = -1);
        self.temperature = torch.as_tensor(d_k);

    def forward(self, input_Q, input_K, input_V, mask = None):
        #input_Q:(batch_size, len_q, d)
        #Q:(batch_size, len_q, d_q)
        Q = self.W_Q(input_Q);
        #input_K:(batch_size, len_k, d)
        #K:(batch_size, len_k, d_k)
        K = self.W_K(input_K);
        #scores:(batch_size, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.temperature);
        if mask is not None:
            scores.masked_fill_(mask, -1e8)
        #attention:(batch_size, len_q, len_k)*(batch_size, len_v, d)
        return torch.matmul(self.Softmax(scores), input_V);

class ScaledDotProductAttentionLite(torch.nn.Module):
    '''Scaled Dot Product Attention used for multihead

    Parameters:
    ----------

    Temperature: float
        To scale scores.
    
    Methods:
    --------

    forward(self, Q, K, V, mask)
        argin:
            -Q: (batch_size, n_heads, len_q, d_q)
            -K: (batch_size, n_heads, len_k, d_k)
            -V: (batch_size, n_heads, len_v, d_v)
            -mask: ((batch_size, n_heads, len_q, len_k))
            normally, d_q = d_k = d_v, len_q = len_k = len_v = seq_len
        argout:
            -attention: (batch_size, n_heads, len_q, len_k) * (batch_size, n_heads, len_v, d_v)
    
    Notes:
    ------
    Generative matrices of Q, K, V implemented in multi-head attention.
    '''
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = torch.as_tensor(temperature);
        self.Softmax = torch.nn.Softmax(dim = -1);

    def forward(self, Q, K, V, mask = None):
        #Q:(batch_size, len_q, d_q)
        #K:(batch_size, len_k, d_k)
        #scores:(batch_size, n_heads, len_q, len_k) and mask the score with -1e8(for amp)
        #scores:(batch_size, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.temperature);
        if mask is not None:
            scores.masked_fill_(mask, -1e8);
        #attention:(batch_size, n_heads, len_q, d_v)
        return torch.matmul(self.Softmax(scores), V);
        
class MultiHeadAttention(torch.nn.Module):
    '''Multi-Head Attention Mechhanism

    Parameters:
    ----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    n_heads:int
    Number of the head

    Methods:
    --------
    forward(self, input_Q, input_K, input_V, mask)
        argin:
            -input_Q: (batch_size, len_q, d)
            -input_K: (batch_size, len_k, d)
            -input_V: (batch_size, len_v, d)
            -mask: (batch_size, len_q, seq_k)
        argout:
            -attention:(batch_size, len_q, d)
    '''
    def __init__(self, d, d_q = 64, d_k = 64, d_v = 64, n_heads = 3) -> None:
        super().__init__();
        self.W_Q = torch.nn.Linear(d, d_q * n_heads, bias = False);
        self.W_K = torch.nn.Linear(d, d_k * n_heads, bias = False);
        self.W_V = torch.nn.Linear(d, d_v * n_heads, bias = False);
        self.W_O = torch.nn.Linear(d_v * n_heads, d);
        self.Attnet = ScaledDotProductAttentionLite(temperature = d_k/n_heads);
        util.set_attr(
            obj = self,
            dict_ = dict(
            d = d,
            d_q = d_q,
            d_k = d_k,
            d_v = d_v,
            n_heads = n_heads
            )
        )

    def forward(self, input_Q, input_K, input_V, mask = None):
        batch_size = input_Q.shape[0];
        #input_Q:(batch_size, len_q, d) -> Q:(batch_szie, n_heads, len_q, d_q)
        Q = self.W_Q(input_Q).reshape(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2);
        #input_K:(batch_size, len_k, d) -> K:(batch_szie, n_heads, len_k, d_k)
        K = self.W_K(input_K).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2);
        #input_V:(batch_size, len_v, d) -> V:(batch_szie, n_heads, len_v, d_v)
        V = self.W_V(input_V).reshape(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2);
        if mask is not None:
            #expand the mask matrix: (batch_size, len_q, len_k) -> (batch_size, n_heads, len_q, len_k)
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1);
        #multihead attention:(batch_size, n_heads, len_q, d_v)
        att = self.Attnet(Q, K, V, mask);
        #contact multihead attention:()
        att = att.transpose(1, 2).reshape(batch_size, -1, self.d_v *self.n_heads);
        #output attention:(batch_size, len_q, d)
        return self.W_O(att);

class TemplateLayer_PostLN(torch.nn.Module):
    ''' Template for Post LayerNorm Layer 

    Parameters:
    ------------
    d:int
    Dimension of embedded victor

    d_fc:int
    Number of nodes in the dense network

    Methods:
    -------
    add_and_norm(self, input_1, input_2)

    '''
    def __init__(self, d, d_fc) -> None:
        super().__init__();
        self.Posi_wise_feed_fwd_net = PositionwiseFeedForwardNet(d, d_fc);
        self.Dropout = torch.nn.Dropout(0.1);
        self.Norm = torch.nn.LayerNorm(d);
    def add_and_norm(self, input_1, input_2):
        ''' Add and Normlization

        Parameters:
        -----------
        input_1:torch.Tensor
        (batch_size, seq_len, d)

        input_2:torch.Tensor
        (batch_size, seq_len, d)
        '''
        #input:(batch_size, seq_len, d)
        return self.Norm(input_1 + input_2);

class EncoderLayer_PostLN(TemplateLayer_PostLN):
    '''Encode Layer for Post LayerNorm Transformer

    Parameters:
    ----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    d_fc:int
    Number of nodes in the dense network
    
    n_heads:int
    Number of the head
    
    Methods:
    -------
    
    forward(self, enc_inputs, mask)
        argin:
            -enc_inputs:(batch_size, src_len, d)
            -enc_self_attn_mask:(batch_size, src_len, src_len)
        argout:
            -enc_outputs(batch_size, src_len, d)

    '''
    def __init__(self, d, d_fc, n_heads) -> None:
        super().__init__(d, d_fc);
        d_q = d_k = d_v = int(d/n_heads);
        self.Multiheadatt = MultiHeadAttention(d, d_q, d_k, d_v, n_heads);

    def forward(self, enc_input, mask = None):
        #enc_input:(batch_size, src_len, d)
        #att:(batch_size, src_len, d)
        att = self.Multiheadatt(enc_input, enc_input, enc_input, mask);
        att = self.add_and_norm(enc_input, self.Dropout(att));
        #dec_output:(batch_size, src_len, d)
        dec_output = self.Posi_wise_feed_fwd_net(att);
        return self.add_and_norm(att, self.Dropout(dec_output));

class DecodeLayer_PostLN(TemplateLayer_PostLN):
    '''Decode Layer for Post LayerNorm Transformer

    Parameters:
    ----------- 

    Parameters:
    ----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    d_fc:int
    Number of nodes in the dense network
    
    n_heads:int
    Number of the head
    
    Methods:
    -------
    
    forward(self, dec_input, enc_output, self_att_mask, enc_att_mask)
        argin:
            -dec_input: (batch_size, tgt_len, d)
            -enc_output: (batch_size, src_len, d)
            -self_att_mask: (batch_size, tgt_len, tgt_len)
            -enc_att_mask: (batch_size, tgt_len, src_len)
        argout:
            -output(batch_size, src_len, d)

    '''
    def __init__(self, d, d_fc, n_heads) -> None:
        super().__init__(d, d_fc);
        d_q = d_k = d_v = int(d/n_heads);
        self.Self_attention = MultiHeadAttention(d, d_q, d_k, d_v, n_heads);
        self.Enc_attention = MultiHeadAttention(d, d_q, d_k, d_v, n_heads);

    def forward(self, dec_input, enc_output, self_att_mask, enc_att_mask):
        #dec_input: (batch_size, tgt_len, d)
        #enc_output: (batch_size, src_len, d)
        #self_att_mask: (batch_size, tgt_len, tgt_len)
        #enc_att_mask: (batch_size, tgt_len, src_len)
        #dec_output: (batch_size, tgt_len, d)
        dec_output = self.Self_attention(dec_input, dec_input, dec_input, self_att_mask);
        dec_output = self.add_and_norm(dec_input, self.Dropout(dec_output));
        #ed_output:(batch_size, tgt_len, d)
        ed_output = self.Enc_attention(dec_output, enc_output, enc_output, enc_att_mask);
        ed_output = self.add_and_norm(dec_output, self.Dropout(ed_output));
        #output:(batch_size, tgt_len, d)
        output = self.Posi_wise_feed_fwd_net(ed_output);
        return self.add_and_norm(ed_output, self.Dropout(output))
    
class EncoderLayer_PreLN(torch.nn.Module):
    '''Encode Layer for Post LayerNorm Transformer

    Parameters:
    ----------
    d:int
    Dimension of embedded victor

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    d_fc:int
    Number of nodes in the dense network
    
    n_heads:int
    Number of the head
    
    Methods:
    -------
    
    forward(self, enc_inputs, mask)
        argin:
            -enc_inputs:(batch_size, src_len, d)
            -enc_self_attn_mask:(batch_size, src_len, src_len)
        argout:
            -enc_outputs(batch_size, src_len, d)

    '''
    def __init__(self, d, d_fc, n_heads) -> None:
        super().__init__();
        d_q, d_k, d_v = 64, 64, 64;
        self.Posi_wise_feed_fwd_net = PositionwiseFeedForwardNet(d, d_fc);
        self.Dropout = torch.nn.Dropout(0.1);
        self.Multiheadatt = MultiHeadAttention(d, d_q, d_k, d_v, n_heads);
        self.Norm = torch.nn.LayerNorm(d);

    def forward(self, enc_input, mask = None):
        #enc_input:(batch_size, src_len, d)
        #att:(batch_size, src_len, d)
        enc_input_norm = self.Norm(enc_input);
        att = self.Multiheadatt(enc_input_norm, enc_input_norm, enc_input_norm, mask);
        att = self.Dropout(att) + enc_input;
        #dec_output:(batch_size, src_len, d)
        att_norm = self.Norm(att);
        dec_output = self.Posi_wise_feed_fwd_net(att_norm);
        return att + self.Dropout(dec_output);