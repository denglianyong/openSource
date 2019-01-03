import tensorflow as tf

class Generator(object):
    """SeqGAN implementation based on https://arxiv.org/abs/1609.05473
        "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient"
        Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu
    """
    def __init__(self, config):
        """ Basic Set up

        Args:
           num_emb: output vocabulary size
           batch_size: batch size for generator
           emb_dim: LSTM hidden unit dimension
           sequence_length: maximum length of input sequence
           start_token: special token used to represent start of sentence
           initializer: initializer for LSTM kernel and output matrix
        """
        self.num_emb = config.num_emb  #词典的容量
        self.batch_size = config.gen_batch_size # 样本数
        self.emb_dim = config.emb_dim  # 词向量的维度数
        self.hidden_dim = config.hidden_dim # 隐藏变量的个数
        self.sequence_length = config.sequence_length #一句话单词的个数
        self.start_token = tf.constant(config.start_token, dtype=tf.int32, shape=[self.batch_size])
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        
    def build_input(self, name):
        """ Buid input placeholder

        Input:
            name: name of network
        Output:
            self.input_seqs_pre (if name == pretrained)
            self.input_seqs_mask (if name == pretrained, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.rewards (if name == 'adversarial')
        """
        assert name in ['pretrain', 'adversarial', 'sample']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_pre")
            self.input_seqs_mask = tf.placeholder(tf.float32, [None, self.sequence_length], name="input_seqs_mask")
        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_adv")
            self.rewards = tf.placeholder(tf.float32, [None, self.sequence_length], name="reward")
    
    #预训练网络
    def build_pretrain_network(self):
        """ Buid pretrained network

        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrained_loss
            self.pretrained_loss_sum (optional)
        """
        self.build_input(name="pretrain")
        self.pretrained_loss = 0.0
        with tf.variable_scope("teller"):
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    with tf.device("/cpu:0"):
                        if j == 0:
                            # <BOS>  查找start token的词向量
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_pre[:, j-1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    #预测下一个单词 
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    # calculate loss   最后一个是标签 label  input_seqs_pre[:,j]
                    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs_pre[:,j], logits=logits)
                    pretrained_loss_t = tf.reduce_sum(tf.multiply(pretrained_loss_t, self.input_seqs_mask[:,j]))
                    self.pretrained_loss += pretrained_loss_t
                    word_predict = tf.to_int32(tf.argmax(logits, 1))
            self.pretrained_loss /= tf.reduce_sum(self.input_seqs_mask)
            self.pretrained_loss_sum = tf.summary.scalar("pretrained_loss",self.pretrained_loss)
        #对抗性神经网络                    
    def build_adversarial_network(self):
        """ Buid adversarial training network

        Input:
            self.input_seqs_adv
            self.rewards
        Output:
            self.gen_loss_adv
        """
        self.build_input(name="adversarial")
        self.softmax_list_reshape = []
        self.softmax_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                #  hidden_dim 隐藏节点的个数
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], "float32", self.initializer)
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 0:
                            # <BOS>
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_adv[:, j-1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax)
                    #1,2列交换位置
            self.softmax_list_reshape = tf.transpose(self.softmax_list, perm=[1, 0, 2])
            self.gen_loss_adv = -tf.reduce_sum(
                tf.reduce_sum(
                    # input_seqs_adv 是采用序列
                    tf.one_hot(tf.to_int32(tf.reshape(self.input_seqs_adv, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.softmax_list_reshape, [-1, self.num_emb]), 1e-20, 1.0)
                       #   rewards 是discriminator 计算的 
                    ), 1) * tf.reshape(self.rewards, [-1]))

    #采样神经网络    供rollout使用            
    def build_sample_network(self):
        """ Buid sampling network

        Output:
            self.sample_word_list_reshape
        """
        self.build_input(name="sample")
        self.sample_word_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                 #word_emb_W,output_W 参数与generator共享    
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                # 是否意味着都是完整序列????
                for j in range(self.sequence_length):
                    with tf.device("/cpu:0"):
                        if j == 0:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32) 
                    #LSTM 获取输出单词(embed表示) 和状态                   
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())
                    logits = tf.matmul(output, output_W)
                    logprob = tf.log(tf.nn.softmax(logits))
                    # 获取单词id
                    sample_word = tf.reshape(tf.to_int32(tf.multinomial(logprob, 1)), shape=[self.batch_size])
                    self.sample_word_list.append(sample_word) #sequence_length * batch_size
            self.sample_word_list_reshape = tf.transpose(tf.squeeze(tf.stack(self.sample_word_list)), perm=[1,0]) #batch_size * sequene_length
    def build(self):
        """Create all network for pretraining, adversairal training and sampling"""
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()
    def generate(self, sess):
        """Helper function for sample generation"""
        return sess.run(self.sample_word_list_reshape)
