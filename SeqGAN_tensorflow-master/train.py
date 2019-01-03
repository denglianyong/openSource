import random
import os
import cPickle
import numpy as np
import tensorflow as tf

from configuration import *
from utils import *
from dataloader import Gen_Data_loader, Dis_dataloader
from discriminator import Discriminator
from generator import Generator
from rollout import rollout
from target_lstm import TARGET_LSTM

#Hardware related setting
config_hardware = tf.ConfigProto()
config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.40
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(unused_argv):
    config_train = training_config()
    config_gen = generator_config()
    config_dis = discriminator_config()
    np.random.seed(config_train.seed)
    assert config_train.start_token == 0

    #Build dataloader for generaotr, testing and discriminator
    gen_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
    likelihood_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
    dis_data_loader = Dis_dataloader(config_dis.dis_batch_size)

    #Build generator and its rollout
    generator = Generator(config=config_gen)
    # 生成 3个神经网络
    generator.build()
    #  快速展开网络，序列未生成完就预测后边的序列，用于计算reward
    rollout_gen = rollout(config=config_gen)

    #Build target LSTM
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(config=config_gen, params=target_params) # The oracle model

    #Build discriminator
    discriminator = Discriminator(config=config_dis)
    discriminator.build_discriminator()

    #Build optimizer op for pretraining
    pretrained_optimizer = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    # 取出 teller 的所有变量， teller在 generator和rollout网络中
    var_pretrained = [v for v in tf.trainable_variables() if 'teller' in v.name] #Using name 'teller' here to prevent name collision of target LSTM
    # zip函数将 2个迭代器  组成tuple
    gradients, variables = zip(*pretrained_optimizer.compute_gradients(generator.pretrained_loss, var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    gen_pre_upate = pretrained_optimizer.apply_gradients(zip(gradients, variables))

    #Initialize all variables
    sess = tf.Session(config=config_hardware)
    sess.run(tf.global_variables_initializer())

    #Initalize data loader of generator   utils.py文件中
    #   target_lstm 网络生成真实数据 写入config_train.positive_file 文件
    generate_samples(sess, target_lstm, config_train.batch_size, config_train.generated_num, config_train.positive_file)
    gen_data_loader.create_batches(config_train.positive_file)

    #Start pretraining
    log = open('save/experiment-log.txt', 'w')
    print 'Start pre-training generator...'
    log.write('pre-training...\n')
    for epoch in xrange(config_train.pretrained_epoch_num):
        gen_data_loader.reset_pointer()
        for it in xrange(gen_data_loader.num_batch):
            #见第60行，加载target_lstm 神经网络的数据,用于预训练生成器====真实样本
            batch = gen_data_loader.next_batch()
            #真实数据训练  generator;有监督学习   batch 最后第一个是label
            _, g_loss = sess.run([gen_pre_upate, generator.pretrained_loss], feed_dict={generator.input_seqs_pre:batch,\
                                                                                    generator.input_seqs_mask:np.ones_like(batch)})
        if epoch % config_train.test_per_epoch == 0:
            #  generator 生成样本  与 真实数据的相似度
            generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
            likelihood_data_loader.create_batches(config_train.eval_file)
            #评估生成质量
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    print 'Start pre-training discriminator...'
    for t in range(config_train.dis_update_time_pre):
        print "Times: " + str(t)
        #   generator生成假数据+ target_lstm的真实数据;; 用于训练
        generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.negative_file)
        #  混合真假数据
        dis_data_loader.load_train_data(config_train.positive_file, config_train.negative_file)
        for _ in range(config_train.dis_update_epoch_pre):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: config_dis.dis_dropout_keep_prob
                }
                #交叉上最小;  主要是训练评分网络 用于给generator提供reward
                _ = sess.run(discriminator.train_op, feed)
    
    #Build optimizer op for adversarial training
    train_adv_opt = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    gradients, variables = zip(*train_adv_opt.compute_gradients(generator.gen_loss_adv,var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    train_adv_update = train_adv_opt.apply_gradients(zip(gradients, variables))

    #Initialize global variables of optimizer for adversarial training
    uninitialized_var = [e for e in tf.global_variables() if e not in tf.trainable_variables()]
    init_vars_uninit_op = tf.variables_initializer(uninitialized_var)
    sess.run(init_vars_uninit_op)

    #Start adversarial training   开始对抗训练
    for total_batch in xrange(config_train.total_batch):
        for iter_gen in xrange(config_train.gen_update_time):
            
            #  用generator进行抽样; LSTM 生成序列
            samples = sess.run(generator.sample_word_list_reshape)

            feed = {"pred_seq_rollout:0":samples}
            reward_rollout = []
            #calcuate the reward given in the specific stpe t by roll out
            # 用rollout网络计算指定动作的回报
            for iter_roll in xrange(config_train.rollout_num):
                
                # 生成器采样的获得的单词传给 rollout  ??有一个疑问？samples看代码是完整序列（与论文不符），为什么还要rollout
                rollout_list = sess.run(rollout_gen.sample_rollout_step, feed_dict=feed)


                rollout_list_stack = np.vstack(rollout_list) #shape: #batch_size * #rollout_step, #sequence length
                # 蒙特卡洛 展开成序列，贝尔曼方程计算 reward
                reward_rollout_seq = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:rollout_list_stack, discriminator.dropout_keep_prob:1.0})
                reward_last_tok = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:samples, discriminator.dropout_keep_prob:1.0})
                reward_allseq = np.concatenate((reward_rollout_seq, reward_last_tok), axis=0)[:,1]
                reward_tmp = []
                for r in xrange(config_gen.gen_batch_size):
                    reward_tmp.append(reward_allseq[range(r, config_gen.gen_batch_size * config_gen.sequence_length, config_gen.gen_batch_size)])
                reward_rollout.append(np.array(reward_tmp))
            #计算reward   
            rewards = np.sum(reward_rollout, axis=0)/config_train.rollout_num
            # 用reward 指导 generator 更新梯度
            _, gen_loss = sess.run([train_adv_update, generator.gen_loss_adv], feed_dict={generator.input_seqs_adv:samples,\
                                                                                        generator.rewards:rewards})
        if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1:
            #对抗训练后 用generator再次生成样本与模拟器(target_lstm，真实数据)进行比对
            generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
            likelihood_data_loader.create_batches(config_train.eval_file)
            #util.py中定义
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)

        for _ in range(config_train.dis_update_time_adv):
            generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.negative_file)
            dis_data_loader.load_train_data(config_train.positive_file, config_train.negative_file)

            for _ in range(config_train.dis_update_epoch_adv):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: config_dis.dis_dropout_keep_prob
                    }
                    #训练这个评分网络, score
                    _ = sess.run(discriminator.train_op, feed)
    log.close()
if __name__ == "__main__":
    tf.app.run()
