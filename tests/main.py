"""
Train with contrastive loss for 10 epochs with batch size 128:
     python main.py --train=True --loss=contrastive --epcs=10 --bs=128 --opt=RMSprop --reset_weights=True
"""

from functions import*
#from plot_functions import*

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adagrad
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('loss','contrastive','loss function.')
flags.DEFINE_integer('epcs', 10, 'number of epochs.')
flags.DEFINE_integer('bs', 128, 'batch size.')
flags.DEFINE_string('opt', 'RMSprop', 'optimizer')
flags.DEFINE_bool('reset_weights', True, 'Reset model weights.')
flags.DEFINE_bool('train', True, 'True to train, False to predict.')

num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]


#create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
#tr_pairs_lbs = [y_train[i] for i in digit_indices]
tr_pairs, tr_y, tr_pairs_lbs = create_pairs(x_train,y_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y, te_pairs_lbs = create_pairs(x_test,y_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# the weights are shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

if FLAGS.reset_weights == False:
    model.load_weights('model_after_train.h5')
else:
    model.save_weights('model.h5')

if FLAGS.loss=='contrastive':
    loss=contrastive_loss
elif FLAGS.loss=='potential':
    loss=potential_loss

def embedding(inputs,base_network):
    intermediate_layer_model = Model(inputs=base_network.get_input_at(0),
                                 outputs=base_network.get_output_at(0))
    return intermediate_layer_model.predict(inputs)


def train(loss=loss, opt=FLAGS.opt, epcs=FLAGS.epcs, bs=FLAGS.bs, reset_weights=FLAGS.reset_weights):
    print('training')
    if reset_weights == True:
        model.load_weights('model.h5')
        #base_network.load_weights('bn.h5')
        print('Norm pre-training', np.linalg.norm(model.get_weights()[0], 'fro'))
        # print('BN Norm pre-training', np.linalg.norm(base_network.get_weights()[0],'fro'))
        # BN norm is the same as model's,as expected

    model.compile(loss= loss, optimizer=FLAGS.opt, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=FLAGS.bs,
              epochs=FLAGS.epcs)

    return model  # , base_network

if FLAGS.train:
        train(loss=loss, opt=FLAGS.opt, epcs=FLAGS.epcs, bs=FLAGS.bs, reset_weights=True)
        model.save_weights('model_after_train.h5')
        print('Norm after training', np.linalg.norm(model.get_weights()[0], 'fro'))
else:
    print('Norm pre-predicting', np.linalg.norm(model.get_weights()[0], 'fro'))
    predict = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, predict)
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

