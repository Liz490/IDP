import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Concatenate, Multiply
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from layers.graph import GraphConv
from keras.layers import MultiHeadAttention

class KerasMultiSourceGCNModel(object):
    def __init__(self,use_mut,use_gexp,use_methy,regr=True):#
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.regr = regr

    def createMaster(self, drug_dim, mutation_dim, gexpr_dim, methy_dim, params):
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        
        mutation_input = Input(shape=(1,mutation_dim,1),name='mutation_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        #drug feature with GCN
        GCN_layer = GraphConv(units=params["unit_list"][0], step_num=1)([drug_feat_input, drug_adj_input])
        if params["use_relu"]:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if params["use_bn"]:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)

        for i in range(len(params["unit_list"]) - 1):
            GCN_layer = GraphConv(units=params["unit_list"][i + 1], step_num=1)([GCN_layer, drug_adj_input])
            if params["use_relu"]:
                GCN_layer = Activation('relu')(GCN_layer)
            else:
                GCN_layer = Activation('tanh')(GCN_layer)
            if params["use_bn"]:
                GCN_layer = BatchNormalization()(GCN_layer)
            GCN_layer = Dropout(0.1)(GCN_layer)
        
        GCN_layer = GraphConv(units=100,step_num=1)([GCN_layer,drug_adj_input])
        if params["use_relu"]:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if params["use_bn"]:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        if params["use_GMP"]:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)

        # genomic mutation feature
        x_mut = mutation_input
        if params["pos_attn_mut"] == 0:
            x_mut = tf.squeeze(x_mut, 1)
            x_mut = MultiHeadAttention(num_heads=params["nb_attn_head_mut"], key_dim=params["key_dim_mut"])(x_mut,
                                                                                                            x_mut)
            x_mut = tf.expand_dims(x_mut, 1)
        x_mut = Conv2D(filters=50, kernel_size=(1, 700), strides=(1, 5), activation='tanh', padding='valid')(x_mut)
        x_mut = MaxPooling2D(pool_size=(1,5))(x_mut)
        if params["pos_attn_mut"] == 1:
            x_mut = tf.squeeze(x_mut, 1)
            x_mut = MultiHeadAttention(num_heads=params["nb_attn_head_mut"], key_dim=params["key_dim_mut"])(x_mut,
                                                                                                            x_mut)
            x_mut = tf.expand_dims(x_mut, 1)
        x_mut = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_mut)
        x_mut = MaxPooling2D(pool_size=(1,10))(x_mut)
        if params["pos_attn_mut"] == 2:
            x_mut = tf.squeeze(x_mut, 1)
            x_mut = MultiHeadAttention(num_heads=params["nb_attn_head_mut"], key_dim=params["key_dim_mut"])(x_mut,
                                                                                                            x_mut)
            x_mut = tf.expand_dims(x_mut, 1)
        x_mut = Flatten()(x_mut)
        x_mut = Dense(100,activation = 'relu')(x_mut)
        x_mut = Dropout(0.1)(x_mut)
        if params["pos_attn_mut"] == 3:
            x_mut = tf.expand_dims(x_mut, 1)
            x_mut = MultiHeadAttention(num_heads=params["nb_attn_head_mut"], key_dim=params["key_dim_mut"])(x_mut,
                                                                                                            x_mut)
            x_mut = tf.squeeze(x_mut, 1)

        #gexp feature
        x_gexpr = gexpr_input
        if params["pos_attn_gexpr"] == 0:
            x_gexpr = tf.expand_dims(x_gexpr, -1)
            x_gexpr = MultiHeadAttention(num_heads=params["nb_attn_head_gexpr"], key_dim=params["key_dim_gexpr"])(
                x_gexpr, x_gexpr)
            x_gexpr = tf.squeeze(x_gexpr, 2)
        x_gexpr = Dense(256)(x_gexpr)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        if params["pos_attn_gexpr"] == 1:
            x_gexpr = tf.expand_dims(x_gexpr, -1)
            x_gexpr = MultiHeadAttention(num_heads=params["nb_attn_head_gexpr"], key_dim=params["key_dim_gexpr"])(
                x_gexpr, x_gexpr)
            x_gexpr = tf.squeeze(x_gexpr, 2)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        if params["pos_attn_gexpr"] == 2:
            x_gexpr = tf.expand_dims(x_gexpr, -1)
            x_gexpr = MultiHeadAttention(num_heads=params["nb_attn_head_gexpr"], key_dim=params["key_dim_gexpr"])(
                x_gexpr, x_gexpr)
            x_gexpr = tf.squeeze(x_gexpr, 2)

        #methylation feature
        x_methy = methy_input
        if params["pos_attn_methy"] == 0:
            x_methy = tf.expand_dims(x_methy, -1)
            x_methy = MultiHeadAttention(num_heads=params["nb_attn_head_methy"], key_dim=params["key_dim_methy"])(
                x_methy, x_methy)
            x_methy = tf.squeeze(x_methy, 2)
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        if params["pos_attn_methy"] == 1:
            x_methy = tf.expand_dims(x_methy, -1)
            x_methy = MultiHeadAttention(num_heads=params["nb_attn_head_methy"], key_dim=params["key_dim_methy"])(
                x_methy, x_methy)
            x_methy = tf.squeeze(x_methy, 2)
        x_methy = Dense(100,activation='relu')(x_methy)
        if params["pos_attn_methy"] == 2:
            x_methy = tf.expand_dims(x_methy, -1)
            x_methy = MultiHeadAttention(num_heads=params["nb_attn_head_methy"], key_dim=params["key_dim_methy"])(
                x_methy, x_methy)
            x_methy = tf.squeeze(x_methy, 2)

        x = x_drug
        if self.use_mut:
            x = Multiply()([x, x_mut]) if params["mul"] else Concatenate()([x, x_mut])
        if self.use_gexp:
            x = Multiply()([x, x_gexpr]) if params["mul"] else Concatenate()([x, x_gexpr])
        if self.use_methy:
            x = Multiply()([x, x_methy]) if params["mul"] else Concatenate()([x, x_methy])
        x = Dense(300, activation='tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model = Model(inputs=[drug_feat_input, drug_adj_input, mutation_input, gexpr_input, methy_input],
                      outputs=output)
        return model
