import tensorflow as tf
import keras
import keras.backend as K

from caloGraphNN import euclidean_squared, gauss, gauss_of_lin

class CreateZeroMask(Layer):
    '''
    Creates a mask based on the 0th index of the vertex
    To apply, use keras.Layers.Multiply
    '''
    def __init__(self, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],1)
    
    def call(self, inputs):
        zeros = tf.zeros(shape=tf.shape(inputs)[:-1])
        mask = tf.where(inputs[:,:,0]>0, zeros+1., zeros)
        mask = tf.expand_dims(mask,axis=2)
        return mask
    
    def get_config(self):
        #config = {'my_configoption': self.my_configoption}
        base_config = super(CreateZeroMask, self).get_config()
        return dict(list(base_config.items())) # + list(config.items() ))
      


class GlobalExchange(keras.layers.Layer):
    def __init__(self, vertex_mask=None, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

        self.vertex_mask = vertex_mask

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        mean = tf.tile(mean, [1, self.num_vertices, 1])
        if self.vertex_mask is not None:
            mean = self.vertex_mask * mean

        return tf.concat([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)


class GravNet(keras.layers.Layer):
    def __init__(self, n_neighbours, n_dimensions, n_filters, n_propagate, name, 
                 also_coordinates=False, feature_dropout=-1, 
                 coordinate_kernel_initializer=keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False, 
                 coordinate_activation=None,
                 masked_coordinate_offset=None,
                 **kwargs):
        super(GravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset
        
        self.input_feature_transform = keras.layers.Dense(n_propagate, name = name+'_FLR', kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = keras.layers.Dense(n_dimensions, name = name+'_S', kernel_initializer=coordinate_kernel_initializer, activation=coordinate_activation)
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh', name = name+'_Fout', kernel_initializer=other_kernel_initializer)

        self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [self.input_feature_transform, self.output_feature_transform]

    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
            
        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)
        
        # tf.ragged FIXME?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
        super(GravNet, self).build(input_shape)

    def call(self, x):
        
        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception('GravNet: in mask mode, input must be list of input,mask')
            mask = x[1]
            x = x[0]
            
        features = self.input_feature_transform(x)
        
        if self.feature_dropout>0 and self.feature_dropout < 1:
            features = keras.layers.Dropout(self.feature_dropout)(features)
        
        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:,:,0:self.n_dimensions]
            
        if self.masked_coordinate_offset is not None:
            sel_mask = tf.tile(mask, [1,1,tf.shape(coordinates)[2]])
            coordinates = tf.where(sel_mask>0., coordinates, tf.zeros_like(coordinates)-self.masked_coordinate_offset)

        collected_neighbours = self.collect_neighbours(coordinates, features)

        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)
        
        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output
        
    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [(input_shape[0], input_shape[1], self.output_feature_transform.units),
                    (input_shape[0], input_shape[1], self.n_dimensions)]
        
        # tf.ragged FIXME? tf.shape() might do the trick already
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, features):
        
        # tf.ragged FIXME?
        # for euclidean_squared see caloGraphNN.py
        distance_matrix = euclidean_squared(coordinates, coordinates)

        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]

        n_batches = tf.shape(features)[0]
        
        # tf.ragged FIXME? or could that work?
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
        neighbour_features = tf.gather_nd(features, indices) # (B, V, N-1, F)
    
        distance = -ranked_distances[:, :, 1:]
    
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
    
        # weight the neighbour_features
        neighbour_features *= weights
    
        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
    
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate,
                  'name':self.name,
                  'also_coordinates': self.also_coordinates,
                  'feature_dropout' : self.feature_dropout,
                  'masked_coordinate_offset'       : self.masked_coordinate_offset}
        base_config = super(GravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, name, vertex_mask=None, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name

        self.vertex_mask = vertex_mask

        self.input_feature_transform = keras.layers.Dense(n_propagate, name=name+'_FLR')
        self.aggregator_distance = keras.layers.Dense(n_aggregators, name=name+'_S')
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh', name=name+'_Fout')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)
        
        # tf.ragged FIXME? tf.shape()?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.aggregator_distance.units + 2 * self.aggregator_distance.units * (self.input_feature_transform.units + self.aggregator_distance.units)))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(GarNet, self).build(input_shape)

    def call(self, x):
        features = self.input_feature_transform(x) # (B, V, F)
        distance = self.aggregator_distance(x) # (B, V, S)

        edge_weights = gauss(distance)

        features = tf.concat([features, edge_weights], axis=-1) # (B, V, F+S)

        if self.vertex_mask is not None:
            features = self.vertex_mask * features
            edge_weights = self.vertex_mask * edge_weights

        # vertices -> aggregators
        edge_weights_trans = tf.transpose(edge_weights, perm=(0, 2, 1)) # (B, S, V)
        aggregated_max = self.apply_edge_weights(features, edge_weights_trans, aggregation=tf.reduce_max) # (B, S, F+S)
        aggregated_mean = self.apply_edge_weights(features, edge_weights_trans, aggregation=tf.reduce_mean) # (B, S, F+S)

        aggregated = tf.concat([aggregated_max, aggregated_mean], axis=-1) # (B, S, 2*(F+S))

        # aggregators -> vertices
        updated_features = self.apply_edge_weights(aggregated, edge_weights) # (B, V, 2*S*(F+S))

        updated_features = tf.concat([x, updated_features, edge_weights], axis=-1) # (B, V, X+2*S*(F+S)+S)

        return self.output_feature_transform(updated_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def apply_edge_weights(self, features, edge_weights, aggregation=None):
        features = tf.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = tf.expand_dims(edge_weights, axis=3) # (B, A, v, 1)

        # tf.ragged FIXME? broadcasting should work
        out = edge_weights * features # (B, u, v, f)
        # tf.ragged FIXME? these values won't work
        n = features.shape[-2].value * features.shape[-1].value

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
            n = features.shape[-1].value
        
        # tf.ragged FIXME? there might be a chance to spell out batch dim instead and use -1 for vertices?
        return tf.reshape(out, [-1, out.shape[1].value, n]) # (B, u, n)
    
    def get_config(self):
        config = {'n_aggregators': self.n_aggregators, 'n_filters': self.n_filters, 'n_propagate': self.n_propagate, 'name': self.name}
        base_config = super(GarNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
# tf.ragged FIXME? the last one should be no problem
class weighted_sum_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(weighted_sum_layer, self).__init__(**kwargs)
        
    def get_config(self):
        base_config = super(weighted_sum_layer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape[2] > 1
        inshape=list(input_shape)
        return tuple((inshape[0],input_shape[2]-1))
    
    def call(self, inputs):
        # input #B x E x F
        weights = inputs[:,:,0:1] #B x E x 1
        tosum   = inputs[:,:,1:]
        weighted = weights * tosum #broadcast to B x E x F-1
        return tf.reduce_sum(weighted, axis=1)    
    
    
    
    
    
    
    
    
