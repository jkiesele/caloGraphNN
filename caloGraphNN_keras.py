import tensorflow as tf
import keras

from caloGraphNN import euclidean_squared, gauss, gauss_of_lin

class GlobalExchange(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (None, V, F)
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        mean = tf.tile(mean, [1, self.num_vertices, 1])
        return tf.concat([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)


class GravNet(keras.layers.Layer):
    def __init__(self, n_neighbours, n_dimensions, n_filters, n_propagate, **kwargs):
        super(GravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        
        self.input_feature_transform = keras.layers.Dense(n_propagate)
        self.input_spatial_transform = keras.layers.Dense(n_dimensions)
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh')

        self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.input_spatial_transform.build(input_shape)
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)
        
        super(GravNet, self).build(input_shape)

    def call(self, x):
        features = self.input_feature_transform(x)
        coordinates = self.input_spatial_transform(x)

        collected_neighbours = self.collect_neighbours(coordinates, features)

        updated_features = tf.concat([x, collected_neighbours], axis=-1)

        return self.output_feature_transform(updated_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours_fullmatrix(self, coordinates, features):
        # implementation changed wrt caloGraphNN to account for batch size (B) being unknown (None)
        # V = number of vertices
        # N = number of neighbours
        # F = number of features per vertex
    
        # distance_matrix is the actual (B, V, V) matrix
        distance_matrix = euclidean_squared(coordinates, coordinates)
        _, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]
    
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]
    
        # make a boolean mask of the neighbours (B, V, N-1)
        neighbour_mask = tf.one_hot(neighbour_indices, depth=n_vertices, axis=-1, dtype=tf.int32)
        neighbour_mask = tf.reduce_sum(neighbour_mask, axis=2)
        neighbour_mask = tf.cast(neighbour_mask, tf.bool)

        # (B, V, F) -[tile]> (B, V, V, F) -[mask]> (B, V, N-1, F)
        neighbour_features = tf.expand_dims(features, axis=1)
        neighbour_features = tf.tile(neighbour_features, [1, n_vertices, 1, 1])
        neighbour_features = tf.boolean_mask(neighbour_features, neighbour_mask)
        neighbour_features = tf.reshape(neighbour_features, [-1, n_vertices, self.n_neighbours - 1, n_features])

        # (B, V, V) -[mask]> (B, V, N-1)
        distance = tf.boolean_mask(distance_matrix, neighbour_mask)
        distance = tf.reshape(distance, [-1, n_vertices, self.n_neighbours - 1])
    
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
    
        # weight the neighbour_features
        neighbour_features *= weights
    
        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
    
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)

    def collect_neighbours(self, coordinates, features):
        # V = number of vertices
        # N = number of neighbours
        # F = number of features per vertex
    
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]

        n_batches = tf.shape(features)[0]
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
        neighbour_features = tf.gather_nd(features, indices) # (B, V, N-1, F)
    
        distance = ranked_distances[:, :, 1:]
    
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
    
        # weight the neighbour_features
        neighbour_features *= weights
    
        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
    
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
            config = {'n_neighbours': self.n_neighbours, 'n_dimensions': self.n_dimensions, 'n_filters': self.n_filters, 'n_propagate': self.n_propagate}
            base_config = super(GravNet, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        self.input_feature_transform = keras.layers.Dense(n_propagate)
        self.aggregator_distance = keras.layers.Dense(n_aggregators)
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)
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
        edge_weights = tf.expand_dims(edge_weights, axis=3) # (B, u, v, 1)

        out = edge_weights * features # (B, u, v, f)
        n = features.shape[-2].value * features.shape[-1].value

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
            n = features.shape[-1].value
        
        return tf.reshape(out, [-1, out.shape[1].value, n]) # (B, u, n)
    
    def get_config(self):
            config = {'n_aggregators': self.n_aggregators, 'n_filters': self.n_filters, 'n_propagate': self.n_propagate}
            base_config = super(GarNet, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    
    
    
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
    
    
    
    
    
    
    
    
