# from sys import path
# path.extend(['/Users/kangzifeng/Desktop/众课如繁星/overseas/summer_intern/work/baseline_link_predict/RelationPrediction-master/code'])
from shared.relation_embedding import RelationEmbedding
from shared.affine_transform import AffineTransform
from link_specific.bilinear_diag import BilinearDiag
from ingredients.graph_representations import Representation

from link_specific.nonlinear_transform import NonlinearTransform
from link_specific.complex import Complex

from shared.bipartite_gcn import BipartiteGcn
from shared.message_gcns.gcn_diag import DiagGcn
from shared.message_gcns.gcn_basis import BasisGcn
from shared.message_gcns.gcn_basis_concat import ConcatGcn
from shared.message_gcns.gcn_basis_stored import BasisGcnStore
from shared.message_gcns.gcn_basis_plus_diag import BasisGcnWithDiag
from shared.message_gcns.gcn_basis_times_diag import BasisGcnTimesDiag

from shared.random_vertex_embedding import RandomEmbedding

from ingredients.residual_layer import ResidualLayer
from ingredients.highway_layer import HighwayLayer
from ingredients.dropover import DropoverLayer

#from ingredients.variational_encoding import VariationalEncoding


def build_shared(shared_settings, triples):
    # if shared_settings['Name'] == "embedding":
    #     input_shape = [int(shared_settings['EntityCount']),
    #                    int(shared_settings['CodeDimension'])]

    #     embedding = AffineTransform(input_shape,
    #                                 shared_settings,
    #                                 onehot_input=True,
    #                                 use_bias=False,
    #                                 use_nonlinearity=False)

    #     full_shared = RelationEmbedding(input_shape,
    #                                      shared_settings,
    #                                      next_component=embedding)

    #     return full_shared

    # if shared_settings['Name'] == "variational_embedding":
    #     input_shape = [int(shared_settings['EntityCount']),
    #                    int(shared_settings['CodeDimension'])]

    #     mu_embedding = AffineTransform(input_shape,
    #                                 shared_settings,
    #                                 onehot_input=True,
    #                                 use_bias=False,
    #                                 use_nonlinearity=False)


    #     sigma_embedding = AffineTransform(input_shape,
    #                                 shared_settings,
    #                                 onehot_input=True,
    #                                 use_bias=False,
    #                                 use_nonlinearity=False)

    #     z = VariationalEncoding(input_shape,
    #                             shared_settings,
    #                             mu_network=mu_embedding,
    #                             sigma_network=sigma_embedding)

    #     full_shared = RelationEmbedding(input_shape,
    #                                      shared_settings,
    #                                      next_component=z)

    #     return full_shared

    # elif shared_settings['Name'] == "gcn_diag":
    #     # Define graph representation:
    #     graph = Representation(triples, shared_settings)

    #     # Define shapes:
    #     input_shape = [int(shared_settings['EntityCount']),
    #                    int(shared_settings['InternalEncoderDimension'])]
    #     internal_shape = [int(shared_settings['InternalEncoderDimension']),
    #                         int(shared_settings['InternalEncoderDimension'])]
    #     projection_shape = [int(shared_settings['InternalEncoderDimension']),
    #                         int(shared_settings['CodeDimension'])]

    #     relation_shape = [int(shared_settings['EntityCount']),
    #                       int(shared_settings['CodeDimension'])]

    #     layers = int(shared_settings['NumberOfLayers'])

    #     # Initial embedding:
    #     encoding = AffineTransform(input_shape,
    #                                 shared_settings,
    #                                 next_component=graph,
    #                                 onehot_input=True,
    #                                 use_bias=True,
    #                                 use_nonlinearity=True)

    #     # Hidden layers:
    #     for layer in range(layers):
    #         use_nonlinearity = layer < layers - 1
    #         encoding = DiagGcn(internal_shape,
    #                            shared_settings,
    #                            next_component=encoding,
    #                            onehot_input=False,
    #                            use_nonlinearity=use_nonlinearity)

    #     # Output transform if chosen:
    #     if shared_settings['UseOutputTransform'] == "Yes":
    #         encoding = AffineTransform(projection_shape,
    #                                    shared_settings,
    #                                    next_component=encoding,
    #                                    onehot_input=False,
    #                                    use_nonlinearity=False,
    #                                    use_bias=True)

    #     # Encode relations:
    #     full_shared = RelationEmbedding(relation_shape,
    #                                      shared_settings,
    #                                      next_component=encoding)

    #     return full_shared

    if shared_settings['Name'] == "gcn_basis":

        # Define graph representation:
        graph = Representation(triples, shared_settings)

        # Define shapes:
        input_shape = [int(shared_settings['EntityCount']),
                       int(shared_settings['InternalEncoderDimension'])]
        internal_shape = [int(shared_settings['InternalEncoderDimension']),
                          int(shared_settings['InternalEncoderDimension'])]
        projection_shape = [int(shared_settings['InternalEncoderDimension']),
                            int(shared_settings['CodeDimension'])]

        relation_shape = [int(shared_settings['EntityCount']),
                          int(shared_settings['CodeDimension'])]

        layers = int(shared_settings['NumberOfLayers'])

        # Initial embedding:
        if shared_settings['UseInputTransform'] == "Yes":
            encoding = AffineTransform(input_shape,
                                       shared_settings,
                                       next_component=graph,
                                       onehot_input=True,
                                       use_bias=True,
                                       use_nonlinearity=True)
        # elif shared_settings['RandomInput'] == 'Yes':
        #     encoding = RandomEmbedding(input_shape,
        #                                shared_settings,
        #                                next_component=graph)
        # elif shared_settings['PartiallyRandomInput'] == 'Yes':
        #     encoding1 = AffineTransform(input_shape,
        #                                shared_settings,
        #                                next_component=graph,
        #                                onehot_input=True,
        #                                use_bias=True,
        #                                use_nonlinearity=False)
        #     encoding2 = RandomEmbedding(input_shape,
        #                                shared_settings,
        #                                next_component=graph)
        #     encoding = DropoverLayer(input_shape,
        #                              next_component=encoding1,
        #                              next_component_2=encoding2)
        else:
            encoding = graph

        # Hidden layers:
        encoding = apply_basis_gcn(shared_settings, encoding, internal_shape, layers)

        # Output transform if chosen:
        if shared_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       shared_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_shared = RelationEmbedding(relation_shape,
                                         shared_settings,
                                         next_component=encoding)

        return full_shared

    # elif shared_settings['Name'] == "variational_gcn_basis":
    #     graph = Representation(triples, shared_settings)

    #     # Define graph representation:
    #     graph = Representation(triples, shared_settings)

    #     # Define shapes:
    #     input_shape = [int(shared_settings['EntityCount']),
    #                    int(shared_settings['InternalEncoderDimension'])]
    #     internal_shape = [int(shared_settings['InternalEncoderDimension']),
    #                       int(shared_settings['InternalEncoderDimension'])]
    #     projection_shape = [int(shared_settings['InternalEncoderDimension']),
    #                         int(shared_settings['CodeDimension'])]

    #     relation_shape = [int(shared_settings['EntityCount']),
    #                       int(shared_settings['CodeDimension'])]

    #     layers = int(shared_settings['NumberOfLayers'])

    #     # Initial embedding:
    #     if shared_settings['UseInputTransform'] == "Yes":
    #         encoding = AffineTransform(input_shape,
    #                                    shared_settings,
    #                                    next_component=graph,
    #                                    onehot_input=True,
    #                                    use_bias=True,
    #                                    use_nonlinearity=True)
    #     else:
    #         encoding = graph

    #     # Hidden layers:
    #     encoding = apply_basis_gcn(shared_settings, encoding, internal_shape, layers)

    #     mu_encoding = AffineTransform(projection_shape,
    #                                    shared_settings,
    #                                    next_component=encoding,
    #                                    onehot_input=False,
    #                                    use_nonlinearity=False,
    #                                    use_bias=True)

    #     sigma_encoding = AffineTransform(projection_shape,
    #                                    shared_settings,
    #                                    next_component=encoding,
    #                                    onehot_input=False,
    #                                    use_nonlinearity=False,
    #                                    use_bias=True)
    #     #mu_encoding = apply_basis_gcn(shared_settings, encoding, internal_shape, layers)
    #     #sigma_encoding = apply_basis_gcn(shared_settings, encoding, internal_shape, layers)

    #     encoding = VariationalEncoding(input_shape,
    #                             shared_settings,
    #                             mu_network=mu_encoding,
    #                             sigma_network=sigma_encoding)

    #     # Output transform if chosen:
    #     if shared_settings['UseOutputTransform'] == "Yes":
    #         encoding = AffineTransform(projection_shape,
    #                                    shared_settings,
    #                                    next_component=encoding,
    #                                    onehot_input=False,
    #                                    use_nonlinearity=False,
    #                                    use_bias=True)

    #     # Encode relations:
    #     full_shared = RelationEmbedding(relation_shape,
    #                                      shared_settings,
    #                                      next_component=encoding)

    #     return full_shared

    else:
        '''
        elif shared_settings['Name'] == "bipartite_gcn":
            graph = Representation(triples, shared_settings, bipartite=True)

            first_layer = BipartiteGcn(shared_settings, graph)
            second_layer = BipartiteGcn(shared_settings, graph, next_component=first_layer)
            third_layer = BipartiteGcn(shared_settings, graph, next_component=second_layer)
            fourth_layer = BipartiteGcn(shared_settings, graph, next_component=third_layer)

            transform = AffineTransform(fourth_layer, shared_settings)

            return RelationEmbedding(transform, shared_settings)
        '''
        return None


def apply_basis_gcn(shared_settings, encoding, internal_shape, layers):
    for layer in range(layers):
        use_nonlinearity = layer < layers - 1

        if layer == 0 \
                and shared_settings['UseInputTransform'] == "No" \
                and shared_settings['RandomInput'] == "No"  \
                and shared_settings['PartiallyRandomInput'] == "No" :
            onehot_input = True
        else:
            onehot_input = False

        if shared_settings['AddDiagonal'] == "Yes":
            model = BasisGcnWithDiag
        elif shared_settings['DiagonalCoefficients'] == "Yes":
            model = BasisGcnTimesDiag
        elif shared_settings['StoreEdgeData'] == "Yes":
            model = BasisGcnStore
        elif 'Concatenation' in shared_settings and shared_settings['Concatenation'] == "Yes":
            model = ConcatGcn
        else:
            model = BasisGcn

        new_encoding = model(internal_shape,
                             shared_settings,
                             next_component=encoding,
                             onehot_input=onehot_input,
                             use_nonlinearity=use_nonlinearity)

        if shared_settings['SkipConnections'] == 'Residual' and onehot_input == False:
            encoding = ResidualLayer(internal_shape, next_component=new_encoding, next_component_2=encoding)
        if shared_settings['SkipConnections'] == 'Highway' and onehot_input == False:
            encoding = HighwayLayer(internal_shape, next_component=new_encoding, next_component_2=encoding)
        else:
            encoding = new_encoding

    return encoding


def build_linkpd(shared, linkspf_settings):
    if linkspf_settings['Name'] == "bilinear-diag":
        return BilinearDiag(shared, linkspf_settings)
    # elif linkspf_settings['Name'] == "complex":
    #     return Complex(int(linkspf_settings['CodeDimension']), linkspf_settings, next_component=shared)
    # elif linkspf_settings['Name'] == "nonlinear-transform":
    #     return NonlinearTransform(shared, linkspf_settings)
    else:
        return None

def build_nodecf(shared, node_settings):
    return Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging)

def build_merge(model1, model2, node_settings):
    return MergeLayer([model1, model2], node_settings);
