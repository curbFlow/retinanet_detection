import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Simple script to convert a detector model to onnx')

    parser.add_argument('--model_path', help='h5 file of the saved detector checkpoint', default='model.h5', type=str)

    parser.add_argument('--pb_out_path', help='Where to save the pb out path', default='model.pb', type=str)

    parser.add_argument('--onnx_out_path', help='Where to save the onnx output file', default='model.onnx', type=str)

    return parser.parse_args()




def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    print ([n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node])
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph





if __name__ == '__main__':
    args = parse_args()
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from evaluate.evalute import ResnetDetector

    K.set_learning_phase(0)

    # load the ResNet-18 detector model
    model = ResnetDetector(model_path=args.model_path)
    sess = tf.compat.v1.Session()
    # Convert the Keras ResNet-18 model to a .pb file
    frozen_graph = freeze_session(sess,
                                  output_names=[out.op.name for out in model.model.outputs])

    # tf2onnx_command = f'python -m tf2onnx.convert  --input {args.pb_out_path} --inputs {in_tensor_name} --outputs {out_tensor_names} --output {args.onnx_out_path}'
    # os.system(tf2onnx_command)
