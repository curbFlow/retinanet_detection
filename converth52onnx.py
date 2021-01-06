import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Simple evaluation script for object detection.')

    parser.add_argument(
        '--h5_path', help='Path to the H5 model to use for evaluation', default='model.h5', type=str)

    parser.add_argument('--onnx_path', help='Path where to save the onnx model', default='model.onnx', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    from evaluate.evalute import ResnetDetector
    import keras2onnx
    import onnx

    # parse arguments
    args = parse_args()

    detector = ResnetDetector(model_path=args.h5_path)
    model = detector.model
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save(onnx_model, args.onnx_path)
