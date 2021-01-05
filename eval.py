import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Simple evaluation script for object detection.')

    parser.add_argument(
        '--eval_csv', help='Eval CSV File', default='val.csv', type=str)
    parser.add_argument(
        '--model_path', help='Path to the H5 model to use for evaluation', default='model.h5', type=str)

    parser.add_argument('--classes_path', help='classes CSV File', default='classes.csv', type=str)

    parser.add_argument('--save_annotations', help='Whether or not to save the annotations', default=False, type=bool)
    parser.add_argument('--save_path', help='Path where to save the annotations', default='output_frames', type=str)
    parser.add_argument('--score_thresh', help='Classification score threshold for each bbox', default=0.5, type=float)
    parser.add_argument('--bbox_thresh', help='Threshold for a box to be considered as a correct detection',
                        default=0.5, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    from evaluate.evalute import RetinanetEval

    # parse arguments
    args = parse_args()

    evaluator = RetinanetEval(
        csv_path=args.eval_csv,
        model_path=args.model_path,
        classes_csv=args.classes_path, score_thresh=args.score_thresh, bbox_thresh=args.bbox_thresh)

    evaluator.evaluate_on_dataset()
    evaluator.write_annotated_images(args.save_path)
